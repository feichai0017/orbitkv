//! `kern run --probe-dir`: the activations of a run, for comparing a
//! manifest against a reference implementation outside kern.
//!
//! A program is run call range by call range, cut after every call whose
//! label the caller asked for, so nothing executes twice and the buffer
//! that call wrote can be read before the next range overwrites it.

use anyhow::{bail, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::{Arg, Dim, Dir};
use kern_runtime::Runtime;
use tracing::info;

use crate::{Caller, Vars};

/// The first prefill chunk and `steps` decode steps: after every call
/// whose label equals or ends with one of `labels`, the buffer that call
/// writes (live rows) lands in `dir` as `<tag>.<point>.bin`, `point`
/// being the label minus its last `.part`; then the step's logits (the
/// buffer its `tokens` output is taken from) and the tokens.
pub(crate) fn probe(
    caller: &mut Caller,
    prompt_ids: &[i64],
    dir: &std::path::Path,
    labels: &str,
    chunk: u64,
    step: &Forward,
    steps: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let labels: Vec<&str> = labels.split(',').filter(|p| !p.is_empty()).collect();
    let matches = |l: &str| labels.iter().any(|p| l == *p || l.ends_with(p));
    let row_bytes = |rt: &Runtime, name: &str| -> usize {
        let b = &rt.manifest.buffers[name];
        b.shape[1..]
            .iter()
            .map(|d| match d {
                Dim::Const(c) => *c as usize,
                _ => 1,
            })
            .product::<usize>()
            * b.dtype.bytes() as usize
    };
    // The buffer a call reads or writes through its first param of `dir`.
    let param_buf = |rt: &Runtime, c: &kern_manifest::types::Call, want: &[Dir]| -> Option<String> {
        let op = &rt.manifest.ops[&c.op];
        c.args.iter().zip(&op.params).find_map(|(a, p)| match (a, p.dir()) {
            (Arg::Buf { buf, .. }, Some(d)) if want.contains(&d) => Some(buf.clone()),
            _ => None,
        })
    };
    let run_probed = |caller: &Caller, f: &Forward, vars: &Vars, rows: usize, tag: &str| -> Result<()> {
        let rt = &caller.rt;
        let calls = &rt.manifest.programs[&f.name].calls;
        let mut lo = 0;
        for (i, c) in calls.iter().enumerate() {
            let l = c.label.clone().unwrap_or_default();
            if !matches(&l) {
                continue;
            }
            let Some(bufname) = param_buf(rt, c, &[Dir::Out, Dir::InOut]) else { continue };
            rt.run_range(&f.name, vars, lo, i + 1)?;
            lo = i + 1;
            let n = match rt.manifest.buffers[&bufname].shape[0] {
                Dim::Const(c) => c as usize,
                _ => rows,
            };
            let point = l.rsplit_once('.').map_or(l.as_str(), |(head, _)| head);
            let data = rt.read_buffer_prefix(&bufname, n * row_bytes(rt, &bufname))?;
            std::fs::write(dir.join(format!("{tag}.{point}.bin")), data)?;
        }
        rt.run_range(&f.name, vars, lo, calls.len())?;
        if let Some(i) = f.emits {
            let tokens = &caller.protocol.fills[i];
            if let Some(logits) = calls
                .iter()
                .rev()
                .find(|c| param_buf(rt, c, &[Dir::Out, Dir::InOut]).as_deref() == Some(&tokens.name))
                .and_then(|c| param_buf(rt, c, &[Dir::In]))
            {
                std::fs::write(dir.join(format!("{tag}.logits.bin")), rt.read_buffer(&logits)?)?;
            }
            std::fs::write(dir.join(format!("{tag}.tokens.bin")), rt.read_output(&tokens.name)?)?;
        }
        Ok(())
    };
    let chunk_f = caller.chunk_forward()?;
    let chunk = chunk as usize;
    let n_pre = caller.prefill_len(prompt_ids.len())?;
    let c = n_pre.min(chunk);
    let e = caller.stage(&prompt_ids[..c])?;
    run_probed(caller, &chunk_f, &e, c, "chunk")?;
    caller.advance(c as u64);
    let mut first = caller.emitted(&chunk_f)?.first().copied();
    if c < n_pre {
        first = caller.prefill(&prompt_ids[c..n_pre], chunk as u64)?;
    }
    let mut tok = match first {
        Some(t) => t,
        None => prompt_ids[n_pre],
    };
    let rows = match step.rows {
        Rows::Const(r) => r,
        Rows::Var => bail!("the step program takes rows as fed; probe steps need a fixed-rows program"),
    };
    for s in 0..steps {
        let e = caller.stage_rows(tok, rows)?;
        run_probed(caller, step, &e, rows as usize, &format!("decode{s}"))?;
        let out = caller.emitted(step)?;
        caller.advance(out.len() as u64);
        tok = *out.last().unwrap();
    }
    info!("probe: wrote activations to {}", dir.display());
    Ok(())
}
