//! The tap's workload, fully determined by `(seed, manifest, options)`:
//! anyone can re-run it, and it does not depend on A's numerics (decode
//! tokens are drawn, not A's argmax).

use anyhow::Result;
use kern_manifest::types::{Manifest, Provision};
use kern_manifest::Protocol;

use crate::Options;

/// splitmix64: a seed is the whole generator.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn draw(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        (self.draw() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.draw() % n.max(1)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Workload {
    pub prefill: Vec<i64>,
    pub chunk: usize,
    pub decode: Vec<i64>,
    pub vocab: u64,
    /// How the prefill length was chosen.
    pub how: &'static str,
}

/// Draw the workload. Prefill length: the prompt's, `--prefill`, or from
/// the seed half the time uniform in `[1, capacity − steps]` and half the
/// time a structural boundary (a page, a chunk, the rows max, ±1) — where
/// kernels break.
pub fn sample(o: &Options, m: &Manifest, pr: &Protocol, p: Provision, page: u64) -> Result<Workload> {
    let capacity = p.tokens;
    let mut rng = Rng(o.seed ^ 0x776f_726b_6c6f_6164);
    let tmax = pr.rows.max.max(1);
    let tokens = &pr.token_rows().name;
    let vocab = m.buffers[tokens]
        .domain
        .as_ref()
        .map(|d| d.resolve(m, &pr.vars(1, 1, 1), &p))
        .transpose()?
        .and_then(|r| r.hi)
        .map(|hi| hi as u64 + 1)
        .ok_or_else(|| anyhow::anyhow!("`{tokens}` has no domain to take the vocabulary from"))?;
    let steps =
        if o.decode_steps <= 1 { 1 } else { o.decode_steps / 2 + rng.below(o.decode_steps - o.decode_steps / 2 + 1) };
    let hi = capacity.saturating_sub(steps).max(1);
    let chunk = if o.chunk > 0 {
        o.chunk
    } else {
        [tmax, 512.min(tmax), page.min(tmax), 1 + rng.below(tmax)][rng.below(4) as usize]
    }
    .clamp(1, tmax);
    let (n_pre, how) = match &o.prompt {
        _ if pr.chunk().is_none() => (0, "no chunk program"),
        Some(p) => {
            anyhow::ensure!(
                p.len() as u64 <= hi,
                "prompt is {} tokens; capacity {capacity} leaves room for {hi} before {steps} decode steps",
                p.len()
            );
            (p.len() as u64, "prompt")
        }
        None if o.prefill > 0 => (o.prefill.min(hi), "given"),
        None if rng.below(2) == 0 => (1 + rng.below(hi), "uniform"),
        None => {
            let mut c: Vec<u64> = vec![
                1,
                page - 1,
                page,
                page + 1,
                2 * page + 1,
                chunk - 1,
                chunk,
                chunk + 1,
                2 * chunk + 1,
                3 * chunk - 1,
                tmax,
                tmax + 1,
                hi,
            ];
            c.retain(|&x| (1..=hi).contains(&x));
            c.sort_unstable();
            c.dedup();
            (c[rng.below(c.len() as u64) as usize], "boundary")
        }
    };
    let prefill = o.prompt.clone().unwrap_or_else(|| (0..n_pre).map(|_| rng.below(vocab) as i64).collect());
    let decode = (0..steps).map(|_| rng.below(vocab) as i64).collect();
    Ok(Workload { prefill, chunk: chunk as usize, decode, vocab, how })
}
