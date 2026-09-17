//! A digest of every weight buffer after binding, so two ways of getting
//! the same checkpoint into a runtime (files, a weight cache) can be
//! compared byte for byte:
//!
//!   weights_digest --manifest m.json --kernels dir --weights <entry>... [--gpu 0] [--rank ep=0/4 ...]
//!
//! One line per weight buffer: name, bytes digested, sha256. A buffer
//! larger than `--limit` GiB (default 4; the host-placed tables) digests
//! its first `--limit` GiB.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use kern_manifest::types::BufferKind;
use kern_manifest::Verified;
use kern_run::Weights;
use kern_runtime::{Capacity, GroupRank, Runtime, Topology};
use sha2::{Digest, Sha256};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut manifest, mut kernels, mut weights, mut gpu, mut limit) = (None, None, Vec::new(), 0usize, 4u64);
    let mut topo = Topology::default();
    while let Some(a) = args.next() {
        let mut v = || args.next().context("flag needs a value");
        match a.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(v()?)),
            "--kernels" => kernels = Some(PathBuf::from(v()?)),
            "--weights" => weights.push(v()?),
            "--gpu" => gpu = v()?.parse()?,
            "--limit" => limit = v()?.parse()?,
            "--rank" => {
                let s = v()?;
                let (group, place) = s.split_once('=').context("--rank group=index/size")?;
                let (index, size) = place.split_once('/').context("--rank group=index/size")?;
                topo.groups.insert(group.into(), GroupRank { index: index.parse()?, size: size.parse()? });
            }
            other => bail!("unknown flag {other}"),
        }
    }
    let manifest = std::fs::read_to_string(manifest.context("--manifest")?)?;
    let verified = Verified::from_json(&manifest)?;
    let has_topology = verified.topology.is_some();
    let capacity = Some(Capacity { tokens: Some(kern_pool::page_unit(&verified)), seqs: 1 });
    let mut rt =
        Runtime::load(&verified, &kernels.context("--kernels")?, gpu, capacity, has_topology.then_some(&topo))?;
    Weights::parse(&weights)?.bind(&mut rt, &topo)?;
    let names: Vec<(String, u64)> = rt
        .buffer_sizes()
        .into_iter()
        .filter(|(_, k, _)| *k == BufferKind::Weight)
        .map(|(n, _, b)| (n.to_string(), b))
        .collect();
    for (name, bytes) in names {
        let take = bytes.min(limit << 30) as usize;
        let data = rt.read_buffer_prefix(&name, take)?;
        println!("{name} {take} {}", hex::encode(Sha256::digest(&data)));
    }
    Ok(())
}
