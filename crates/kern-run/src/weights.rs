//! Where a target's weights are. A `--weights` entry is a checkpoint
//! directory (every `*.safetensors` in it, in name order) or one
//! `.safetensors` file; the tensors of every entry bind by name.
//!
//! A manifest with a topology may name a rank's shard in a file entry:
//! `{ep}` / `{tp}` stand for the rank's index in that group and `*` in a
//! file name matches the files around it, so one entry names every rank's
//! shard.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use kern_runtime::{Runtime, Safetensors, Topology};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Weights {
    Files(Vec<PathBuf>),
}

impl Weights {
    pub fn parse(entries: &[String]) -> Result<Weights> {
        if entries.is_empty() {
            bail!("no weights");
        }
        Ok(Weights::Files(entries.iter().map(PathBuf::from).collect()))
    }

    /// The checkpoint directories, where the tokenizer and the model's
    /// config live: each entry's (the directory itself, or the one holding
    /// the file or the rank's shards).
    pub fn dirs(&self) -> Vec<PathBuf> {
        match self {
            Weights::Files(paths) => paths.iter().map(|p| checkpoint_dir(p)).collect(),
        }
    }

    /// Bind the weights into `rt` for its place in the topology.
    pub fn bind(&self, rt: &mut Runtime, topo: &Topology) -> Result<()> {
        match self {
            Weights::Files(paths) => {
                Ok(rt.load_weights(&Safetensors::open(&crate::shard_files(&rank_files(paths, topo)?)?)?)?)
            }
        }
    }
}

impl std::fmt::Display for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Weights::Files(paths) => {
                write!(f, "{}", paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" + "))
            }
        }
    }
}

/// The checkpoint directory a file entry names: the entry up to its first
/// per-rank component (`{group}` or `*`), less a `.safetensors` file name.
fn checkpoint_dir(w: &Path) -> PathBuf {
    let s = |c: &std::path::Component| c.as_os_str().to_string_lossy().into_owned();
    let fixed: Vec<String> = w.components().map(|c| s(&c)).take_while(|c| !c.contains(['{', '*'])).collect();
    let dir = fixed.iter().take(fixed.len() - usize::from(fixed.last().is_some_and(|f| f.ends_with(".safetensors"))));
    let d: PathBuf = dir.collect();
    if d.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        d
    }
}

/// One rank's files: every `{group}` in an entry is the rank's index in
/// that topology group, and a `*` in a file name matches that directory's
/// files around it, in name order.
fn rank_files(paths: &[PathBuf], topo: &Topology) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        let mut s = p.to_string_lossy().into_owned();
        for (g, r) in &topo.groups {
            s = s.replace(&format!("{{{g}}}"), &r.index.to_string());
        }
        if let Some(open) = s.find('{') {
            let close = s[open..].find('}').map_or(s.len(), |c| open + c + 1);
            bail!("weights path {s}: `{}` is not a group of the manifest's topology", &s[open..close]);
        }
        let p = PathBuf::from(&s);
        let Some((pre, post)) = p.file_name().and_then(|f| f.to_str()).and_then(|f| f.split_once('*')) else {
            out.push(p);
            continue;
        };
        let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .with_context(|| format!("weights {s}: listing {}", dir.display()))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|f| f.len() >= pre.len() + post.len() && f.starts_with(pre) && f.ends_with(post))
            .collect();
        if names.is_empty() {
            bail!("weights {s}: nothing matches");
        }
        names.sort_unstable();
        out.extend(names.into_iter().map(|f| dir.join(f)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_runtime::GroupRank;

    fn entries(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn entries_are_files() {
        assert_eq!(
            Weights::parse(&entries(&["a", "b/c.safetensors"])).unwrap(),
            Weights::Files(vec!["a".into(), "b/c.safetensors".into()])
        );
        assert!(Weights::parse(&[]).unwrap_err().to_string().contains("no weights"));
        assert_eq!(Weights::parse(&entries(&["a", "b"])).unwrap().to_string(), "a + b");
    }

    #[test]
    fn checkpoint_dir_is_the_entry_up_to_its_per_rank_part() {
        let d = |s: &str| checkpoint_dir(Path::new(s));
        assert_eq!(
            (d("weights/Qwen3-4B"), d("weights/Qwen3-4B/model-00001.safetensors"), d("dense-tp4/r{tp}/l*.safetensors")),
            ("weights/Qwen3-4B".into(), "weights/Qwen3-4B".into(), "dense-tp4".into())
        );
        assert_eq!(d("model.safetensors"), PathBuf::from("."));
    }

    #[test]
    fn rank_files_substitute_groups_and_expand_stars() {
        let dir = std::env::temp_dir().join(format!("kern-run-weights-{}", std::process::id()));
        let shard = dir.join("dense-tp4").join("r2");
        std::fs::create_dir_all(&shard).unwrap();
        for f in ["l10.safetensors", "l1.safetensors", "l0.safetensors", "notes.txt"] {
            std::fs::write(shard.join(f), b"").unwrap();
        }
        let mut topo = Topology::one("ep", 3, 4);
        topo.groups.insert("tp".into(), GroupRank { index: 2, size: 4 });
        let paths = [dir.join("bookends.safetensors"), dir.join("dense-tp4/r{tp}/l*.safetensors")];
        let got = rank_files(&paths, &topo).unwrap();
        // Name order, not layer order: the runtime binds by tensor name.
        let want = [
            dir.join("bookends.safetensors"),
            shard.join("l0.safetensors"),
            shard.join("l1.safetensors"),
            shard.join("l10.safetensors"),
        ];
        assert_eq!(got, want);
        let e = rank_files(&[dir.join("experts/ep{world}-r{ep}.safetensors")], &topo).unwrap_err();
        assert!(e.to_string().contains("`{world}` is not a group"), "{e}");
        let e = rank_files(&[shard.join("x*.safetensors")], &topo).unwrap_err();
        assert!(e.to_string().contains("nothing matches"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
