//! What every command resolves its inputs to: the flag, else the target
//! in `kern.toml`, else the checkpoint's own files, else a default.

use std::path::{Path, PathBuf};

use kern_run::config::Config;
use kern_run::{Given, Inputs, Weights};

/// A kern.toml with one target, beside a checkpoint dir that declares a
/// tokenizer and two eos ids.
fn fixture(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("kern-inputs-{}-{name}", std::process::id()));
    let ck = root.join("ckpt");
    std::fs::create_dir_all(&ck).unwrap();
    std::fs::write(ck.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(ck.join("generation_config.json"), r#"{"eos_token_id": [11, 13]}"#).unwrap();
    std::fs::write(ck.join("model.safetensors"), "").unwrap();
    std::fs::write(
        root.join("kern.toml"),
        r#"
gpu = 3

[targets.t]
manifest  = "b.json"
reference = "a.json"
kernels   = "cubins"
weights   = ["ckpt"]
"#,
    )
    .unwrap();
    root
}

fn resolve(root: &Path, given: Given) -> Inputs {
    let cfg = Config::find(Some(&root.join("kern.toml"))).unwrap().unwrap();
    let t = cfg.one(Some("t")).unwrap().1.clone();
    Inputs::resolve(given, Some(&cfg), Some(&t)).unwrap()
}

#[test]
fn the_target_gives_what_no_flag_did() {
    let root = fixture("target");
    let i = resolve(&root, Given::default());
    assert_eq!((&i.manifest, &i.reference), (&root.join("b.json"), &Some(root.join("a.json"))));
    assert_eq!((&i.kernels, &i.weights), (&root.join("cubins"), &Weights::Files(vec![root.join("ckpt")])));
    // nothing named a tokenizer or a stop token: the checkpoint's own
    assert_eq!((i.tokenizer.clone(), i.stop_tokens.clone()), (Some(root.join("ckpt/tokenizer.json")), vec![11, 13]));
    assert_eq!(i.gpu, 3);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_flag_beats_the_target_and_the_checkpoint() {
    let root = fixture("flags");
    let i = resolve(
        &root,
        Given {
            manifest: Some("/m.json".into()),
            kernels: Some("/k".into()),
            weights: vec![root.join("ckpt").display().to_string()],
            tokenizer: Some("/tok.json".into()),
            stop_tokens: vec![13, 7],
            gpu: Some(1),
            ..Default::default()
        },
    );
    assert_eq!((&i.manifest, &i.kernels, i.gpu), (&PathBuf::from("/m.json"), &PathBuf::from("/k"), 1));
    assert_eq!(i.tokenizer, Some(PathBuf::from("/tok.json")));
    // the checkpoint's ids first, then the flag's, each once
    assert_eq!(i.stop_tokens, vec![11, 13, 7]);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn an_input_nobody_gave_names_the_kern_toml() {
    let root = fixture("missing");
    let cfg = Config::find(Some(&root.join("kern.toml"))).unwrap().unwrap();
    let e = Inputs::resolve(Given::default(), Some(&cfg), None).unwrap_err().to_string();
    assert!(e.contains("no --weights") && e.contains("kern.toml"), "{e}");
    let e =
        Inputs::resolve(Given { weights: vec!["x".into()], ..Default::default() }, None, None).unwrap_err().to_string();
    assert!(e.contains("no --manifest") && e.contains("no kern.toml found"), "{e}");
    std::fs::remove_dir_all(&root).unwrap();
}
