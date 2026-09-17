//! End-to-end mapped immutable weights: original safetensors rectangle, stable
//! precompiled pointers, execution guards, CUDA graph reads, and shared lifetime.

use std::collections::BTreeMap;

use half::bf16;
use kern_manifest::Verified;
use kern_runtime::{Capacity, HostWeights, Runtime, Safetensors};

fn checkpoint() -> Vec<u8> {
    let mut header = serde_json::json!({
        "table": {"dtype": "BF16", "shape": [32, 32], "data_offsets": [0, 2048]}
    })
    .to_string()
    .into_bytes();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend(header);
    for i in 0..1024 {
        data.extend(bf16::from_f32((i % 17) as f32 - 8.0).to_le_bytes());
    }
    data
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn mapped_weight_rectangle_replay_shared_scope_and_guards() {
    let manifest = serde_json::json!({
        "schema_version": 5, "model": "host-weight-test", "vars": {}, "states": {},
        "buffers": {
            "a": {"kind": "input", "dtype": "bf16", "shape": [16, 16]},
            "w": {"kind": "weight", "placement": "host", "dtype": "bf16", "shape": [16, 16],
                  "bind": [{"tensor": "table", "rows": [8, 24], "cols": [7, 23]}]},
            "out": {"kind": "output", "dtype": "bf16", "shape": [16, 16]}
        },
        "modules": {},
        "ops": {"gemm": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}
        }},
        "programs": {"probe": {"calls": [{"op": "gemm", "args": [
            {"buf": "a"}, {"buf": "w"}, {"buf": "out"}, {"i32": 16}, {"i32": 16}, {"i32": 16}
        ]}]}}
    });
    let verified = Verified::from_json(&manifest.to_string()).unwrap();
    let dir = std::env::temp_dir().join(format!("kern-host-weight-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scope = HostWeights::new();
    let capacity = Some(Capacity { tokens: Some(1), seqs: 1 });
    let mut first = Runtime::load_with_host_weights(&verified, &dir, 0, capacity, None, &scope).unwrap();
    let mut second = Runtime::load_with_host_weights(&verified, &dir, 0, capacity, None, &scope).unwrap();
    assert_eq!(scope.allocated_bytes().unwrap(), 512);
    let vars = BTreeMap::new();
    assert!(first.run("probe", &vars).is_err());
    assert!(first.capture("probe", &vars).is_err());
    let data = checkpoint();
    let tensors = Safetensors::parse(&[&data]).unwrap();
    first.load_weights(&tensors).unwrap();
    second.load_weights(&tensors).unwrap();
    assert!(first.load_weights(&tensors).is_err());
    let identity: Vec<u8> =
        (0..256).flat_map(|i| bf16::from_f32(if i / 16 == i % 16 { 1.0 } else { 0.0 }).to_le_bytes()).collect();
    let expected: Vec<u8> = (0..256)
        .flat_map(|i| bf16::from_f32((((8 + i % 16) * 32 + 7 + i / 16) % 17) as f32 - 8.0).to_le_bytes())
        .collect();
    first.write_input("a", &identity).unwrap();
    second.write_input("a", &identity).unwrap();
    first.run("probe", &vars).unwrap();
    assert_eq!(first.read_output("out").unwrap(), expected);
    drop(first);
    drop(scope);
    second.capture("probe", &vars).unwrap();
    second.run_captured("probe", &vars).unwrap();
    assert_eq!(second.read_output("out").unwrap(), expected);
    drop(second);
    std::fs::remove_dir_all(dir).unwrap();
}
