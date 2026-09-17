//! The verifier, driven the way a load does: a manifest as JSON in,
//! `Verified` or every diagnostic out.

use std::collections::BTreeMap;

use kern_manifest::types::*;
use kern_manifest::{verify, Manifest, Verified, VerifyErrors};
/// The base fixture deliberately exercises the impl machinery: `embed`
/// is the minimal single-launch op (ABI = interface, wiring defaulted);
/// `attn` is a two-launch implementation with a private scratch buffer.
const BASE: &str = r#"{
  "schema_version": 5, "model": "toy",
  "vars": { "tokens": { "max": 128 } },
  "states": { "kv": { "bytes_per_token": 4096 } },
  "buffers": {
    "x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },
    "w": { "dtype": "bf16", "shape": [64, 64], "kind": "weight", "bind": [{ "tensor": "w" }] },
    "h": { "dtype": "bf16", "shape": ["tokens", 64], "kind": "workspace" },
    "y": { "dtype": "bf16", "shape": ["tokens", 64], "kind": "output" }
  },
  "modules": {
    "toy": { "source": "toy.cubin", "sha256": "abababababababababababababababababababababababababababababababab" }
  },
  "ops": {
    "embed": {
      "params": ["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
      "impl": {
        "launches": [
          { "module": "toy", "entry": "embed_k",
            "block": [128, 1, 1], "grid": [{ "ceil_div": ["tokens", 128] }, 1, 1] }
        ]
      }
    },
    "attn": {
      "params": ["in buffer<bf16>", "inout state", "out buffer<bf16>", "i32", "i64"],
      "impl": {
        "scratch": {
          "part": { "dtype": "f32", "shape": ["tokens", 8] }
        },
        "launches": [
          {
            "module": "toy", "entry": "attn_part_k",
            "params": ["in buffer<bf16>", "inout state", "out buffer<f32>", "i32", "i64"],
            "block": [128, 1, 1],
            "grid": ["tokens", 8, 1],
            "args": [{ "param": 0 }, { "param": 1 }, { "scratch": "part" }, { "param": 3 }, { "param": 4 }]
          },
          {
            "module": "toy", "entry": "attn_reduce_k",
            "params": ["in buffer<f32>", "out buffer<bf16>", "i32"],
            "block": [128, 1, 1],
            "grid": ["tokens", 1, 1],
            "args": [{ "scratch": "part" }, { "param": 2 }, { "i32": 8 }]
          }
        ]
      }
    }
  },
  "programs": {
    "decode": { "calls": [
      { "label": "embed", "op": "embed",
        "args": [{ "buf": "x" }, { "buf": "w" }, { "buf": "h" }, { "var": "tokens" }] },
      { "label": "attn", "op": "attn",
        "args": [{ "buf": "h" }, { "state": "kv" }, { "buf": "y" }, { "var": "tokens" }, { "i64": 0 }] }
    ] }
  }
}"#;

fn base() -> serde_json::Value {
    serde_json::from_str(BASE).unwrap()
}

fn check(v: serde_json::Value) -> Result<Verified, VerifyErrors> {
    let m: Manifest = serde_json::from_value(v).map_err(|e| VerifyErrors(vec![e.to_string()]))?;
    verify(m)
}

fn assert_err(v: serde_json::Value, needle: &str) {
    let errs = check(v).expect_err("expected verification failure");
    assert!(errs.iter().any(|e| e.contains(needle)), "no error containing `{needle}` in {errs:#?}");
}

#[test]
fn base_manifest_verifies() {
    check(base()).unwrap();
}

#[test]
fn roundtrip_keeps_defaults_implicit() {
    let m = Manifest::from_json(BASE).unwrap();
    let j = m.to_json();
    let v: serde_json::Value = serde_json::from_str(&j).unwrap();
    let embed = &v["ops"]["embed"]["impl"]["launches"][0];
    assert!(
        embed.get("args").is_none() && embed.get("params").is_none(),
        "defaulted wiring must not be materialized: {embed}"
    );
    let again = Manifest::from_json(&j).unwrap();
    assert_eq!(again.to_json(), j);
}

#[test]
fn launch_defaults_resolve_to_interface() {
    let m = Manifest::from_json(BASE).unwrap();
    let op = &m.ops["embed"];
    let l = &op.imp.launches[0];
    assert_eq!(l.params_of(op), &op.params[..]);
    assert_eq!(
        l.args_of(op).as_ref(),
        &[
            LaunchArg::Param { param: 0 },
            LaunchArg::Param { param: 1 },
            LaunchArg::Param { param: 2 },
            LaunchArg::Param { param: 3 }
        ]
    );
}

#[test]
fn unknown_module() {
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0]["module"] = "ghost".into();
    assert_err(v, "unknown module `ghost`");
}

#[test]
fn unused_module() {
    let mut v = base();
    v["modules"]["spare"] = serde_json::json!({ "source": "spare.cubin", "sha256": "cd".repeat(32) });
    assert_err(v, "module `spare` is never launched");
}

#[test]
fn module_sha256_shape() {
    let mut v = base();
    v["modules"]["toy"]["sha256"] = "abc".into();
    assert_err(v, "is not 64 hex chars");
}

#[test]
fn registry_module_malformed_ref() {
    let mut v = base();
    v["modules"]["toy"]["source"] = "hf:org/repo".into();
    assert_err(v, "invalid registry ref `hf:org/repo`");
}

#[test]
fn registry_module_verifies() {
    let mut v = base();
    v["modules"]["toy"]["source"] = "hf:org/repo/pkg/embed.cubin@v1".into();
    check(v).unwrap();
}

#[test]
fn registry_ref_parsing() {
    use kern_manifest::types::RegistryRef;
    assert!(RegistryRef::parse("embed.cubin").is_none());
    let r = RegistryRef::parse("hf:org/repo/a/b.cubin").unwrap().unwrap();
    assert_eq!((r.org.as_str(), r.repo.as_str()), ("org", "repo"));
    assert_eq!((r.path.as_str(), r.revision.as_str()), ("a/b.cubin", "main"));
    let r = RegistryRef::parse("hf:org/repo/a.cubin@abc123").unwrap().unwrap();
    assert_eq!(r.revision, "abc123");
    for bad in ["hf:org", "hf:org/repo", "hf:org/repo/", "hf:org//x", "hf:o/r/x@", "hf:o/r/../x", "hf:o/r/a//b"] {
        assert!(RegistryRef::parse(bad).unwrap().is_err(), "{bad}");
    }
}

#[test]
fn extern_launch_has_no_geometry() {
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0] = serde_json::json!({ "entry": "extern:cublaslt_bf16_tn" });
    // arity: extern gemm inherits the 4-param interface here; only the
    // geometry rule is under test
    check(v).unwrap();
    // geometry on an extern: matches neither launch shape
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0] =
        serde_json::json!({ "entry": "extern:cublaslt_bf16_tn", "block": [1, 1, 1], "grid": [1, 1, 1] });
    assert!(check(v).is_err());
    // a module with an extern entry
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0]["entry"] = "extern:x".into();
    assert_err(v, "an extern entry has no module or launch geometry");
}

#[test]
fn kernel_launch_needs_module_and_geometry() {
    // no module, no geometry, not extern: parses as an extern launch and
    // the verifier names the rule
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0] = serde_json::json!({ "entry": "embed_k" });
    assert_err(v, "a launch without a module must be a runtime built-in");
    // geometry without a module: no shape accepts it
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0].as_object_mut().unwrap().remove("module");
    assert!(check(v).is_err());
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0].as_object_mut().unwrap().remove("grid");
    assert!(check(v).is_err());
}

#[test]
fn batch_span_names_a_var() {
    let mut v = base();
    v["programs"]["decode"]["batch"] = serde_json::json!({"groups": 1, "rows": "tokens", "span": "nope"});
    assert_err(v, "batch.span names unknown var `nope`");
}

#[test]
fn graph_needs_a_batch() {
    let mut v = base();
    v["programs"]["decode"]["graph"] = true.into();
    assert_err(v, "`graph` (captured per call shape) needs a `batch`");
}

#[test]
fn wrong_version() {
    let mut v = base();
    v["schema_version"] = 3.into();
    assert_err(v, "unsupported schema_version 3");
}

#[test]
fn dtype_mismatch() {
    let mut v = base();
    v["buffers"]["h"]["dtype"] = "f32".into();
    assert_err(v, "has dtype f32 but param expects bf16");
}

#[test]
fn unknown_op() {
    let mut v = base();
    v["programs"]["decode"]["calls"][0]["op"] = "nope".into();
    assert_err(v, "unknown op `nope`");
}

#[test]
fn arg_count_mismatch() {
    let mut v = base();
    v["programs"]["decode"]["calls"][1]["args"].as_array_mut().unwrap().pop();
    assert_err(v, "takes 5 params, got 4 args");
}

#[test]
fn read_before_write() {
    let mut v = base();
    let calls = v["programs"]["decode"]["calls"].as_array_mut().unwrap();
    calls.swap(0, 1);
    assert_err(v, "read before ever being written");
}

#[test]
fn write_to_weight() {
    let mut v = base();
    v["programs"]["decode"]["calls"][0]["args"][2] = serde_json::json!({ "buf": "w" });
    assert_err(v, "writes to read-only weight buffer `w`");
}

#[test]
fn var_exceeds_i32() {
    let mut v = base();
    v["vars"]["tokens"]["max"] = 3_000_000_000u64.into();
    assert_err(v, "exceeds i32 range");
}

#[test]
fn output_never_written() {
    let mut v = base();
    v["programs"]["decode"]["calls"].as_array_mut().unwrap().pop();
    assert_err(v, "output buffer `y` is never written");
}

#[test]
fn duplicate_name_rejected() {
    let dup = BASE.replace(
        r#""x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },"#,
        r#""x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },
           "x": { "dtype": "i32", "shape": ["tokens"], "kind": "input" },"#,
    );
    let err = Manifest::from_json(&dup).expect_err("duplicate must fail");
    assert!(err.to_string().contains("duplicate name `x`"), "{err}");
}

#[test]
fn unknown_field_rejected() {
    let mut v = base();
    v["surprise"] = 1.into();
    let errs = check(v).expect_err("unknown field must fail");
    assert!(errs[0].contains("unknown field"), "{errs:?}");
}

#[test]
fn grid_division_by_zero() {
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0]["grid"][0] = serde_json::json!({ "ceil_div": ["tokens", 0] });
    assert_err(v, "division by zero");
}

#[test]
fn unused_buffer() {
    let mut v = base();
    v["buffers"]["dead"] = serde_json::json!({ "dtype": "bf16", "shape": [8], "kind": "workspace" });
    assert_err(v, "buffer `dead` is never used");
}

#[test]
fn block_too_large() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["launches"][0]["block"] = serde_json::json!([1024, 2, 1]);
    assert_err(v, "exceeds 1024 threads");
}

#[test]
fn buf_offset_ok() {
    let mut v = base();
    // h is bf16 [tokens=128, 64] -> 16384 bytes max
    v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offset": 128 });
    check(v).unwrap();
}

#[test]
fn buf_offset_misaligned() {
    let mut v = base();
    v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offset": 3 });
    assert_err(v, "not 2-aligned");
}

#[test]
fn buf_offset_out_of_range() {
    let mut v = base();
    v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offset": 16384 });
    assert_err(v, "outside buffer `h`");
}

#[test]
fn u8_param_and_var_range() {
    let mut v = base();
    v["ops"]["attn"]["params"][3] = "u8".into();
    v["ops"]["attn"]["impl"]["launches"][0]["params"][3] = "u8".into();
    v["programs"]["decode"]["calls"][1]["args"][3] = serde_json::json!({ "u8": 1 });
    check(v).unwrap();
    // binding a var with max 300 to a u8 param is rejected
    let mut v = base();
    v["ops"]["attn"]["params"][3] = "u8".into();
    v["ops"]["attn"]["impl"]["launches"][0]["params"][3] = "u8".into();
    v["vars"]["tokens"]["max"] = 300.into();
    assert_err(v, "exceeds u8 range");
}

#[test]
fn u32_scalar_is_gone() {
    let mut v = base();
    v["ops"]["attn"]["params"][3] = "u32".into();
    let errs = check(v).expect_err("u32 scalars are not a thing");
    assert!(errs[0].contains("invalid param type"), "{errs:?}");
}

#[test]
fn shared_mem_within_limit_ok() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["launches"][0]["shared_mem"] = 167_184u64.into();
    check(v).unwrap();
}

#[test]
fn shared_mem_exceeds_limit() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["launches"][0]["shared_mem"] = 300_000u64.into();
    assert_err(v, "exceeds opt-in limit 232448");
}

#[test]
fn buffer_arg_to_state_param() {
    let mut v = base();
    v["programs"]["decode"]["calls"][1]["args"][1] = serde_json::json!({ "buf": "h" });
    assert_err(v, "does not match param `inout state`");
}

#[test]
fn grid_exceeds_cuda_limit_y() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["launches"][0]["grid"][1] = 100_000u64.into();
    assert_err(v, "exceeds CUDA limit 65535");
}

#[test]
fn bad_param_string_rejected() {
    let mut v = base();
    v["ops"]["attn"]["params"][0] = "buffer<bf16>".into();
    let errs = check(v).expect_err("param without direction must fail");
    assert!(errs[0].contains("invalid param type"), "{errs:?}");
}

// --- impl-layer checks ---

#[test]
fn launch_param_index_out_of_range() {
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0]["args"] =
        serde_json::json!([{ "param": 0 }, { "param": 1 }, { "param": 2 }, { "param": 9 }]);
    assert_err(v, "interface param #9 out of range");
}

#[test]
fn launch_writes_interface_in_param() {
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0]["params"] =
        serde_json::json!(["out buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32"]);
    assert_err(v, "writes through interface `in` param #0");
}

#[test]
fn launch_iface_kind_mismatch() {
    let mut v = base();
    // launch param says bf16 buffer where the interface forwards i32
    v["ops"]["embed"]["impl"]["launches"][0]["params"] =
        serde_json::json!(["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"]);
    assert_err(v, "does not match launch param");
}

#[test]
fn defaulted_args_with_explicit_params_must_agree_in_arity() {
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"][0]["params"] =
        serde_json::json!(["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"]);
    assert_err(v, "takes 5 params, got 4 args");
}

#[test]
fn scratch_read_before_write() {
    let mut v = base();
    let launches = v["ops"]["attn"]["impl"]["launches"].as_array_mut().unwrap();
    launches.swap(0, 1);
    assert_err(v, "scratch `part` is read before any launch wrote it");
}

#[test]
fn scratch_unknown() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["launches"][0]["args"][2] = serde_json::json!({ "scratch": "nope" });
    assert_err(v, "unknown scratch `nope`");
}

#[test]
fn scratch_unused() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["scratch"]["dead"] = serde_json::json!({ "dtype": "f32", "shape": [4] });
    assert_err(v, "scratch `dead` is never used");
}

#[test]
fn scratch_dtype_mismatch() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["launches"][0]["params"][2] = "out buffer<bf16>".into();
    assert_err(v, "scratch `part` has dtype f32 but param expects bf16");
}

#[test]
fn scratch_offset_is_gone() {
    let mut v = base();
    v["ops"]["attn"]["impl"]["launches"][1]["args"][0] = serde_json::json!({ "scratch": "part", "offset": 4096 });
    let errs = check(v).expect_err("scratch offsets are not a thing");
    assert!(errs[0].contains("did not match any variant"), "{errs:?}");
    // and a typo in a call arg is a parse error, not a silently ignored key
    let mut v = base();
    v["programs"]["decode"]["calls"][1]["args"][0] = serde_json::json!({ "buf": "h", "offest": 128 });
    check(v).expect_err("unknown keys in args must fail");
}

#[test]
fn interface_out_never_written() {
    let mut v = base();
    // reduce launch now writes scratch instead of the interface out param
    v["ops"]["attn"]["impl"]["launches"][1]["params"][1] = "out buffer<f32>".into();
    v["ops"]["attn"]["impl"]["launches"][1]["args"][1] = serde_json::json!({ "scratch": "part" });
    assert_err(v, "param #2 is never written by any launch");
}

#[test]
fn empty_impl_rejected() {
    let mut v = base();
    v["ops"]["embed"]["impl"]["launches"] = serde_json::json!([]);
    assert_err(v, "implementation has no launches");
}

#[test]
fn state_bytes_forms() {
    let mut v = base();
    v["states"]["kv"] = serde_json::json!({ "bytes": 4096 });
    check(v).unwrap();
    let mut v = base();
    v["states"]["kv"] = serde_json::json!({ "bytes": 4096, "bytes_per_token": 1 });
    assert_err(v, "are exclusive");
    let mut v = base();
    v["states"]["kv"] = serde_json::json!({});
    assert_err(v, "must be > 0");
    let mut v = base();
    v["states"]["kv"] = serde_json::json!({ "bytes": 4096, "align": 256 });
    let errs = check(v).expect_err("align is gone");
    assert!(errs[0].contains("unknown field"), "{errs:?}");
}

// --- topology / export / peer / rank ---

/// The base fixture plus an `ep` group of 4, an exported flag buffer,
/// its peer address array and a barrier op taking both plus the rank.
fn peer_base() -> serde_json::Value {
    let mut v = base();
    v["topology"] = serde_json::json!({ "groups": { "ep": 4 } });
    v["buffers"]["flags"] = serde_json::json!({ "dtype": "u32", "shape": [64], "kind": "carry", "export": true });
    v["buffers"]["flags_peers"] =
        serde_json::json!({ "dtype": "u64", "shape": [4], "kind": "peer", "of": "flags", "group": "ep" });
    v["ops"]["barrier"] = serde_json::json!({
        "params": ["inout buffer<u32>", "in buffer<u64>", "i32", "i32"],
        "impl": { "launches": [
            { "module": "toy", "entry": "barrier_k", "block": [32, 1, 1], "grid": [1, 1, 1],
              "args": [{ "param": 0 }, { "param": 1 }, { "param": 2 }, { "rank": "ep" }] }
        ] }
    });
    v["programs"]["decode"]["calls"].as_array_mut().unwrap().push(serde_json::json!({
        "op": "barrier",
        "args": [{ "buf": "flags" }, { "buf": "flags_peers" }, { "rank": "ep" }, { "i32": 0 }]
    }));
    v
}

#[test]
fn peer_manifest_verifies() {
    check(peer_base()).unwrap();
    // a peer array of a state
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["of"] = "kv".into();
    check(v).unwrap();
    // rank into an i64 param
    let mut v = peer_base();
    v["ops"]["barrier"]["params"][2] = "i64".into();
    check(v).unwrap();
}

#[test]
fn peer_shape_and_dtype() {
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["shape"] = serde_json::json!([8]);
    assert_err(v, "has shape [4], one address per member");
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["shape"] = serde_json::json!([4, 1]);
    assert_err(v, "has shape [4], one address per member");
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["dtype"] = "i64".into();
    v["ops"]["barrier"]["params"][1] = "in buffer<i64>".into();
    assert_err(v, "dtype must be u64");
}

#[test]
fn peer_of_must_be_exported() {
    let mut v = peer_base();
    v["buffers"]["flags"]["export"] = false.into();
    assert_err(v, "`of` buffer `flags` is not exported");
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["of"] = "ghost".into();
    assert_err(v, "`of` unknown buffer/state `ghost`");
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["of"] = "flags_peers".into();
    assert_err(v, "cannot be `of` itself");
    let mut v = peer_base();
    v["buffers"]["flags_peers"].as_object_mut().unwrap().remove("of");
    assert_err(v, "must name the exported buffer or state");
}

#[test]
fn peer_group_rules() {
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["group"] = "tp".into();
    assert_err(v, "unknown topology group `tp`");
    let mut v = peer_base();
    v.as_object_mut().unwrap().remove("topology");
    assert_err(v, "declares no topology");
    let mut v = peer_base();
    v["topology"]["groups"]["cp"] = 2.into();
    assert_err(v, "topology group `cp` is never used");
    let mut v = peer_base();
    v["topology"]["groups"]["ep"] = 0.into();
    assert_err(v, "size must be > 0");
    let mut v = peer_base();
    v["buffers"]["flags"]["group"] = "ep".into();
    assert_err(v, "`group` only applies to peer buffers");
    let mut v = peer_base();
    v["buffers"]["flags"]["of"] = "flags".into();
    assert_err(v, "`of` only applies to peer buffers");
    let mut v = peer_base();
    v["buffers"]["flags_peers"]["export"] = true.into();
    assert_err(v, "cannot itself be exported");
}

#[test]
fn peer_is_read_only_and_never_extern() {
    let mut v = peer_base();
    v["ops"]["barrier"]["params"][1] = "inout buffer<u64>".into();
    assert_err(v, "writes to read-only peer buffer `flags_peers`");
    // a peer buffer reaching an op with an extern launch
    let mut v = peer_base();
    v["ops"]["barrier"]["impl"]["launches"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "entry": "extern:cublaslt_bf16_tn", "params": [], "args": [] }));
    assert_err(v, "runtime built-ins never receive peer memory");
}

#[test]
fn rank_binds_integers_only() {
    let mut v = peer_base();
    v["ops"]["barrier"]["params"][2] = "f32".into();
    v["ops"]["barrier"]["impl"]["launches"][0]["args"][2] = serde_json::json!({ "f32": 0.0 });
    assert_err(v, "rank in group `ep` does not match param `f32`");
    let mut v = peer_base();
    v["ops"]["barrier"]["params"][3] = "u8".into();
    v["programs"]["decode"]["calls"][2]["args"][3] = serde_json::json!({ "u8": 0 });
    assert_err(v, "a rank binds only to an i32 or i64 param, not `u8`");
    let mut v = peer_base();
    v["ops"]["barrier"]["impl"]["launches"][0]["args"][3] = serde_json::json!({ "rank": "nope" });
    assert_err(v, "unknown topology group `nope`");
}

/// A launch taking a tensor map over interface buffer #1 (u8 raw bytes),
/// launched as clusters of 2.
fn pack_base() -> serde_json::Value {
    let mut v = base();
    v["buffers"]["pk_out"] = serde_json::json!({ "dtype": "f32", "shape": [64], "kind": "output" });
    v["ops"]["pk"] = serde_json::json!({
        "params": ["out buffer<f32>", "in buffer<bf16>", "i32"],
        "impl": { "launches": [
            { "module": "toy", "entry": "pk_k", "block": [128, 1, 1], "grid": [1, 1, 1],
              "params": ["bytes<24>", "in buffer<bf16>"],
              "args": [{ "pack": { "size": 24, "fields": [
                            { "at": 0, "param": 0 }, { "at": 8, "param": 2 }, { "at": 12, "var": "tokens" },
                            { "at": 16, "i64": 4608 } ] } },
                       { "param": 1 }] }
        ] }
    });
    v["programs"]["decode"]["calls"].as_array_mut().unwrap().push(serde_json::json!({
        "op": "pk",
        "args": [{ "buf": "pk_out" }, { "buf": "w" }, { "var": "tokens" }]
    }));
    v
}

#[test]
fn pack_manifest_verifies() {
    check(pack_base()).unwrap();
    assert_eq!("bytes<48>".parse::<ParamType>(), Ok(ParamType::Bytes(48)));
    assert_eq!(ParamType::Bytes(48).size_bytes(), 48);
    assert!("bytes<0>".parse::<ParamType>().is_err());
}

#[test]
fn pack_rules() {
    let mut v = pack_base();
    v["ops"]["pk"]["impl"]["launches"][0]["params"][0] = "bytes<32>".into();
    assert_err(v, "pack of 24 bytes bound to a `bytes<32>` param");
    let mut v = pack_base();
    v["ops"]["pk"]["impl"]["launches"][0]["args"][0]["pack"]["fields"][1]["at"] = 4.into();
    assert_err(v, "field #1 overlaps field #0");
    let mut v = pack_base();
    v["ops"]["pk"]["impl"]["launches"][0]["args"][0]["pack"]["fields"][3]["at"] = 20.into();
    assert_err(v, "field #3 at 20 spans 8 bytes, past the 24 byte image");
    let mut v = pack_base();
    v["ops"]["pk"]["impl"]["launches"][0]["args"][0]["pack"]["fields"][2]["var"] = "ghost".into();
    assert_err(v, "unknown var `ghost`");
    let mut v = pack_base();
    v["ops"]["pk"]["params"][0] = "bytes<8>".into();
    assert_err(v, "interface param #0 is a byte aggregate");
    let mut v = pack_base();
    v["ops"]["pk"]["impl"]["launches"][0]["args"][1] = serde_json::json!({ "pack": { "size": 8, "fields": [] } });
    assert_err(v, "a pack binds only to a `bytes<n>` param");
}

#[test]
fn tensormap_spans_the_buffer() {
    // outermost 0 = as many slices as the buffer holds: no footprint to check against the call
    let mut v = tensormap_base();
    v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["pack"]["fields"][0]["tensormap"]["dims"] =
        serde_json::json!([128, 0]);
    check(v).unwrap();
    let mut v = tensormap_base();
    v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["pack"]["fields"][0]["tensormap"]["dims"] =
        serde_json::json!([0, 4]);
    assert_err(v, "dims[0] is 0; only the outermost dim may span the buffer");
}

fn tensormap_base() -> serde_json::Value {
    let mut v = base();
    v["buffers"]["tm_out"] = serde_json::json!({ "dtype": "bf16", "shape": [64], "kind": "output" });
    v["buffers"]["raw"] = serde_json::json!({ "dtype": "u8", "shape": [512], "kind": "input" });
    v["ops"]["tm"] = serde_json::json!({
        "params": ["out buffer<bf16>", "in buffer<u8>"],
        "impl": { "launches": [
            { "module": "toy", "entry": "tm_k", "block": [128, 1, 1], "grid": [2, 1, 1], "cluster": [2, 1, 1],
              "params": ["out buffer<bf16>", "bytes<128>"],
              "args": [{ "param": 0 }, { "pack": { "size": 128, "fields": [
                  { "at": 0, "tensormap": { "param": 1, "dtype": "u8", "dims": [128, 4],
                                            "strides": [128], "box": [128, 4], "swizzle": 128 } }] } }] }
        ] }
    });
    v["programs"]["decode"]["calls"].as_array_mut().unwrap().push(serde_json::json!({
        "op": "tm",
        "args": [{ "buf": "tm_out" }, { "buf": "raw" }]
    }));
    v
}

#[test]
fn tensormap_manifest_verifies() {
    check(tensormap_base()).unwrap();
    // the call's offset shrinks what the descriptor may address: 512 - 0 ok, exactly fits
    let mut v = tensormap_base();
    v["programs"]["decode"]["calls"][2]["args"][1]["offset"] = 0.into();
    check(v).unwrap();
}

#[test]
fn tensormap_shape_rules() {
    fn tm(v: &mut serde_json::Value) -> &mut serde_json::Value {
        &mut v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["pack"]["fields"][0]["tensormap"]
    }
    let mut v = tensormap_base();
    tm(&mut v)["box"] = serde_json::json!([512, 4]);
    assert_err(v, "box[0] = 512, must be 1..=256");
    let mut v = tensormap_base();
    tm(&mut v)["strides"] = serde_json::json!([120]);
    assert_err(v, "strides[0] = 120 is not a positive multiple of 16");
    let mut v = tensormap_base();
    tm(&mut v)["swizzle"] = 64.into();
    assert_err(v, "box[0] spans 128 bytes, more than the 64 byte swizzle span");
    let mut v = tensormap_base();
    tm(&mut v)["dims"] = serde_json::json!([128, 5]);
    assert_err(v, "addresses 640 bytes but buffer `raw` has 512 bytes past offset 0");
    let mut v = tensormap_base();
    v["programs"]["decode"]["calls"][2]["args"][1]["offset"] = 16.into();
    assert_err(v, "has 496 bytes past offset 16");
}

#[test]
fn tensormap_binding_rules() {
    // only over a buffer or state param
    let mut v = tensormap_base();
    v["ops"]["tm"]["params"][1] = "i32".into();
    v["programs"]["decode"]["calls"][2]["args"][1] = serde_json::json!({ "i32": 1 });
    assert_err(v, "field #0: tensormap over interface param #1 (`i32`), which is not a buffer or state");
    // 128 bytes at a 64-byte aligned offset
    fn field(v: &mut serde_json::Value) -> &mut serde_json::Value {
        &mut v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["pack"]["fields"][0]
    }
    let mut v = tensormap_base();
    field(&mut v)["width"] = 64.into();
    assert_err(v, "field #0: a tensormap field is 128 bytes, not 64");
    let mut v = tensormap_base();
    v["ops"]["tm"]["impl"]["launches"][0]["params"][1] = "bytes<192>".into();
    v["ops"]["tm"]["impl"]["launches"][0]["args"][1]["pack"]["size"] = 192.into();
    field(&mut v)["at"] = 32.into();
    assert_err(v, "field #0: tensormap at 32 is not 64-byte aligned");
    // never into an extern
    let mut v = tensormap_base();
    v["ops"]["tm"]["impl"]["launches"][0] = serde_json::json!({
        "entry": "extern:cublaslt_bf16_tn", "params": ["out buffer<bf16>", "bytes<128>"],
        "args": [{ "param": 0 }, { "pack": { "size": 128, "fields": [
            { "at": 0, "tensormap": { "param": 1, "dtype": "u8", "dims": [128], "box": [128] } }] } }]
    });
    assert_err(v, "an extern launch takes pointers and scalars, not a pack");
}

#[test]
fn cluster_divides_grid() {
    let mut v = tensormap_base();
    v["ops"]["tm"]["impl"]["launches"][0]["grid"] = serde_json::json!([3, 1, 1]);
    assert_err(v, "grid.x = 3 at var upper bounds is not a multiple of cluster.x = 2");
    let mut v = tensormap_base();
    v["ops"]["tm"]["impl"]["launches"][0]["cluster"] = serde_json::json!([4, 4, 2]);
    assert_err(v, "has a zero dim or more than 16 blocks");
    let mut v = tensormap_base();
    v["ops"]["tm"]["impl"]["launches"][0]["cluster"] = serde_json::json!([2, 0, 1]);
    assert_err(v, "has a zero dim or more than 16 blocks");
}

#[test]
fn exported_buffer_without_peer_array_verifies() {
    // export alone is legal (a rank may hand its handle to something
    // outside the manifest); the group must still be used somewhere.
    let mut v = peer_base();
    v["buffers"].as_object_mut().unwrap().remove("flags_peers");
    v["ops"]["barrier"]["params"] = serde_json::json!(["inout buffer<u32>", "i32", "i32"]);
    v["ops"]["barrier"]["impl"]["launches"][0]["args"] =
        serde_json::json!([{ "param": 0 }, { "param": 1 }, { "rank": "ep" }]);
    v["programs"]["decode"]["calls"][2]["args"] =
        serde_json::json!([{ "buf": "flags" }, { "rank": "ep" }, { "i32": 0 }]);
    check(v).unwrap();
}

// --- domains ---

#[test]
fn domain_index_into_and_bounds_verify() {
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "w" });
    v["buffers"]["y"]["domain"] = serde_json::json!({ "min": -1.5, "max": 1.5 });
    check(v).unwrap();
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 0, "max": "tokens", "monotone": true });
    check(v).unwrap();
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "kv", "stride": 16 });
    check(v).unwrap();
}

#[test]
fn domain_resolves() {
    use kern_manifest::types::ResolvedDomain;
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "kv", "stride": 16 });
    let m: Manifest = serde_json::from_value(v).unwrap();
    let vars = BTreeMap::from([("tokens".to_string(), 4u64)]);
    let r = m.buffers["x"]
        .domain
        .as_ref()
        .unwrap()
        .resolve(&m, &vars, &Provision { tokens: 4096, seq_slots: m.seq_slots() })
        .unwrap();
    assert_eq!(r, ResolvedDomain { lo: Some(0.0), hi: Some(255.0), monotone: false });
    assert!(r.contains(255.0) && !r.contains(256.0) && !r.contains(-1.0));

    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "w" });
    let m: Manifest = serde_json::from_value(v).unwrap();
    let r = m.buffers["x"]
        .domain
        .as_ref()
        .unwrap()
        .resolve(&m, &vars, &Provision { tokens: 0, seq_slots: m.seq_slots() })
        .unwrap();
    assert_eq!(r.hi, Some(63.0));

    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 1, "max": "tokens" });
    let m: Manifest = serde_json::from_value(v).unwrap();
    let r = m.buffers["x"]
        .domain
        .as_ref()
        .unwrap()
        .resolve(&m, &vars, &Provision { tokens: 0, seq_slots: m.seq_slots() })
        .unwrap();
    assert_eq!((r.lo, r.hi), (Some(1.0), Some(4.0)));
}

#[test]
fn domain_rejects_malformed() {
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "nope" });
    assert_err(v, "unknown buffer/state `nope`");
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "index_into": "w", "max": 3 });
    assert_err(v, "mutually exclusive");
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 0.5 });
    assert_err(v, "float `min` on a i32 buffer");
    let mut v = base();
    v["buffers"]["h"]["domain"] = serde_json::json!({ "index_into": "w" });
    assert_err(v, "a bf16 buffer cannot index anything");
    let mut v = base();
    v["buffers"]["h"]["domain"] = serde_json::json!({ "min": 0, "monotone": true });
    assert_err(v, "`monotone` requires a one-dimensional buffer");
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 5, "max": 2 });
    assert_err(v, "min 5 > max 2");
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({});
    assert_err(v, "empty");
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "max": "ghost" });
    assert_err(v, "unknown var `ghost`");
    let mut v = base();
    v["buffers"]["x"]["domain"] = serde_json::json!({ "min": 0, "stride": 4 });
    assert_err(v, "`stride` only applies with `index_into`");
}

#[test]
fn bind_is_a_weight_s_and_only_a_weight_s() {
    let mut v = base();
    v["buffers"]["w"]["bind"] = serde_json::json!([]);
    assert_err(v, "a weight buffer binds at least one checkpoint tensor");
    let mut v = base();
    v["buffers"]["h"]["bind"] = serde_json::json!([{ "tensor": "h" }]);
    assert_err(v, "not a workspace buffer");
    let mut v = base();
    v["buffers"]["w"]["bind"] = serde_json::json!([{ "tensor": "" }]);
    assert_err(v, "bind[0]: empty tensor name");
    let mut v = base();
    v["buffers"]["w"]["bind"] = serde_json::json!([{ "tensor": "a" }, { "tensor": "b", "rows": [4, 4] }]);
    assert_err(v, "bind[1]: rows [4, 4) is empty");
    let mut v = base();
    v["buffers"]["w"]["bind"] =
        serde_json::json!([{ "tensor": "a", "cols": [0, 32] }, { "tensor": "a", "cols": [32, 64], "rows": [0, 64] }]);
    let m: Manifest = serde_json::from_value(v).unwrap();
    assert!(verify(m).is_ok());
}

#[test]
fn microscale_is_a_float_and_cannot_index_state() {
    let mut v = base();
    v["buffers"]["x"]["dtype"] = "fp8e8m0".into();
    v["ops"]["embed"]["params"][0] = "in buffer<fp8e8m0>".into();
    assert!(check(v.clone()).is_ok());
    v["buffers"]["x"]["domain"] = serde_json::json!({"index_into": "kv"});
    assert_err(v, "cannot index anything");
}

#[test]
fn ranked_binding_requires_one_name_per_group_member() {
    let mut v = base();
    v["topology"] = serde_json::json!({"groups": {"ep": 2}});
    v["buffers"]["w"]["bind"][0]["tensor"] = serde_json::json!({"group": "ep", "tensors": ["w0", "w1"]});
    assert!(check(v.clone()).is_ok());
    let mut bad = v.clone();
    bad["buffers"]["w"]["bind"][0]["tensor"]["tensors"] = serde_json::json!(["w0"]);
    assert_err(bad, "needs 2 tensor names");
    let mut bad = v.clone();
    bad["buffers"]["w"]["bind"][0]["tensor"]["tensors"][1] = "".into();
    assert_err(bad, "empty tensor name");
    v["buffers"]["w"]["bind"][0]["tensor"]["group"] = "missing".into();
    assert_err(v, "unknown topology group");
}

#[test]
fn ranked_rows_shard_one_tensor_across_a_group() {
    let mut v = base();
    v["topology"] = serde_json::json!({"groups": {"ep": 2}});
    v["buffers"]["w"]["bind"][0]["rows"] = serde_json::json!({"group": "ep", "ranges": [[0, 4], [3, 7]]});
    assert!(check(v.clone()).is_ok());
    let mut bad = v.clone();
    bad["buffers"]["w"]["bind"][0]["rows"]["ranges"] = serde_json::json!([[0, 4]]);
    assert_err(bad, "needs 2 row ranges");
    let mut bad = v.clone();
    bad["buffers"]["w"]["bind"][0]["rows"]["ranges"][1] = serde_json::json!([4, 4]);
    assert_err(bad, "rows [4, 4) is empty");
    v["buffers"]["w"]["bind"][0]["rows"]["group"] = "missing".into();
    assert_err(v, "unknown topology group");
}

#[test]
fn host_placement_is_immutable_and_rank_independent() {
    let mut v = base();
    v["buffers"]["w"]["placement"] = serde_json::json!("host");
    assert!(check(v.clone()).is_ok());
    v["buffers"]["w"]["export"] = serde_json::json!(true);
    assert!(check(v.clone()).unwrap_err().to_string().contains("non-exported immutable"));
    v["buffers"]["w"]["export"] = serde_json::json!(false);
    v["topology"] = serde_json::json!({"groups": {"ep": 2}});
    v["buffers"]["w"]["bind"][0]["tensor"] = serde_json::json!({"group": "ep", "tensors": ["w0", "w1"]});
    assert!(check(v.clone()).unwrap_err().to_string().contains("cannot select tensors or rows by rank"));
    v["buffers"]["w"]["bind"][0]["tensor"] = serde_json::json!("w0");
    v["buffers"]["w"]["bind"][0]["rows"] = serde_json::json!({"group": "ep", "ranges": [[0, 4], [4, 8]]});
    assert!(check(v).unwrap_err().to_string().contains("cannot select tensors or rows by rank"));
    let mut v = base();
    v["buffers"]["x"]["placement"] = serde_json::json!("host");
    assert!(check(v).unwrap_err().to_string().contains("immutable weight"));
}
