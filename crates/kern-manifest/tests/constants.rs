use kern_manifest::{verify, Manifest, Protocol};
use serde_json::{json, Value};

fn minimal() -> Value {
    serde_json::from_str(include_str!("../../../examples/minimal.json")).unwrap()
}

#[test]
fn names_expand_only_in_numeric_positions() {
    let original = minimal();
    let mut named = original.clone();
    named["constants"] = json!({"width": 64, "capacity": 1024, "kv_bytes": 256, "x": 123, "scale": 456});
    named["vars"]["tokens"]["max"] = json!("capacity");
    named["states"]["kv"]["bytes_per_token"] = json!("kv_bytes");
    for b in ["x", "w", "y"] {
        let shape = named["buffers"][b]["shape"].as_array_mut().unwrap();
        *shape.last_mut().unwrap() = json!("width");
    }
    named["ops"]["scale"]["impl"]["launches"][0]["block"][0] = json!("width");
    let parsed: Manifest = serde_json::from_value(named).unwrap();
    assert_eq!(parsed.to_json(), serde_json::from_value::<Manifest>(original).unwrap().to_json());
    verify(parsed).unwrap();
}

#[test]
fn scalar_types_offsets_and_nested_expressions() {
    let mut m = minimal();
    m["constants"] = json!({"n": 64, "offset": 2, "scale": 0.125, "large": 9223372036854775807i64, "byte": 255});
    m["programs"]["step"]["calls"][0]["args"] = json!([
        {"buf": "x", "offset": "offset"}, {"i32": "n"}, {"i64": "large"},
        {"u8": "byte"}, {"f32": "scale"}, {"expr": {"ceil_div": [{"mul": ["tokens", "n"]}, "n"]}}
    ]);
    m["ops"]["scale"]["impl"]["launches"][0]["args"] = json!([
        {"pack": {"size": "n", "fields": [{"at": "offset", "i32": "n", "width": "offset"}]}}
    ]);
    let parsed: Manifest = serde_json::from_value(m).unwrap();
    let v = serde_json::to_value(parsed).unwrap();
    let args = &v["programs"]["step"]["calls"][0]["args"];
    assert_eq!(args[0]["offset"], 2);
    assert_eq!(args[1]["i32"], 64);
    assert_eq!(args[2]["i64"], 9223372036854775807i64);
    assert_eq!(args[3]["u8"], 255);
    assert_eq!(args[4]["f32"], 0.125);
    assert_eq!(args[5]["expr"], json!({"ceil_div": [{"mul": ["tokens", 64]}, 64]}));
    assert_eq!(v["ops"]["scale"]["impl"]["launches"][0]["args"][0]["pack"]["fields"][0]["at"], 2);
}

#[test]
fn invalid_declarations_references_and_types_are_rejected() {
    for constants in [
        json!(null),
        json!([]),
        json!({"": 1}),
        json!({"n": "other"}),
        json!({"n": {"mul": [2, 4]}}),
        json!({"tokens": 4}),
        json!({"n": true}),
    ] {
        let mut m = minimal();
        m["constants"] = constants;
        assert!(serde_json::from_value::<Manifest>(m).is_err());
    }
    for (ty, value) in
        [("i32", json!(2147483648u64)), ("i64", json!(9223372036854775808u64)), ("u8", json!(256)), ("i32", json!(1.5))]
    {
        let mut m = minimal();
        m["constants"] = json!({"n": value});
        m["programs"]["step"]["calls"][0]["args"][0] = json!({ty: "n"});
        assert!(serde_json::from_value::<Manifest>(m).is_err(), "{ty}");
    }
    let mut m = minimal();
    m["constants"] = json!({"n": -1});
    m["buffers"]["x"]["shape"][1] = json!("n");
    assert!(serde_json::from_value::<Manifest>(m.clone()).is_err());
    m["vars"]["tokens"]["max"] = json!("missing");
    assert!(serde_json::from_value::<Manifest>(m).unwrap_err().to_string().contains("undefined constant `missing`"));
    let mut m = minimal();
    m["constants"] = json!({"version": 4});
    m["schema_version"] = json!("version");
    assert!(serde_json::from_value::<Manifest>(m).is_err());
}

#[test]
fn duplicate_and_unknown_fields_remain_errors() {
    let original = include_str!("../../../examples/minimal.json");
    for declaration in [r#""constants":{"n":1,"n":2},"#, r#""constants":{},"constants":{},"#] {
        let s = original.replacen('{', &format!("{{{declaration}"), 1);
        assert!(Manifest::from_json(&s).unwrap_err().to_string().contains("duplicate name"));
    }
    let mut m = minimal();
    m["constants"] = json!({"n": 64});
    m["buffers"]["x"]["typo"] = json!("n");
    assert!(serde_json::from_value::<Manifest>(m).is_err());
}

#[test]
fn hybrid_and_dflash_examples_verify_and_roundtrip() {
    for text in
        [include_str!("../../../examples/qwen3.8-27b.json"), include_str!("../../../examples/qwen3.8-27b-dflash2.json")]
    {
        let m = verify(Manifest::from_json(text).unwrap()).unwrap();
        Protocol::check(&m).unwrap();
        assert_eq!(m.to_json(), Manifest::from_json(&m.to_json()).unwrap().to_json());
    }
}
