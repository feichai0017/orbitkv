//! Named numeric literals are a wire-only convenience. The runtime sees the
//! same typed manifest as it would after spelling every number out.
use crate::types::*;
use serde::de::{Error, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::LazyLock;

// Remote derive keeps the final conversion subject to the ordinary typed
// deserializer (integer widths, unknown fields, required fields, etc.).
#[derive(Deserialize)]
#[serde(remote = "Manifest", deny_unknown_fields)]
struct LiteralManifest {
    schema_version: u32,
    model: String,
    #[serde(default)]
    topology: Option<Topology>,
    #[serde(default)]
    vars: BTreeMap<String, Var>,
    #[serde(default)]
    states: BTreeMap<String, State>,
    buffers: BTreeMap<String, Buffer>,
    modules: BTreeMap<String, Module>,
    ops: BTreeMap<String, Op>,
    programs: BTreeMap<String, Program>,
}

static LITERAL_SCHEMA: LazyLock<Value> =
    LazyLock::new(|| serde_json::to_value(schemars::schema_for!(Manifest)).expect("schema serializes"));

/// The public wire schema: the typed manifest plus optional named literals.
/// Numeric positions accept names; string-only positions remain untouched.
/// Name resolution and the referenced number's bounds are checked by parsing.
pub fn json_schema() -> Value {
    fn names(v: &mut Value) {
        match v {
            Value::Object(m) => {
                for child in m.values_mut() {
                    names(child);
                }
                if m.get("type").is_some_and(|t| has_type(t, "integer") || has_type(t, "number")) {
                    let literal = std::mem::take(v);
                    let description = literal.get("description").cloned();
                    *v = json!({"anyOf": [literal, {"type": "string", "minLength": 1,
                        "description": "Name of a numeric constant declared in constants."}]});
                    if let Some(description) = description {
                        v["description"] = description;
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(names),
            _ => {}
        }
    }
    let mut schema = LITERAL_SCHEMA.clone();
    names(&mut schema);
    // The format discriminator is always literal, never a model constant.
    schema["properties"]["schema_version"] = LITERAL_SCHEMA["properties"]["schema_version"].clone();
    schema["properties"]["constants"] = json!({
        "type": "object", "additionalProperties": {"type": "number"},
        "propertyNames": {"minLength": 1},
        "description": "Optional named numeric literals, recommended for readability. Use C-style UPPER_SNAKE_CASE names. For example {\"HIDDEN_SIZE\": 2560}. Reference a name in any numeric position except schema_version, including {\"i32\": \"HIDDEN_SIZE\"}, shapes, expressions, capacities and byte offsets. Names must not overlap vars. Values are numbers, not expressions or aliases. References expand before typed validation; serialization emits resolved literals."
    });
    schema
}

// Gather alternatives at one schema position, including refs. Recursive
// expression refs are traversed only as we descend through the input value.
fn alternatives<'a>(schema: &'a Value, out: &mut Vec<&'a Value>) {
    if let Some(r) = schema.get("$ref").and_then(Value::as_str) {
        alternatives(LITERAL_SCHEMA.pointer(r.strip_prefix('#').expect("local schema ref")).expect("schema ref"), out);
    }
    if let Some(a) =
        schema.get("anyOf").or_else(|| schema.get("oneOf")).or_else(|| schema.get("allOf")).and_then(Value::as_array)
    {
        for s in a {
            alternatives(s, out);
        }
    }
    out.push(schema);
}

fn has_type(t: &Value, kind: &str) -> bool {
    t == kind || t.as_array().is_some_and(|a| a.iter().any(|v| v == kind))
}

fn expand(
    value: &mut Value,
    schemas: &[&Value],
    constants: &BTreeMap<String, Value>,
    path: &str,
) -> Result<(), String> {
    let mut choices = Vec::new();
    for s in schemas {
        alternatives(s, &mut choices);
    }
    let numeric = choices.iter().any(|s| has_type(&s["type"], "integer") || has_type(&s["type"], "number"));
    if let Value::String(name) = value {
        if numeric && path != "/schema_version" {
            if let Some(number) = constants.get(name) {
                *value = number.clone();
            } else if !choices.iter().any(|s| has_type(&s["type"], "string")) {
                return Err(format!("{path}: undefined constant `{name}`"));
            }
        }
    } else if let Value::Object(m) = value {
        for (k, v) in m {
            let children: Vec<_> = choices
                .iter()
                .filter_map(|s| {
                    s.get("properties")
                        .and_then(|p| p.get(k))
                        .or_else(|| s.get("additionalProperties").filter(|a| a.is_object()))
                })
                .collect();
            expand(v, &children, constants, &format!("{path}/{k}"))?;
        }
    } else if let Value::Array(a) = value {
        for (i, v) in a.iter_mut().enumerate() {
            let children: Vec<_> = choices
                .iter()
                .filter_map(|s| {
                    s.get("prefixItems").and_then(|p| p.get(i)).or_else(|| s.get("items").filter(|a| a.is_object()))
                })
                .collect();
            expand(v, &children, constants, &format!("{path}/{i}"))?;
        }
    }
    Ok(())
}

impl<'de> Deserialize<'de> for Manifest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = UniqueValue::deserialize(deserializer)?.0;
        let constants = match value.as_object_mut().and_then(|m| m.remove("constants")) {
            None => BTreeMap::new(),
            Some(Value::Object(m)) => m
                .into_iter()
                .map(|(k, v)| {
                    if k.is_empty() || !v.is_number() {
                        Err(D::Error::custom("constants require nonempty names and numeric literal values"))
                    } else if value.get("vars").and_then(|vars| vars.get(&k)).is_some() {
                        Err(D::Error::custom(format!("constant `{k}` conflicts with a var")))
                    } else {
                        Ok((k, v))
                    }
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(D::Error::custom("constants must be an object")),
        };
        if !constants.is_empty() {
            expand(&mut value, &[&LITERAL_SCHEMA], &constants, "").map_err(D::Error::custom)?;
        }
        LiteralManifest::deserialize(value).map_err(D::Error::custom)
    }
}

// serde_json::Value normally loses duplicate keys. Reject them before the
// expansion pass so naming a constant cannot weaken strict parsing.
struct UniqueValue(Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = UniqueValue;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON value with unique object keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut m = serde_json::Map::new();
                while let Some((k, UniqueValue(v))) = a.next_entry::<String, UniqueValue>()? {
                    match m.entry(k) {
                        serde_json::map::Entry::Vacant(entry) => {
                            entry.insert(v);
                        }
                        serde_json::map::Entry::Occupied(entry) => {
                            return Err(A::Error::custom(format!("duplicate name `{}`", entry.key())));
                        }
                    }
                }
                Ok(UniqueValue(Value::Object(m)))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                while let Some(UniqueValue(x)) = a.next_element()? {
                    v.push(x);
                }
                Ok(UniqueValue(Value::Array(v)))
            }
            fn visit_bool<E: Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_i64<E: Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_u64<E: Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_f64<E: Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("non-finite number"))
            }
            fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(v.into()))
            }
            fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
        }
        d.deserialize_any(V)
    }
}
