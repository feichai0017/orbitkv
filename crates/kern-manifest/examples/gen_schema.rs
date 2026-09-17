//! Emit the manifest wire-format definition as JSON Schema.
//!
//! The Rust types in `kern_manifest::types` are the ground truth (the parser
//! is the law); this is their publishable projection. The committed copy at
//! `schema/manifest-v5.schema.json` is golden-checked in CI:
//!
//! ```sh
//! cargo run -p kern-manifest --example gen_schema > schema/manifest-v5.schema.json
//! ```

fn main() {
    let mut v = kern_manifest::json_schema();
    v["$id"] = "https://kern-baa.pages.dev/schema/manifest-v5.schema.json".into();
    v["title"] = "kern manifest v5".into();
    println!("{}", serde_json::to_string_pretty(&v).expect("schema serializes"));
}
