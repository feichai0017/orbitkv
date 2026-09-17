//! Parse, verify and read the serving protocol of manifests:
//! `cargo run -p kern-manifest --example verify -- a.json b.json`.
fn main() {
    let mut bad = 0;
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).expect("read manifest");
        match kern_manifest::Verified::from_json(&text) {
            Ok(m) => match kern_manifest::Protocol::check(&m) {
                Ok(p) => println!("{path}: ok, {} forwards, {} fills", p.forwards.len(), p.fills.len()),
                Err(e) => println!("{path}: verified; {e}"),
            },
            Err(e) => {
                bad += 1;
                println!("{path}: {e}");
            }
        }
    }
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
