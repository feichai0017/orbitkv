fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/engine.proto");

    tonic_prost_build::configure()
        .boxed(".orbitkv.InventoryClientFrame.body.open")
        .compile_protos(&["proto/engine.proto"], &["proto"])?;

    Ok(())
}
