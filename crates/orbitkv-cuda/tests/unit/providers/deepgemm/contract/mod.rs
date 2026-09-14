use super::*;

#[test]
fn packed_pointer_requires_alignment_and_checked_addition() {
    let layout = PackedActivationLayout::new(3, SCALE_BLOCK).unwrap();
    assert_eq!(layout.scale_pointer(4096).unwrap(), 4096 + 3 * 128);
    assert!(layout.scale_pointer(4097).is_err());
    assert!(layout.scale_pointer(u64::MAX - 15).is_err());
}

#[test]
fn rendered_quantizer_records_abi_and_has_no_unresolved_placeholders() {
    let source = quantizer_source();
    assert!(source.contains(PACKED_ACTIVATION_ABI));
    assert!(!source.contains('@'));
    // Bit-exact numerical constants must survive the Rust-to-CUDA bridge.
    let literal = |name: &str| {
        let prefix = format!("constexpr float {name} = ");
        source
            .split_once(&prefix)
            .unwrap()
            .1
            .split_once("f;")
            .unwrap()
            .0
            .parse::<f32>()
            .unwrap()
    };
    assert_eq!(literal("kFp8MaxFinite").to_bits(), 448.0_f32.to_bits());
    assert_eq!(
        literal("kQuantizationAmaxFloor").to_bits(),
        1.0e-4_f32.to_bits()
    );
}
