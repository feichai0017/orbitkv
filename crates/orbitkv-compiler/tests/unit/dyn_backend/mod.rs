use super::*;

#[test]
fn empty_bytes_preserve_reference_dtype() {
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::F32),
        ReferenceData::F32(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::F64),
        ReferenceData::F64(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::F16),
        ReferenceData::F16(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::Bf16),
        ReferenceData::Bf16(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::Int),
        ReferenceData::Int(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::I64),
        ReferenceData::I64(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::I8),
        ReferenceData::I8(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::U8),
        ReferenceData::U8(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::I16),
        ReferenceData::I16(values) if values.is_empty()
    ));
    assert!(matches!(
        bytes_to_reference_data(Vec::new(), DType::Bool),
        ReferenceData::Bool(values) if values.is_empty()
    ));
}

#[test]
fn narrow_integer_bytes_preserve_width_and_signedness() {
    assert!(matches!(
        bytes_to_reference_data(vec![0x80, 0xff, 0x7f], DType::I8),
        ReferenceData::I8(values) if values == [-128, -1, 127]
    ));
    assert!(matches!(
        bytes_to_reference_data(vec![0, 128, 255], DType::U8),
        ReferenceData::U8(values) if values == [0, 128, 255]
    ));
    assert!(matches!(
        bytes_to_reference_data(vec![0x00, 0x80, 0xff, 0x7f], DType::I16),
        ReferenceData::I16(values) if values == [-32_768, 32_767]
    ));
}
