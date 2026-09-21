use super::*;

#[test]
fn validate_direct_io_rejects_unaligned_offset() {
    let iovecs = vec![(SSD_ALIGNMENT, SSD_ALIGNMENT)];
    let err = UringIoEngine::validate_direct_io(iovecs, 1).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn validate_direct_io_rejects_unaligned_buffer() {
    let iovecs = vec![(SSD_ALIGNMENT + 1, SSD_ALIGNMENT)];
    let err = UringIoEngine::validate_direct_io(iovecs, 0).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn validate_direct_io_rejects_unaligned_length() {
    let iovecs = vec![(SSD_ALIGNMENT, SSD_ALIGNMENT - 1)];
    let err = UringIoEngine::validate_direct_io(iovecs, 0).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}
