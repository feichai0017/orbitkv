use super::*;

#[test]
fn rejects_incompatible_and_unbounded_frames_before_allocating() {
    let header = LifecycleHeader {
        code: 2,
        epoch: 42,
        payload_len: 512,
    };
    let mut bytes = header.encode().unwrap();
    assert_eq!(LifecycleHeader::decode(bytes).unwrap().epoch, 42);
    bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(LifecycleHeader::decode(bytes).is_err());
    bytes = header.encode().unwrap();
    bytes[4] = 99;
    assert!(LifecycleHeader::decode(bytes).is_err());
}
