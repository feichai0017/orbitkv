use super::*;

#[tokio::test]
async fn segmented_roundtrip_detects_corruption_and_releases_budget() {
    let codec = Codec::new(65536).unwrap();
    let a = vec![0x81; 8192];
    let b: Vec<_> = (0..16384).map(|i| (i % 31) as u8).collect();
    let (encoding, mut buffer) = codec.encode(&[&a, &b], 4096).unwrap();
    assert_eq!(buffer.len, 4096);
    assert_eq!(buffer.ptr() as usize % ALIGNMENT, 0);
    assert!(codec.budget.available_permits() < codec.capacity);
    let mut out_a = vec![0; a.len()];
    let mut out_b = vec![0; b.len()];
    decode(&encoding, &mut buffer, &mut [&mut out_a, &mut out_b]).unwrap();
    assert_eq!((out_a.clone(), out_b.clone()), (a, b));

    let Encoding::Lz4V1(mut metadata) = encoding.clone() else {
        unreachable!()
    };
    metadata[1].checksum ^= 1;
    assert!(
        decode(
            &Encoding::Lz4V1(metadata),
            &mut buffer,
            &mut [&mut out_a, &mut out_b]
        )
        .is_err()
    );
    assert!(decode(&encoding, &mut buffer, &mut [&mut out_a]).is_err());
    out_b.truncate(10);
    assert!(decode(&encoding, &mut buffer, &mut [&mut out_a, &mut out_b]).is_err());
    buffer.len = 1;
    assert!(decode(&encoding, &mut buffer, &mut [&mut out_a, &mut out_b]).is_err());
    drop(buffer);
    assert_eq!(codec.budget.available_permits(), codec.capacity);
    let read = codec.read_buffer(4096).await.unwrap();
    assert_eq!(codec.budget.available_permits(), codec.capacity - 4096);
    drop(read);
    assert_eq!(codec.budget.available_permits(), codec.capacity);
}

#[tokio::test]
async fn alignment_entropy_and_budget_can_all_choose_raw() {
    use rand::{Rng, SeedableRng};
    let codec = Codec::new(32768).unwrap();
    assert!(codec.encode(&[&[0; 512]], 4096).is_none());
    let mut random = vec![0; 8192];
    rand::rngs::StdRng::seed_from_u64(42).fill_bytes(&mut random);
    assert!(codec.encode(&[&random], 512).is_none());
    assert!(codec.encode(&[&[0; 65536]], 512).is_none());
    let held = codec.read_buffer(32768).await.unwrap();
    assert!(codec.encode(&[&[0; 8192]], 512).is_none());
    drop(held);
    assert!(codec.encode(&[&[0; 8192]], 512).is_some());
    assert!(codec.read_buffer(32769).await.is_err());
    assert!(Codec::new(0).is_err());
}
