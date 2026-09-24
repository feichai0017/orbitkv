use super::*;

#[tokio::test]
async fn mixed_quantized_and_exact_segments_validate_and_release_budget() {
    let codec = Codec::new(65536).unwrap();
    let a = [0xa0, 0x3f].repeat(8192); // BF16 1.25 is exactly representable in FP8.
    let b: Vec<_> = (0..4096).map(|i| (i % 251) as u8).collect();
    let (encoding, mut buffer) = codec
        .encode(
            &[(&a, StorageFormat::Fp8FromBf16), (&b, StorageFormat::Exact)],
            4096,
        )
        .unwrap();
    assert_eq!(buffer.len, 12288);
    assert_eq!(buffer.ptr() as usize % ALIGNMENT, 0);
    let mut out_a = vec![0; a.len()];
    let mut out_b = vec![0; b.len()];
    decode(&encoding, &mut buffer, &mut [&mut out_a, &mut out_b]).unwrap();
    assert_eq!((&out_a, &out_b), (&a, &b));
    buffer.as_mut_slice()[0] ^= 1;
    assert!(decode(&encoding, &mut buffer, &mut [&mut out_a, &mut out_b]).is_err());
    buffer.as_mut_slice()[0] ^= 1;
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
async fn exact_states_alignment_outliers_and_budget_choose_raw() {
    let codec = Codec::new(32768).unwrap();
    let format = StorageFormat::Fp8FromBf16;
    assert!(codec.encode(&[(&[0; 512], format)], 4096).is_none());
    assert!(
        codec
            .encode(&[(&[0; 8192], StorageFormat::Exact)], 512)
            .is_none()
    );
    assert!(codec.encode(&[(&[0; 131072], format)], 512).is_none());
    for bits in [0x7f80u16, 0x7fc0, 0x4480] {
        // inf, NaN, 1024
        assert!(
            codec
                .encode(&[(&bits.to_le_bytes().repeat(4096), format)], 512)
                .is_none()
        );
    }
    let held = codec.read_buffer(32768).await.unwrap();
    assert!(codec.encode(&[(&[0; 8192], format)], 512).is_none());
    drop(held);
    assert!(codec.encode(&[(&[0; 8192], format)], 512).is_some());
    assert!(codec.read_buffer(32769).await.is_err());
    assert!(Codec::new(0).is_err());
}
