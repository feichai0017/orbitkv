use super::*;

#[test]
fn storage_identity_is_stable_and_isolates_incompatible_slots() {
    let first = StorageSlot {
        format: crate::StorageFormat::Exact,
        layer: "layer.0".into(),
        group: 0,
        tp_rank: 0,
        pp_rank: 0,
        segment_bytes: 128,
        padded_block_bytes: 256,
        split: true,
    };
    let mut second = first.clone();
    second.layer = "layer.1".into();
    let namespace = storage_namespace("model-v1", false, vec![first.clone(), second.clone()]);
    assert_eq!(
        namespace,
        storage_namespace(
            "model-v1",
            false,
            vec![second.clone(), first.clone(), first.clone()]
        )
    );
    for changed in [
        StorageSlot {
            format: crate::StorageFormat::Fp8FromBf16,
            ..first.clone()
        },
        StorageSlot {
            group: 1,
            ..first.clone()
        },
        StorageSlot {
            tp_rank: 1,
            ..first.clone()
        },
        StorageSlot {
            pp_rank: 1,
            ..first.clone()
        },
        StorageSlot {
            segment_bytes: 256,
            ..first.clone()
        },
        StorageSlot {
            padded_block_bytes: 512,
            ..first.clone()
        },
        StorageSlot {
            split: false,
            ..first.clone()
        },
    ] {
        assert_ne!(
            namespace,
            storage_namespace("model-v1", false, vec![changed, second.clone()])
        );
    }
    assert_ne!(
        namespace,
        storage_namespace("model-v2", false, vec![first.clone(), second.clone()])
    );
    assert_ne!(
        namespace,
        storage_namespace("model-v1", true, vec![first, second])
    );
}

#[test]
fn group_encoding_is_versioned_and_unambiguous_for_every_hash_length() {
    assert_eq!(
        group_hash(&[0xaa, 0xbb], 0),
        [
            b"OKS\x01".as_slice(),
            &0u32.to_le_bytes(),
            &2u64.to_le_bytes(),
            &[0xaa, 0xbb]
        ]
        .concat()
    );
    assert_ne!(group_hash(&[1, 2], 0), vec![1, 2]);
    assert_ne!(group_hash(&[1, 2], 1), group_hash(&[1, 2], 2));
    assert_ne!(group_hash(&[1, 2], 1), group_hash(&[1, 2, 0, 0, 0, 1], 0));
}
