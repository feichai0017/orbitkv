use super::*;
use crate::{AttentionRole, Scalar16, StorageFormat};

#[test]
fn storage_formats_map_to_bounded_representation_families() {
    for (format, expected) in [
        (StorageFormat::Exact, ReplicaRepresentation::Raw),
        (StorageFormat::Ans, ReplicaRepresentation::Ans),
        (StorageFormat::Ans16, ReplicaRepresentation::Ans),
        (StorageFormat::AnsFp8, ReplicaRepresentation::Ans),
        (StorageFormat::Fp8FromBf16, ReplicaRepresentation::Fp8),
        (StorageFormat::Fp8FromFp16, ReplicaRepresentation::Fp8),
        (StorageFormat::Fp8Native, ReplicaRepresentation::Raw),
        (
            StorageFormat::Attention {
                scalar: Scalar16::Bf16,
                role: AttentionRole::KeyValue,
                head_dim: 128,
                layer_index: 0,
                layer_count: 1,
            },
            ReplicaRepresentation::Raw,
        ),
        (
            StorageFormat::TurboQuant {
                scalar: Scalar16::Fp16,
                role: AttentionRole::Key,
                head_dim: 128,
                seed: 42,
                bits: 4,
            },
            ReplicaRepresentation::TurboQuant,
        ),
    ] {
        assert_eq!(ReplicaRepresentation::from(format), expected);
    }
}
