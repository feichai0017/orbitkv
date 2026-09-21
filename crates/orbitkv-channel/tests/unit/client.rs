use super::*;
use crate::PublishLayer;

#[test]
fn publish_chunks_preserve_all_layers_and_pages_within_the_slot_limit() {
    let request = PublishRequest {
        instance_id: "qwen3-8b".to_string(),
        tp_rank: 2,
        pp_rank: 1,
        device_id: 3,
        layers: (0..36)
            .map(|layer| PublishLayer {
                layer_name: format!("model.layers.{layer}.attn"),
                block_ids: (0..192).collect(),
                block_hashes: (0..192).map(|page| vec![page as u8; 32]).collect(),
            })
            .collect(),
    };
    let capacity = crate::arena::DEFAULT_SLOT_CAPACITY;
    let payloads = publish_payloads(&request, capacity).unwrap();
    assert!(payloads.len() > 1);
    let mut reconstructed = request.clone();
    for layer in &mut reconstructed.layers {
        layer.block_ids.clear();
        layer.block_hashes.clear();
    }
    for payload in payloads {
        assert!(payload.len() <= capacity);
        let chunk = PublishRequest::decode(&payload).unwrap();
        assert_eq!(chunk.instance_id, request.instance_id);
        assert_eq!((chunk.tp_rank, chunk.pp_rank, chunk.device_id), (2, 1, 3));
        assert_eq!(chunk.layers.len(), request.layers.len());
        for (layer, original) in chunk.layers.into_iter().zip(&mut reconstructed.layers) {
            assert_eq!(layer.layer_name, original.layer_name);
            original.block_ids.extend(layer.block_ids);
            original.block_hashes.extend(layer.block_hashes);
        }
    }
    assert_eq!(reconstructed, request);
    assert!(publish_payloads(&request, 32).is_err());
    assert_eq!(
        publish_payloads(&request, usize::MAX).unwrap(),
        vec![request.encode().unwrap()]
    );
}

#[test]
fn publish_chunks_preserve_ragged_cache_groups() {
    let request = PublishRequest {
        instance_id: "hybrid".to_string(),
        tp_rank: 0,
        pp_rank: 0,
        device_id: 0,
        layers: (0..6)
            .map(|index| PublishLayer {
                layer_name: format!("layer.{index}"),
                block_ids: (0..index * 4).collect(),
                block_hashes: (0..index * 4)
                    .map(|page| vec![page as u8; 8 + page as usize])
                    .collect(),
            })
            .collect(),
    };
    let mut reconstructed = request.clone();
    for layer in &mut reconstructed.layers {
        layer.block_ids.clear();
        layer.block_hashes.clear();
    }
    for payload in publish_payloads(&request, 512).unwrap() {
        assert!(payload.len() <= 512);
        for layer in PublishRequest::decode(&payload).unwrap().layers {
            let original = reconstructed
                .layers
                .iter_mut()
                .find(|candidate| candidate.layer_name == layer.layer_name)
                .unwrap();
            original.block_ids.extend(layer.block_ids);
            original.block_hashes.extend(layer.block_hashes);
        }
    }
    assert_eq!(reconstructed, request);
}
