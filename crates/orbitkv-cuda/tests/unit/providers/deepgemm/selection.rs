use super::*;

#[test]
fn descriptor_pins_a_tile_across_interior_rows() {
    let selection = Selection {
        row_limit: 256,
        config: candidates(CANONICAL_SELECTION_ROWS, 5120, 5120, 78)[0],
    };
    let encoded = serde_json::to_string(&selection).unwrap();
    let restored: Selection = serde_json::from_str(&encoded).unwrap();
    restored.validate().unwrap();
    assert_eq!(restored, selection);
    for rows in [1, 4, 8, 16, 32, 64, 128, 256] {
        restored.validate_rows(rows).unwrap();
        assert_eq!(restored.config, selection.config);
    }
    assert!(restored.validate_rows(0).is_err());
    assert!(restored.validate_rows(257).is_err());
    let mut corrupted = restored;
    corrupted.config.smem_bytes += 1;
    assert!(corrupted.validate().is_err());
    corrupted = restored;
    corrupted.config.num_sms = 0;
    assert!(corrupted.validate().is_err());
}

#[test]
fn primitive_uses_one_numerical_tile_across_row_limits() {
    let mut graph = orbitkv_compiler::egglog_utils::primitives::egglog::EGraph::default();
    graph.add_primitive(
        orbitkv_compiler::egglog_utils::primitives::EgglogPrimitive::new::<TileCandidate>(),
    );
    let expected_config = candidates(CANONICAL_SELECTION_ROWS, 5120, 17408, 78)[0];
    for rows in [1, 4, 8, 32, 256] {
        let selection = Selection {
            row_limit: rows,
            config: expected_config,
        };
        let literal = serde_json::to_string(&serde_json::to_string(&selection).unwrap()).unwrap();
        graph
            .parse_and_run_program(
                None,
                &format!("(check (= (deepgemm-tile-candidate {rows} 5120 17408 78 0) {literal}))"),
            )
            .unwrap();
    }
}

#[test]
fn every_admitted_descriptor_passes_legality_validation() {
    for sms in [1, 7, 78, 132] {
        for rows in [1, 16, 17, 32, 33, 128, 256, 65535] {
            for config in candidates(rows, 5120, 5120, sms) {
                Selection {
                    row_limit: rows,
                    config,
                }
                .validate()
                .unwrap();
            }
        }
    }
    for (m, n, k, sms) in [
        (0, 128, 128, 78),
        (1, 127, 128, 78),
        (1, 128, 127, 78),
        (1, 128, 128, 0),
        (65536, 128, 128, 78),
    ] {
        assert!(candidates(m, n, k, sms).is_empty());
    }
}

#[test]
fn primitive_is_partial_and_returns_a_serializable_descriptor() {
    let mut graph = orbitkv_compiler::egglog_utils::primitives::egglog::EGraph::default();
    graph.add_primitive(
        orbitkv_compiler::egglog_utils::primitives::EgglogPrimitive::new::<TileCandidate>(),
    );
    let selection = Selection {
        row_limit: 32,
        config: candidates(CANONICAL_SELECTION_ROWS, 128, 128, 78)[0],
    };
    let literal = serde_json::to_string(&serde_json::to_string(&selection).unwrap()).unwrap();
    graph
        .parse_and_run_program(
            None,
            &format!("(check (= (deepgemm-tile-candidate 32 128 128 78 0) {literal}))"),
        )
        .unwrap();
    graph.parse_and_run_program(None, "(relation row-value (i64)) (relation candidate-value (String)) (row-value -1) (row-value 0) (row-value 65536) (rule ((row-value m) (= result (deepgemm-tile-candidate m 128 128 78 0))) ((candidate-value result))) (run 1)").unwrap();
    assert!(
        graph
            .parse_and_run_program(
                None,
                "(check (= (deepgemm-tile-candidate 32 128 128 78 4) \"invalid\"))"
            )
            .is_err()
    );
}
