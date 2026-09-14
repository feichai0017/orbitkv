use super::*;

fn delta() -> IntExpr {
    IntExpr::Sub {
        lhs: Box::new(IntExpr::QueryPosition),
        rhs: Box::new(IntExpr::KeyPosition),
    }
}

fn state(predicate: Predicate) -> RetentionStateDecl {
    RetentionStateDecl {
        name: "state".into(),
        layers: vec![0],
        kv_head_range: None,
        bytes_per_token_per_layer: 128,
        may_read: predicate,
    }
}

#[test]
fn infers_fixed_window_from_difference_bound() {
    let analysis = analyze_state(&state(Predicate::LessThan {
        lhs: delta(),
        rhs: IntExpr::Constant { value: 32 },
    }))
    .unwrap();
    assert_eq!(
        analysis.inferred,
        InferredRetention::FixedWindow { window_tokens: 32 }
    );
    assert_eq!(analysis.proven_query_key_delta_upper_bound, Some(31));
}

#[test]
fn unrecognized_affine_relation_fails_closed_to_unbounded() {
    let analysis = analyze_state(&state(Predicate::LessThan {
        lhs: IntExpr::Add {
            lhs: Box::new(IntExpr::QueryPosition),
            rhs: Box::new(IntExpr::KeyPosition),
        },
        rhs: IntExpr::Constant { value: 128 },
    }))
    .unwrap();
    assert_eq!(analysis.inferred, InferredRetention::Unbounded);
}

#[test]
fn finite_or_takes_the_widest_branch() {
    let analysis = analyze_state(&state(Predicate::Or {
        terms: vec![
            Predicate::LessThan {
                lhs: delta(),
                rhs: IntExpr::Constant { value: 16 },
            },
            Predicate::LessEqual {
                lhs: delta(),
                rhs: IntExpr::Constant { value: 63 },
            },
        ],
    }))
    .unwrap();
    assert_eq!(
        analysis.inferred,
        InferredRetention::FixedWindow { window_tokens: 64 }
    );
}

#[test]
fn unbounded_or_branch_fails_closed_to_unbounded() {
    let analysis = analyze_state(&state(Predicate::Or {
        terms: vec![
            Predicate::LessThan {
                lhs: delta(),
                rhs: IntExpr::Constant { value: 16 },
            },
            Predicate::True,
        ],
    }))
    .unwrap();
    assert_eq!(analysis.inferred, InferredRetention::Unbounded);
}

#[test]
fn splits_sink_or_sliding_into_lifetime_regions() {
    let analysis = analyze_state(&state(Predicate::Or {
        terms: vec![
            Predicate::LessThan {
                lhs: IntExpr::KeyPosition,
                rhs: IntExpr::Constant { value: 16 },
            },
            Predicate::LessThan {
                lhs: delta(),
                rhs: IntExpr::Constant { value: 32 },
            },
        ],
    }))
    .unwrap();
    assert_eq!(
        analysis.inferred,
        InferredRetention::Partitioned {
            regions: vec![
                InferredRegion {
                    label: "sink".into(),
                    start_token: 0,
                    end_token_exclusive: Some(16),
                    retention: AtomicRetention::Unbounded,
                },
                InferredRegion {
                    label: "local".into(),
                    start_token: 16,
                    end_token_exclusive: None,
                    retention: AtomicRetention::FixedWindow { window_tokens: 32 },
                },
            ]
        }
    );
}

#[test]
fn non_affine_key_prefix_fails_closed_to_unbounded() {
    let declaration = state(Predicate::Or {
        terms: vec![
            Predicate::LessThan {
                lhs: IntExpr::Add {
                    lhs: Box::new(IntExpr::KeyPosition),
                    rhs: Box::new(IntExpr::FloorDiv {
                        value: Box::new(IntExpr::Sub {
                            lhs: Box::new(IntExpr::Constant { value: 0 }),
                            rhs: Box::new(IntExpr::QueryPosition),
                        }),
                        divisor: 2,
                    }),
                },
                rhs: IntExpr::Constant { value: 4 },
            },
            Predicate::LessThan {
                lhs: delta(),
                rhs: IntExpr::Constant { value: 8 },
            },
        ],
    });
    let analysis = analyze_state(&declaration).unwrap();
    assert_eq!(analysis.inferred, InferredRetention::Unbounded);
    assert_eq!(analysis.proven_query_key_delta_upper_bound, None);
    assert!(declaration.may_read.may_read(100, 50));
}

#[test]
fn inferred_window_matches_exhaustive_relation() {
    for window in 1..=65_i64 {
        let declaration = state(Predicate::LessThan {
            lhs: delta(),
            rhs: IntExpr::Constant { value: window },
        });
        let analysis = analyze_state(&declaration).unwrap();
        let maximum_delta = (0..=window * 3)
            .flat_map(|query| (0..=query).map(move |key| (query, key)))
            .filter(|&(query, key)| declaration.may_read.may_read(query, key))
            .map(|(query, key)| query - key)
            .max()
            .unwrap();
        assert_eq!(
            analysis.proven_query_key_delta_upper_bound,
            Some(u64::try_from(maximum_delta).unwrap())
        );
        assert_eq!(
            analysis.inferred,
            InferredRetention::FixedWindow {
                window_tokens: u64::try_from(window).unwrap()
            }
        );
    }
}

#[test]
fn infers_same_chunk_lifetime_from_floor_division() {
    let declaration = state(Predicate::Equal {
        lhs: IntExpr::FloorDiv {
            value: Box::new(IntExpr::QueryPosition),
            divisor: 16,
        },
        rhs: IntExpr::FloorDiv {
            value: Box::new(IntExpr::KeyPosition),
            divisor: 16,
        },
    });
    let analysis = analyze_state(&declaration).unwrap();
    assert_eq!(
        analysis.inferred,
        InferredRetention::Chunked { chunk_tokens: 16 }
    );
    assert_eq!(analysis.proven_query_key_delta_upper_bound, Some(15));
    for query in 0..64_i64 {
        for key in 0..=query {
            assert_eq!(
                declaration.may_read.may_read(query, key),
                query / 16 == key / 16
            );
        }
    }
}

#[test]
fn infers_dilated_window_last_read_exactly() {
    for window in 1..=65_i64 {
        for dilation in 1..=17_i64 {
            let declaration = state(Predicate::And {
                terms: vec![
                    Predicate::LessThan {
                        lhs: delta(),
                        rhs: IntExpr::Constant { value: window },
                    },
                    Predicate::Equal {
                        lhs: IntExpr::Mod {
                            value: Box::new(delta()),
                            modulus: dilation,
                        },
                        rhs: IntExpr::Constant { value: 0 },
                    },
                ],
            });
            let analysis = analyze_state(&declaration).unwrap();
            let maximum_delta = (0..=window * 3)
                .flat_map(|query| (0..=query).map(move |key| (query, key)))
                .filter(|&(query, key)| declaration.may_read.may_read(query, key))
                .map(|(query, key)| query - key)
                .max()
                .unwrap();
            assert_eq!(
                analysis.proven_query_key_delta_upper_bound,
                Some(u64::try_from(maximum_delta).unwrap())
            );
            assert_eq!(
                analysis.inferred,
                InferredRetention::FixedWindow {
                    window_tokens: u64::try_from(maximum_delta + 1).unwrap()
                }
            );
        }
    }
}

#[test]
fn invalid_modulus_fails_closed() {
    let declaration = state(Predicate::Equal {
        lhs: IntExpr::Mod {
            value: Box::new(delta()),
            modulus: 0,
        },
        rhs: IntExpr::Constant { value: 0 },
    });
    assert_eq!(
        analyze_state(&declaration),
        Err(RetentionError::InvalidModulus)
    );
}

#[test]
fn invalid_floor_divisor_fails_closed() {
    let declaration = state(Predicate::Equal {
        lhs: IntExpr::FloorDiv {
            value: Box::new(IntExpr::QueryPosition),
            divisor: 0,
        },
        rhs: IntExpr::FloorDiv {
            value: Box::new(IntExpr::KeyPosition),
            divisor: 0,
        },
    });
    assert_eq!(
        analyze_state(&declaration),
        Err(RetentionError::InvalidFloorDivisor)
    );
}
