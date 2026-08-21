use super::*;

fn sliding_input(window: u64, page: u64) -> KvPlanInput {
    KvPlanInput {
        page_tokens: page,
        classes: vec![KvClassSpec {
            name: "swa".into(),
            layers: vec![0],
            retention: RetentionKind::Sliding,
            bytes_per_token_per_layer: 128,
            window_tokens: Some(window),
        }],
    }
}

#[test]
fn sliding_formula_matches_exhaustive_overlap() {
    for page in [1, 4, 16] {
        for window in 1..=65 {
            let plan = compile_plan(sliding_input(window, page)).unwrap();
            let mut maximum = 0;
            for query in 0..window + 2 * page {
                let first_key = query.saturating_sub(window - 1);
                let first_block = first_key / page;
                let last_block = query / page;
                maximum = maximum.max(last_block - first_block + 1);
            }
            assert_eq!(plan.classes[0].slot_count, Some(maximum));
        }
    }
}

#[test]
fn applicability_distinguishes_full_uniform_and_hybrid_lifetimes() {
    let full = compile_plan(KvPlanInput {
        page_tokens: 16,
        classes: vec![KvClassSpec {
            name: "full".into(),
            layers: vec![0, 1],
            retention: RetentionKind::Full,
            bytes_per_token_per_layer: 128,
            window_tokens: None,
        }],
    })
    .unwrap()
    .applicability_report(8192)
    .unwrap();
    assert_eq!(full.applicability, ApplicabilityClass::SafeFallback);
    assert_eq!(full.static_reduction_percent_milli, 0);
    assert_eq!(full.generated_layouts, vec!["append_only"]);
    assert_eq!(full.bounded_class_count, 0);

    let sliding = compile_plan(sliding_input(4096, 16))
        .unwrap()
        .applicability_report(8192)
        .unwrap();
    assert_eq!(sliding.applicability, ApplicabilityClass::UniformBounded);
    assert_eq!(sliding.generated_layouts, vec!["periodic"]);
    assert_eq!(sliding.bounded_class_count, 1);
    assert!(sliding.static_reduction_percent_milli > 49_000);

    let hybrid = compile_plan(KvPlanInput {
        page_tokens: 16,
        classes: vec![
            KvClassSpec {
                name: "full".into(),
                layers: vec![0],
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 128,
                window_tokens: None,
            },
            KvClassSpec {
                name: "swa".into(),
                layers: vec![1],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(1024),
            },
        ],
    })
    .unwrap()
    .applicability_report(8192)
    .unwrap();
    assert_eq!(hybrid.applicability, ApplicabilityClass::HybridLifetimes);
    assert_eq!(hybrid.generated_layouts, vec!["append_only", "periodic"]);
    assert_eq!(hybrid.bounded_class_count, 1);
    assert_eq!(hybrid.unbounded_class_count, 1);
}

#[test]
fn continuation_cut_has_w_minus_one_old_tokens() {
    let plan = compile_plan(sliding_input(32, 16)).unwrap();
    assert_eq!(plan.continuation_blocks(64).unwrap()["swa"], vec![2, 3]);
}

#[test]
fn full_and_swa_capacity_matches_page_geometry() {
    let plan = compile_plan(KvPlanInput {
        page_tokens: 16,
        classes: vec![
            KvClassSpec {
                name: "full".into(),
                layers: (0..10).collect(),
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 4096,
                window_tokens: None,
            },
            KvClassSpec {
                name: "swa".into(),
                layers: (10..62).collect(),
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 4096,
                window_tokens: Some(1024),
            },
        ],
    })
    .unwrap();
    let resident = plan.resident_bytes_at(32_768).unwrap();
    let baseline = plan.all_full_baseline_bytes_at(32_768).unwrap();
    assert_eq!(resident, 1_563_688_960);
    assert_eq!(baseline, 8_321_499_136);
    let layout = plan.layout_program().unwrap();
    assert_eq!(layout.schema, "orbitkv.layout-program.v1");
    assert_eq!(layout.plan_fingerprint, plan.fingerprint());
    assert_eq!(layout.classes[0].address, AddressProgram::AppendOnly);
    assert_eq!(
        layout.classes[1].address,
        AddressProgram::Periodic { period_blocks: 65 }
    );
    assert_eq!(
        layout.classes[1].retirement,
        RetirementProgram::BlockEndPlus {
            offset_tokens: 1023
        }
    );
}

#[test]
fn periodic_address_program_derives_cell_and_cycle() {
    let address = AddressProgram::Periodic { period_blocks: 65 }
        .evaluate("request-7", "swa", 131)
        .unwrap();
    assert_eq!(address.cell.request_id, "request-7");
    assert_eq!(address.cell.class_name, "swa");
    assert_eq!(address.cell.cell_index, 1);
    assert_eq!(address.version.cycle, 2);
}

#[test]
fn retirement_program_matches_sliding_death_formula() {
    assert_eq!(
        RetirementProgram::BlockEndPlus {
            offset_tokens: 1023
        }
        .death_boundary(16, 0)
        .unwrap(),
        Some(1039)
    );
    assert_eq!(
        RetirementProgram::Never
            .death_boundary(16, u64::MAX)
            .unwrap(),
        None
    );
}

#[test]
fn canonical_manager_plan_and_retention_ir_compile_identically() {
    let input = KvPlanInput {
        page_tokens: 16,
        classes: vec![
            KvClassSpec {
                name: "full".into(),
                layers: vec![0],
                retention: RetentionKind::Full,
                bytes_per_token_per_layer: 128,
                window_tokens: None,
            },
            KvClassSpec {
                name: "swa".into(),
                layers: vec![1],
                retention: RetentionKind::Sliding,
                bytes_per_token_per_layer: 128,
                window_tokens: Some(1024),
            },
        ],
    };
    let canonical_plan = compile_plan(input.clone()).unwrap();
    let retention_plan =
        compile_retention_program(input.into_retention_program().unwrap()).unwrap();
    assert_eq!(canonical_plan, retention_plan);
    assert_eq!(
        canonical_plan.layout_program().unwrap(),
        retention_plan.layout_program().unwrap()
    );
    assert_eq!(canonical_plan.fingerprint(), retention_plan.fingerprint());
}

#[test]
fn sink_sliding_relation_synthesizes_pinned_and_periodic_regions() {
    let program = RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 4,
        states: vec![RetentionStateDecl {
            name: "attention".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::Or {
                terms: vec![
                    Predicate::LessThan {
                        lhs: IntExpr::KeyPosition,
                        rhs: IntExpr::Constant { value: 4 },
                    },
                    Predicate::LessThan {
                        lhs: IntExpr::Sub {
                            lhs: Box::new(IntExpr::QueryPosition),
                            rhs: Box::new(IntExpr::KeyPosition),
                        },
                        rhs: IntExpr::Constant { value: 8 },
                    },
                ],
            },
        }],
    };
    let plan = compile_retention_program(program).unwrap();
    assert_eq!(plan.classes.len(), 2);
    assert_eq!(plan.classes[0].spec.name, "attention::sink");
    assert_eq!(plan.classes[0].slot_count, Some(1));
    assert_eq!(
        plan.classes[0].block_domain,
        BlockDomain {
            start_block: 0,
            end_block_exclusive: Some(1)
        }
    );
    assert_eq!(plan.classes[1].spec.name, "attention::local");
    assert_eq!(plan.classes[1].slot_count, Some(3));
    assert_eq!(plan.classes[1].block_domain.start_block, 1);

    let layout = plan.layout_program().unwrap();
    assert_eq!(layout.classes[0].address, AddressProgram::Pinned);
    assert_eq!(
        layout.classes[1].address,
        AddressProgram::PeriodicFrom {
            period_blocks: 3,
            origin_block: 1
        }
    );
    let continuation = plan.continuation_blocks(20).unwrap();
    assert_eq!(continuation["attention::sink"], vec![0]);
    assert_eq!(continuation["attention::local"], vec![3, 4]);
    let capacity = plan.capacity_at(20).unwrap();
    assert_eq!(capacity[0].semantic_live_tokens, 4);
    assert_eq!(capacity[0].physical_token_slots, 4);
    assert_eq!(capacity[1].semantic_live_tokens, 7);
    assert_eq!(capacity[1].physical_token_slots, 12);
    assert_eq!(plan.all_full_baseline_bytes_at(20).unwrap(), 20 * 128);

    assert!(matches!(
        layout.temporal_address("request", "attention::sink", 1),
        Err(PlanError::AddressOutsideBlockDomain {
            class,
            ordinal: 1
        }) if class == "attention::sink"
    ));
    assert!(matches!(
        layout.temporal_address("request", "attention::local", 0),
        Err(PlanError::AddressOutsideBlockDomain {
            class,
            ordinal: 0
        }) if class == "attention::local"
    ));
}

#[test]
fn sink_boundary_must_align_with_reclamation_page() {
    let program = RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 4,
        states: vec![RetentionStateDecl {
            name: "attention".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::Or {
                terms: vec![
                    Predicate::LessThan {
                        lhs: IntExpr::KeyPosition,
                        rhs: IntExpr::Constant { value: 3 },
                    },
                    Predicate::LessThan {
                        lhs: IntExpr::Sub {
                            lhs: Box::new(IntExpr::QueryPosition),
                            rhs: Box::new(IntExpr::KeyPosition),
                        },
                        rhs: IntExpr::Constant { value: 8 },
                    },
                ],
            },
        }],
    };
    assert!(matches!(
        compile_retention_program(program),
        Err(PlanError::SinkBoundaryNotPageAligned {
            sink_tokens: 3,
            page_tokens: 4
        })
    ));
}

#[test]
fn same_chunk_relation_synthesizes_resettable_arena() {
    let program = RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 4,
        states: vec![RetentionStateDecl {
            name: "chunked".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::Equal {
                lhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::QueryPosition),
                    divisor: 16,
                },
                rhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::KeyPosition),
                    divisor: 16,
                },
            },
        }],
    };
    let plan = compile_retention_program(program).unwrap();
    assert_eq!(plan.classes[0].spec.retention, RetentionKind::Chunked);
    assert_eq!(plan.classes[0].chunk_tokens, Some(16));
    assert_eq!(plan.classes[0].slot_count, Some(4));
    let layout = plan.layout_program().unwrap();
    assert_eq!(
        layout.classes[0].address,
        AddressProgram::ResettableArena {
            blocks_per_epoch: 4
        }
    );
    assert_eq!(
        layout.classes[0].retirement,
        RetirementProgram::EpochEnd {
            blocks_per_epoch: 4
        }
    );
    for ordinal in 0..4 {
        assert_eq!(
            layout.classes[0]
                .retirement
                .death_boundary(4, ordinal)
                .unwrap(),
            Some(16)
        );
    }
    assert_eq!(
        layout.classes[0].retirement.death_boundary(4, 4).unwrap(),
        Some(32)
    );
    assert_eq!(plan.continuation_blocks(20).unwrap()["chunked"], vec![4]);
    assert_eq!(plan.capacity_at(20).unwrap()[0].semantic_live_tokens, 4);
}

#[test]
fn chunk_size_must_align_with_reclamation_page() {
    let program = RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 4,
        states: vec![RetentionStateDecl {
            name: "chunked".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::Equal {
                lhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::QueryPosition),
                    divisor: 10,
                },
                rhs: IntExpr::FloorDiv {
                    value: Box::new(IntExpr::KeyPosition),
                    divisor: 10,
                },
            },
        }],
    };
    assert_eq!(
        compile_retention_program(program),
        Err(PlanError::ChunkNotPageAligned {
            chunk_tokens: 10,
            page_tokens: 4
        })
    );
}

#[test]
fn lifetime_normalization_matches_multi_scale_head_theory() {
    let state = |name: &str, start: u32, end_exclusive: u32, window: i64| -> RetentionStateDecl {
        RetentionStateDecl {
            name: name.into(),
            layers: vec![0],
            kv_head_range: Some(KvHeadRange {
                start,
                end_exclusive,
            }),
            bytes_per_token_per_layer: u64::from(end_exclusive - start) * 512,
            may_read: Predicate::LessThan {
                lhs: IntExpr::Sub {
                    lhs: Box::new(IntExpr::QueryPosition),
                    rhs: Box::new(IntExpr::KeyPosition),
                },
                rhs: IntExpr::Constant { value: window },
            },
        }
    };
    let plan = compile_retention_program(RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 16,
        states: vec![
            state("w512", 0, 8, 512),
            state("w2048", 8, 16, 2048),
            state("w8192", 16, 32, 8192),
        ],
    })
    .unwrap();
    assert_eq!(plan.classes[0].slot_count, Some(33));
    assert_eq!(plan.classes[1].slot_count, Some(129));
    assert_eq!(plan.classes[2].slot_count, Some(513));
    let report = plan.lifetime_normalization_report().unwrap();
    let unit_bytes = 16 * 512;
    assert_eq!(report.normalized_bytes_per_request, 9504 * unit_bytes);
    assert_eq!(
        report.max_window_baseline_bytes_per_request,
        16416 * unit_bytes
    );
    assert_eq!(report.savings_bytes_per_request, 6912 * unit_bytes);
    assert_eq!(report.savings_percent_milli, 42_105);
    assert_eq!(report.retention_amplification_milli, 1727);
}

#[test]
fn overlapping_head_ranges_fail_closed() {
    let state = |name: &str, start: u32, end_exclusive: u32| RetentionStateDecl {
        name: name.into(),
        layers: vec![0],
        kv_head_range: Some(KvHeadRange {
            start,
            end_exclusive,
        }),
        bytes_per_token_per_layer: u64::from(end_exclusive - start) * 512,
        may_read: Predicate::LessThan {
            lhs: IntExpr::Sub {
                lhs: Box::new(IntExpr::QueryPosition),
                rhs: Box::new(IntExpr::KeyPosition),
            },
            rhs: IntExpr::Constant { value: 512 },
        },
    };
    assert!(matches!(
        compile_retention_program(RetentionProgramInput {
            schema: "orbitkv.retention-ir.v1".into(),
            page_tokens: 16,
            states: vec![state("a", 0, 8), state("b", 4, 12)],
        }),
        Err(PlanError::KvHeadOverlap {
            layer: 0,
            first,
            second
        }) if first == "a" && second == "b"
    ));
}

#[test]
fn partitioned_retention_rejects_zero_page_tokens_before_lowering() {
    let program = RetentionProgramInput {
        schema: "orbitkv.retention-ir.v1".into(),
        page_tokens: 0,
        states: vec![RetentionStateDecl {
            name: "attention".into(),
            layers: vec![0],
            kv_head_range: None,
            bytes_per_token_per_layer: 128,
            may_read: Predicate::True,
        }],
    };
    assert_eq!(
        compile_retention_program(program),
        Err(PlanError::ZeroPageTokens)
    );
}

#[test]
fn sink_sliding_continuation_matches_declared_relation_exhaustively() {
    for page_tokens in [1_u64, 2, 4, 8] {
        for window_tokens in 1_u64..=17 {
            let sink_tokens = 2 * page_tokens;
            let declaration = RetentionStateDecl {
                name: "attention".into(),
                layers: vec![0],
                kv_head_range: None,
                bytes_per_token_per_layer: 128,
                may_read: Predicate::Or {
                    terms: vec![
                        Predicate::LessThan {
                            lhs: IntExpr::KeyPosition,
                            rhs: IntExpr::Constant {
                                value: i64::try_from(sink_tokens).unwrap(),
                            },
                        },
                        Predicate::LessThan {
                            lhs: IntExpr::Sub {
                                lhs: Box::new(IntExpr::QueryPosition),
                                rhs: Box::new(IntExpr::KeyPosition),
                            },
                            rhs: IntExpr::Constant {
                                value: i64::try_from(window_tokens).unwrap(),
                            },
                        },
                    ],
                },
            };
            let plan = compile_retention_program(RetentionProgramInput {
                schema: "orbitkv.retention-ir.v1".into(),
                page_tokens,
                states: vec![declaration.clone()],
            })
            .unwrap();
            let layout = plan.layout_program().unwrap();

            for boundary in 0_u64..=8 * page_tokens + 2 * window_tokens {
                let continuation = plan.continuation_blocks(boundary).unwrap();
                let mut expected_sink = BTreeSet::new();
                let mut expected_local = BTreeSet::new();
                for key in 0..boundary {
                    if !declaration.may_read.may_read(
                        i64::try_from(boundary).unwrap(),
                        i64::try_from(key).unwrap(),
                    ) {
                        continue;
                    }
                    let block = key / page_tokens;
                    if key < sink_tokens {
                        expected_sink.insert(block);
                    } else {
                        expected_local.insert(block);
                    }
                }
                assert_eq!(
                    continuation["attention::sink"],
                    expected_sink.into_iter().collect::<Vec<_>>()
                );
                assert_eq!(
                    continuation["attention::local"],
                    expected_local.iter().copied().collect::<Vec<_>>()
                );

                let mut live_cells = BTreeSet::new();
                for ordinal in expected_local {
                    let address = layout
                        .temporal_address("request", "attention::local", ordinal)
                        .unwrap();
                    assert!(
                        live_cells.insert(address.cell.cell_index),
                        "live local blocks collided at page={page_tokens}, window={window_tokens}, boundary={boundary}"
                    );
                }
            }
        }
    }
}
