use std::collections::BTreeMap;

use crate::model::tuning;

use super::*;

fn compile_fixture() -> DecoderCompileConfig {
    DecoderCompileConfig {
        maximum_query_tokens: 32,
        representative_prefill_tokens: 8,
        maximum_batch_size: 4,
        maximum_context_pages: 8,
        representative_context_pages: 2,
        search_graphs: 4,
        search_seed: 1,
    }
}

#[test]
fn dependency_width_must_be_positive_and_defaults_to_coordinate_search() {
    assert!(DecoderTuningProfile::from_json(br#"{"hotspot_max_changes":0}"#).is_err());
    assert_eq!(
        DecoderTuningProfile::from_json(b"{}")
            .unwrap()
            .hotspot_max_changes
            .get(),
        1
    );
}

#[test]
fn tuning_identity_covers_workload_budget_and_experimental_candidates() {
    let config = test_config(4);
    let plan = hybrid_executor_plan();
    let arenas = hybrid_arenas();
    let compile = compile_fixture();
    let identity = |tuning: &DecoderTuningProfile| {
        artifact::decoder_artifact_identity_with_tuning(
            &config,
            &plan,
            &arenas,
            &[],
            DecoderWeightFeatures::default(),
            compile,
            "facts",
            tuning,
        )
        .unwrap()
    };
    let original = identity(&DecoderTuningProfile::default());
    for tuning in [
        DecoderTuningProfile {
            hotspot_max_changes: std::num::NonZeroUsize::new(3).unwrap(),
            ..Default::default()
        },
        DecoderTuningProfile {
            hotspot_candidates: 16,
            ..Default::default()
        },
        DecoderTuningProfile {
            initial_candidates: 4,
            ..Default::default()
        },
        DecoderTuningProfile {
            batch_sizes: vec![1, 2],
            ..Default::default()
        },
        DecoderTuningProfile {
            keep_best: 3,
            ..Default::default()
        },
        DecoderTuningProfile {
            search_time_limit_ms: Some(5000),
            ..Default::default()
        },
        DecoderTuningProfile {
            enable_shared_fp8_quantization: true,
            ..Default::default()
        },
    ] {
        assert_ne!(identity(&tuning), original);
    }
}

#[test]
fn tuning_buckets_supply_valid_ragged_metadata_and_cover_feasible_intervals() {
    let mut graph = Graph::default();
    let decoder = DecoderGraph::build(
        &mut graph,
        &test_config(4),
        DecoderWeightFeatures::default(),
        &hybrid_executor_plan(),
        &hybrid_arenas(),
        &[],
    )
    .unwrap();
    let compile = compile_fixture();
    let tuning = DecoderTuningProfile {
        batch_sizes: vec![1, 2, 4],
        prefill_tokens: vec![2, 16],
        keep_best: 3,
        initial_candidates: 4,
        hotspot_candidates: 16,
        hotspot_max_changes: std::num::NonZeroUsize::new(3).unwrap(),
        search_time_limit_ms: Some(5000),
        ..Default::default()
    };
    let options = tuning::decoder_compile_options(&decoder, compile, &tuning, 16).unwrap();
    assert_eq!(options.keep_best, 3);
    assert_eq!(options.initial_population, 4);
    assert_eq!(options.hotspot_candidates, 16);
    assert_eq!(options.hotspot_max_changes.get(), 3);
    assert_eq!(options.search_time_limit, std::time::Duration::from_secs(5));
    let profiles = options.bucket_representatives.as_ref().unwrap();
    let inputs = representative::RepresentativeInputs::new(&decoder, compile, 16);
    let mut saw_ragged = false;
    for dims in profiles {
        let values = inputs.values(dims);
        let get = |input: orbitkv_compiler::prelude::NodeIndex| {
            values
                .iter()
                .find(|(tensor, _)| *tensor == input)
                .unwrap()
                .1
                .clone()
        };
        let query_indptr = get(decoder.inputs.query_indptr.id);
        let batch = dims[&sym("b")];
        let tokens = dims[&sym("s")];
        assert_eq!(query_indptr.len(), batch + 1);
        assert_eq!(query_indptr.last(), Some(&(i32::try_from(tokens).unwrap())));
        assert!(query_indptr.windows(2).all(|pair| pair[0] < pair[1]));
        let lengths = query_indptr
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>();
        saw_ragged |= lengths.windows(2).any(|pair| pair[0] != pair[1]);
        for (class, dimensions) in decoder.inputs.classes.iter().zip(&decoder.class_dimensions) {
            let pages = get(class.attention.page_indices.id);
            let page_indptr = get(class.attention.page_indptr.id);
            let last = get(class.attention.last_page_len.id);
            let slots = get(class.write_slots.id);
            assert_eq!(pages.len(), dims[&dimensions.context_pages]);
            assert_eq!(page_indptr.len(), batch + 1);
            assert_eq!(last.len(), batch);
            assert_eq!(slots.iter().collect::<BTreeSet<_>>().len(), slots.len());
            for row in 0..batch {
                let row_pages = &pages[usize::try_from(page_indptr[row]).unwrap()
                    ..usize::try_from(page_indptr[row + 1]).unwrap()];
                assert!(!row_pages.is_empty());
                let context_tokens =
                    (row_pages.len() - 1) * 16 + usize::try_from(last[row]).unwrap();
                assert!(context_tokens >= usize::try_from(lengths[row]).unwrap());
                for &slot in &slots[usize::try_from(query_indptr[row]).unwrap()
                    ..usize::try_from(query_indptr[row + 1]).unwrap()]
                {
                    assert!(row_pages.contains(&(slot / 16)));
                }
            }
        }
    }
    assert!(saw_ragged);
    // Every valid runtime (s,b) point selects one profiled combination, even
    // when the independent representatives would have had s < b.
    for s in 1..=32 {
        for b in 1..=4.min(s) {
            let matches = profiles
                .iter()
                .filter(|profile| {
                    ['s', 'b'].into_iter().all(|dim| {
                        let value = if dim == 's' { s } else { b };
                        options.dim_buckets[&Symbol::from(dim)]
                            .iter()
                            .any(|bucket| {
                                bucket.contains(value)
                                    && bucket.contains(profile[&Symbol::from(dim)])
                            })
                    })
                })
                .count();
            assert_eq!(matches, 1, "s={s}, b={b}");
        }
    }
}

#[test]
fn tuning_rejects_ambiguous_representatives_and_unbounded_bucket_growth() {
    let decoder = DecoderGraph::build(
        &mut Graph::default(),
        &test_config(4),
        DecoderWeightFeatures::default(),
        &hybrid_executor_plan(),
        &hybrid_arenas(),
        &[],
    )
    .unwrap();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 32,
        representative_prefill_tokens: 8,
        maximum_batch_size: 4,
        maximum_context_pages: 8,
        representative_context_pages: 2,
        search_graphs: 4,
        search_seed: 1,
    };
    for tuning in [
        DecoderTuningProfile {
            initial_candidates: 0,
            ..Default::default()
        },
        DecoderTuningProfile {
            batch_sizes: vec![2, 2],
            ..Default::default()
        },
        DecoderTuningProfile {
            keep_best: 5,
            ..Default::default()
        },
        DecoderTuningProfile {
            maximum_buckets: 1,
            ..Default::default()
        },
        DecoderTuningProfile {
            search_time_limit_ms: Some(0),
            ..Default::default()
        },
    ] {
        assert!(tuning::decoder_compile_options(&decoder, compile, &tuning, 16).is_err());
    }
    assert!(serde_json::from_str::<DecoderTuningProfile>(r#"{"unknown":true}"#).is_err());
    let options = tuning::decoder_compile_options(
        &decoder,
        compile,
        &DecoderTuningProfile {
            batch_sizes: vec![4],
            ..Default::default()
        },
        16,
    )
    .unwrap();
    let singleton = &options.dim_buckets[&Symbol::from('b')][0];
    assert_eq!((singleton.min, singleton.max), (1, 1));
}

#[test]
fn tuning_obeys_caller_budget_without_an_unrelated_global_ceiling() {
    let decoder = DecoderGraph::build(
        &mut Graph::default(),
        &test_config(4),
        DecoderWeightFeatures::default(),
        &hybrid_executor_plan(),
        &hybrid_arenas(),
        &[],
    )
    .unwrap();
    let compile = DecoderCompileConfig {
        maximum_query_tokens: 32,
        representative_prefill_tokens: 8,
        maximum_batch_size: 4,
        maximum_context_pages: 8,
        representative_context_pages: 2,
        search_graphs: 1,
        search_seed: 1,
    };
    let tuning = DecoderTuningProfile {
        maximum_buckets: 1024,
        ..Default::default()
    };
    let options = tuning::decoder_compile_options(&decoder, compile, &tuning, 16).unwrap();
    assert_eq!(options.limit, 1);
    assert_eq!(options.bucket_representatives.as_ref().unwrap().len(), 2);
    assert!(tuning::decoder_compile_options(&decoder, compile, &tuning, 0).is_err());
}

#[test]
fn tuning_covers_correlated_contexts_across_page_geometries_and_arena_offsets() {
    for page_tokens in [8, 32] {
        let mut plan = hybrid_executor_plan();
        plan.page_tokens = page_tokens;
        for class in &mut plan.classes {
            class.page_tokens = page_tokens;
        }
        let mut arenas = hybrid_arenas();
        // Each attention class has a distinct address range in its own arena.
        arenas[0].backend_base_index = 7;
        arenas[1].backend_base_index = 19;
        let decoder = DecoderGraph::build(
            &mut Graph::default(),
            &test_config(4),
            DecoderWeightFeatures::default(),
            &plan,
            &arenas,
            &[],
        )
        .unwrap();
        let compile = DecoderCompileConfig {
            maximum_query_tokens: 32,
            representative_prefill_tokens: 8,
            maximum_batch_size: 4,
            maximum_context_pages: 8,
            representative_context_pages: 2,
            search_graphs: 1,
            search_seed: 1,
        };
        let tuning = DecoderTuningProfile {
            batch_sizes: vec![1, 3],
            prefill_tokens: vec![2, 12],
            context_pages: vec![1, 3, 5],
            maximum_buckets: 128,
            ..Default::default()
        };
        let page_tokens = usize::try_from(page_tokens).unwrap();
        let options =
            tuning::decoder_compile_options(&decoder, compile, &tuning, page_tokens).unwrap();
        let profiles = options.bucket_representatives.as_ref().unwrap();
        let inputs = representative::RepresentativeInputs::new(&decoder, compile, page_tokens);
        for dims in profiles {
            let values = inputs.values(dims).into_iter().collect::<BTreeMap<_, _>>();
            let query_indptr = &values[&decoder.inputs.query_indptr.id];
            for (class, dimensions) in decoder.inputs.classes.iter().zip(&decoder.class_dimensions)
            {
                let pages = &values[&class.attention.page_indices.id];
                let page_indptr = &values[&class.attention.page_indptr.id];
                let slots = &values[&class.write_slots.id];
                assert_eq!(pages.len(), dims[&dimensions.context_pages]);
                assert_eq!(
                    pages[0],
                    i32::try_from(dimensions.backend_base_index).unwrap()
                );
                assert_eq!(slots.iter().collect::<BTreeSet<_>>().len(), slots.len());
                for row in 0..dims[&sym("b")] {
                    let row_pages = &pages[usize::try_from(page_indptr[row]).unwrap()
                        ..usize::try_from(page_indptr[row + 1]).unwrap()];
                    let row_slots = &slots[usize::try_from(query_indptr[row]).unwrap()
                        ..usize::try_from(query_indptr[row + 1]).unwrap()];
                    for slot in row_slots {
                        let page = usize::try_from(*slot).unwrap() / page_tokens;
                        assert!(row_pages.contains(&i32::try_from(page).unwrap()));
                    }
                }
            }
        }
        // Check every feasible packed shape, including independent context
        // lengths for global and windowed attention, has exactly one profile.
        for s in 1..=compile.maximum_query_tokens {
            for b in 1..=compile.maximum_batch_size.min(s) {
                for global_pages in b..=arenas[0].page_count as usize {
                    for local_pages in b..=arenas[1].page_count as usize {
                        if s > global_pages.min(local_pages) * page_tokens {
                            continue;
                        }
                        let runtime_dims = [
                            (sym("s"), s),
                            (sym("b"), b),
                            (decoder.class_dimensions[0].context_pages, global_pages),
                            (decoder.class_dimensions[1].context_pages, local_pages),
                        ];
                        let matches = profiles
                            .iter()
                            .filter(|profile| {
                                runtime_dims.iter().all(|&(dim, value)| {
                                    options.dim_buckets[&dim].iter().any(|bucket| {
                                        bucket.contains(value) && bucket.contains(profile[&dim])
                                    })
                                })
                            })
                            .count();
                        assert_eq!(
                            matches, 1,
                            "page_tokens={page_tokens}, dims={runtime_dims:?}"
                        );
                    }
                }
            }
        }
    }
}
