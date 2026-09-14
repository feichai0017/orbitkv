use orbitkv_compiler::prelude::*;
use orbitkv_ops::ops::attention::*;

fn spec() -> AttentionSpec {
    AttentionSpec {
        query_heads: 6,
        kv_heads: 2,
        query_key_dim: 48,
        value_dim: 32,
        dtype: DType::Bf16,
        scale: 0.25,
        mask: AttentionMask::Causal,
    }
}

fn inputs(graph: &mut Graph, spec: AttentionSpec) -> AttentionInputs {
    AttentionInputs {
        query: graph
            .tensor((3, spec.query_heads, spec.query_key_dim))
            .as_dtype(spec.dtype),
        query_indptr: graph.tensor(3).as_dtype(DType::Int),
        kv: KvView::Paged(PagedKvView {
            state_class_id: 5,
            key: graph
                .tensor((4, 8, spec.kv_heads, spec.query_key_dim))
                .as_dtype(spec.dtype),
            value: graph
                .tensor((4, 8, spec.kv_heads, spec.value_dim))
                .as_dtype(spec.dtype),
            page_size: 8,
            layout: PagedKvLayout::TokenMajor,
            page_indices: graph.tensor(3).as_dtype(DType::Int),
            page_indptr: graph.tensor(3).as_dtype(DType::Int),
            last_page_len: graph.tensor(2).as_dtype(DType::Int),
        }),
    }
}

#[test]
fn logical_dimensions_and_visibility_do_not_depend_on_a_provider() {
    for mask in [
        AttentionMask::Causal,
        AttentionMask::Sliding { window_left: 11 },
        AttentionMask::Unmasked,
    ] {
        let mut graph = Graph::default();
        let spec = AttentionSpec { mask, ..spec() };
        let inputs = inputs(&mut graph, spec);
        let output = attention(inputs, spec).unwrap();
        assert_eq!(output.dims(), [6, 3, 32].map(Expression::from));
        graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let space = graph.search_space().unwrap();
        assert!(space.custom_ops.iter().all(|op| !op.is_lowered()));
        assert!(
            graph
                .egraph()
                .unwrap()
                .enodes
                .values()
                .any(|(label, _)| label == "persistent-state-attention-op")
        );
    }
}

#[test]
fn storage_layout_changes_only_the_view_facts_and_does_not_insert_a_copy() {
    let mut graph = Graph::default();
    let original = inputs(&mut graph, spec());
    let first = attention(original, spec()).unwrap();
    let KvView::Paged(mut view) = original.kv;
    view.layout = PagedKvLayout::HeadMajor;
    let second = attention(
        AttentionInputs {
            kv: KvView::Paged(view),
            ..original
        },
        spec(),
    )
    .unwrap();
    assert_eq!(graph.get_sources(first.id), graph.get_sources(second.id));
    let facts = graph
        .custom_ops
        .iter()
        .map(|op| op.compiler_facts(0))
        .collect::<Vec<_>>();
    assert_eq!(facts[0].lines().next(), facts[1].lines().next());
    assert_ne!(facts[0], facts[1]);
}

#[test]
fn invalid_geometry_and_scale_fail_before_inserting_an_operation() {
    let mut graph = Graph::default();
    let inputs = inputs(&mut graph, spec());
    for scale in [0.0, -0.25, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            attention(inputs, AttentionSpec { scale, ..spec() }),
            Err(AttentionError::Geometry(_))
        ));
    }
    for (query_heads, kv_heads) in [(0, 2), (6, 0), (5, 2)] {
        assert!(matches!(
            attention(
                inputs,
                AttentionSpec {
                    query_heads,
                    kv_heads,
                    ..spec()
                }
            ),
            Err(AttentionError::Geometry(_))
        ));
    }
    assert!(graph.custom_ops.is_empty());
}

#[test]
fn malformed_metadata_and_partial_pages_are_rejected() {
    let mut graph = Graph::default();
    let valid = inputs(&mut graph, spec());
    let KvView::Paged(view) = valid.kv;
    for invalid in [
        PagedKvView {
            page_indptr: graph.tensor(2).as_dtype(DType::Int),
            ..view
        },
        PagedKvView {
            key: graph.tensor(4 * 8 * 2 * 48 - 1).as_dtype(DType::Bf16),
            ..view
        },
        PagedKvView {
            value: graph.tensor((5, 8, 2, 32)).as_dtype(DType::Bf16),
            ..view
        },
    ] {
        assert!(matches!(
            attention(
                AttentionInputs {
                    kv: KvView::Paged(invalid),
                    ..valid
                },
                spec()
            ),
            Err(AttentionError::Shape(_))
        ));
    }
    let invalid = PagedKvView {
        page_indices: graph.tensor(3),
        ..view
    };
    assert!(matches!(
        attention(
            AttentionInputs {
                kv: KvView::Paged(invalid),
                ..valid
            },
            spec()
        ),
        Err(AttentionError::DType(_))
    ));
    assert!(graph.custom_ops.is_empty());
}

#[test]
fn a_strided_query_is_not_reinterpreted_as_a_contiguous_pointer() {
    let mut graph = Graph::default();
    let valid = inputs(&mut graph, spec());
    let query = graph
        .tensor((6, 3, 48))
        .as_dtype(DType::Bf16)
        .transpose(0, 1);
    assert!(matches!(
        attention(AttentionInputs { query, ..valid }, spec()),
        Err(AttentionError::Layout(_))
    ));
    assert!(graph.custom_ops.is_empty());
}

#[test]
fn tensors_from_another_graph_are_rejected() {
    let mut graph = Graph::default();
    let mut other = Graph::default();
    let valid = inputs(&mut graph, spec());
    let query_indptr = other.tensor(3).as_dtype(DType::Int);
    assert!(matches!(
        attention(
            AttentionInputs {
                query_indptr,
                ..valid
            },
            spec()
        ),
        Err(AttentionError::GraphOwnership)
    ));
    assert!(graph.custom_ops.is_empty());
}

#[test]
fn compiler_facts_preserve_small_positive_scales() {
    let mut graph = Graph::default();
    let spec = AttentionSpec {
        scale: 1e-20,
        ..spec()
    };
    let inputs = inputs(&mut graph, spec);
    attention(inputs, spec).unwrap();
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    assert!(
        graph
            .egraph()
            .unwrap()
            .enodes
            .values()
            .any(|(value, _)| { value.parse::<f64>().is_ok_and(|value| value == spec.scale) })
    );
}
