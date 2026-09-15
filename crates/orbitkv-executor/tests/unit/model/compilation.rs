//! Export a real normalized decoder graph for isolated rule benchmarks.

use super::*;
use orbitkv_compiler::{
    egglog_utils::{base::interval_facts_egglog, hlir_to_egglog},
    op::IntoEgglogOp,
    prelude::CompileOptions,
    shape::{DimInterval, DynDimIntervals},
};

pub(super) fn export_saturation_fixture(
    graph: &mut Graph,
    decoder: &DecoderGraph,
    plan: &ExecutorPlan,
    arenas: &[ExecutorArena],
) {
    let Some(path) = std::env::var_os("ORBITKV_RULE_FIXTURE_OUTPUT") else {
        return;
    };
    let compile = DecoderCompileConfig {
        output_rows: crate::model::DecoderOutputRows::AllTokens,
        maximum_query_tokens: 32,
        representative_prefill_tokens: 32,
        maximum_batch_size: 8,
        maximum_context_pages: 64,
        representative_context_pages: 8,
        search_graphs: 8,
        search_seed: 7,
    };
    let options: CompileOptions = super::super::tuning::decoder_compile_options(
        decoder,
        compile,
        &DecoderTuningProfile::default(),
        usize::try_from(plan.page_tokens).unwrap(),
    )
    .unwrap();
    graph.prepare_selected_schedule(&options);
    let intervals: DynDimIntervals = options
        .dim_buckets
        .iter()
        .map(|(&dim, buckets)| {
            let bucket = &buckets[0];
            (
                dim,
                DimInterval::new(
                    i64::try_from(bucket.min).unwrap(),
                    i64::try_from(bucket.max).unwrap(),
                ),
            )
        })
        .collect();
    let facts = interval_facts_egglog(&intervals, []);
    let (program, root) = hlir_to_egglog(graph);
    let mut extra = plan.compiler_facts(arenas).unwrap().egglog().to_owned();
    extra.push_str(&orbitkv_cuda::target::CudaTarget { major: 9, minor: 0 }.compiler_facts());
    let backend = <CudaRuntime as Runtime>::Ops::into_vec()
        .iter()
        .flat_map(|op| op.egglog_declarations())
        .collect::<BTreeSet<_>>();
    for declaration in graph
        .custom_ops
        .iter()
        .map(|op| op.compiler_declarations())
        .filter(|text| !text.is_empty() && !backend.contains(*text))
        .collect::<BTreeSet<_>>()
    {
        extra.push('\n');
        extra.push_str(declaration);
    }
    for (id, op) in graph.custom_ops.iter().enumerate() {
        extra.push('\n');
        extra.push_str(&op.compiler_facts(id));
    }
    std::fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({
            "program": format!("{facts}\n{program}"), "root": root,
            "extra": extra, "interval_analysis": !facts.is_empty(),
        }))
        .unwrap(),
    )
    .unwrap();
}
