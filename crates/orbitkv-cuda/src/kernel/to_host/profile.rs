//! Diagnostics for completed graph launches. Measurement uses the existing
//! opt-in timing nodes; reporting does not insert events or synchronize CUDA.

use std::collections::BTreeMap;

use itertools::Itertools;
use orbitkv_compiler::prelude::DynMap;

use super::{CompiledStep, CudaGraphOp};
use crate::kernel::event_elapsed_ms;

struct StepMeasurement {
    operator: String,
    implementation: Option<String>,
    duration_ms: f64,
}

impl CudaGraphOp {
    /// Read a separately instrumented direct execution after stream completion.
    /// Each library island is one region; fused interiors retain their LLIR IDs.
    pub(crate) fn measured_regions(
        &self,
    ) -> anyhow::Result<Vec<orbitkv_compiler::search::ProfiledRegion>> {
        let state = self.state.borrow();
        anyhow::ensure!(
            state.timing_events.len() > state.steps.len(),
            "missing step timing events"
        );
        state
            .steps
            .iter()
            .enumerate()
            .map(|(index, step)| {
                let elapsed = event_elapsed_ms(
                    self.stream.context(),
                    state.timing_events[index],
                    state.timing_events[index + 1],
                )?;
                let nodes = match *step {
                    CompiledStep::Kernel(i) => state.kernels[i].source_nodes.clone(),
                    CompiledStep::CuBlasLt(i) => vec![state.cublaslt_ops[i].node],
                    CompiledStep::FlashInferDecode(i) => vec![state.flashinfer_ops[i].node],
                    CompiledStep::CapturedHost(i) => vec![state.captured_host_ops[i].node],
                };
                Ok(orbitkv_compiler::search::ProfiledRegion {
                    nodes,
                    cost: f64::from(elapsed) / 1_000.0,
                })
            })
            .collect()
    }

    pub(crate) fn print_step_profile(&self, dyn_map: &DynMap, graph_node: usize) {
        if std::env::var_os("ORBITKV_CUDA_PROFILE_GRAPH_STEPS").is_none() {
            return;
        }
        let detailed = std::env::var_os("ORBITKV_CUDA_PROFILE_GRAPH_STEP_DETAILS").is_some();
        let state = self.state.borrow();
        let profile = tracing::info_span!(target: "orbitkv::stage", "cuda.graph.profile",
            graph_node, detailed, step_count = state.steps.len(),
            total_device_ms = tracing::field::Empty, status = "unavailable");
        let _entered = profile.enter();
        if state.timing_events.len() < state.steps.len() + 1 {
            return;
        }
        let mut measurements = Vec::with_capacity(state.steps.len());
        for (index, step) in state.steps.iter().enumerate() {
            let Ok(elapsed) = event_elapsed_ms(
                self.stream.context(),
                state.timing_events[index],
                state.timing_events[index + 1],
            ) else {
                return;
            };
            let (operator, implementation) = match *step {
                CompiledStep::Kernel(idx) => (
                    state.kernels[idx].kernel_name.to_owned(),
                    detailed.then(|| format!("{:?}", state.kernels[idx].kernel_op)),
                ),
                CompiledStep::CuBlasLt(idx) => (
                    "CuBlasLt".to_owned(),
                    detailed.then(|| format!("{:?}", state.cublaslt_ops[idx].host_op)),
                ),
                CompiledStep::FlashInferDecode(idx) => (
                    "FlashInferAttention".to_owned(),
                    detailed.then(|| format!("{:?}", state.flashinfer_ops[idx].host_op)),
                ),
                CompiledStep::CapturedHost(idx) => (
                    state.captured_host_ops[idx]
                        .host_op
                        .stats_name()
                        .unwrap_or("CapturedHost")
                        .to_owned(),
                    detailed.then(|| format!("{:?}", state.captured_host_ops[idx].host_op)),
                ),
            };
            measurements.push(StepMeasurement {
                operator,
                implementation,
                duration_ms: f64::from(elapsed),
            });
        }
        // Emit only complete measurements, retaining order and full descriptors.
        // Rule/provider names are data; diagnostics never classify model layers.
        for (dimension, value) in dyn_map.iter().sorted_by_key(|(name, _)| name.to_string()) {
            tracing::event!(name: "cuda.graph.dimension", target: "orbitkv::stage", tracing::Level::INFO,
                dimension = %dimension, value = *value);
        }
        let mut totals = BTreeMap::<String, (usize, f64)>::new();
        let mut total_ms = 0.0;
        for (index, step) in measurements.iter().enumerate() {
            tracing::event!(name: "cuda.graph.step", target: "orbitkv::stage", tracing::Level::INFO,
                index, operator = step.operator.as_str(),
                implementation = step.implementation.as_deref().unwrap_or(""),
                duration_ms = step.duration_ms);
            let label = match &step.implementation {
                Some(detail) => format!("{}:{detail}", step.operator),
                None => step.operator.clone(),
            };
            let total = totals.entry(label).or_default();
            total.0 += 1;
            total.1 += step.duration_ms;
            total_ms += step.duration_ms;
        }
        profile.record("total_device_ms", total_ms);
        profile.record("status", "measured");
        eprintln!(
            "CUDA_GRAPH_STEP_PROFILE dyn={dyn_map:?} total_ms={total_ms:.3} {}",
            totals
                .into_iter()
                .map(|(name, (count, ms))| format!("{name}[{count}]={ms:.3}ms"))
                .join(" ")
        );
    }
}
