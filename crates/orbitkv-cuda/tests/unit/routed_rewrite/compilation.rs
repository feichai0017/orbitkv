//! Same-input saturation benchmark. Baseline rule text is supplied explicitly
//! from a recorded Git revision; production has no rule-override mechanism.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Instant};

use orbitkv_compiler::{
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef},
        hlir_to_egglog, run_egglog_with_report_late_passes_interval_analysis_and_log,
    },
    hlir::HLIROps,
    op::{EgglogOp, IntoEgglogOp, Runtime},
    prelude::Graph,
};
use sha2::{Digest, Sha256};

use super::{CudaRuntime, SEQ, build_gemma_moe_graph, build_qwen_moe_graph};

#[derive(Debug)]
struct BaselineRules {
    original: Arc<Box<dyn EgglogOp>>,
    source: String,
}

impl EgglogOp for BaselineRules {
    fn sort(&self) -> SortDef {
        self.original.sort()
    }
    fn cleanup(&self) -> bool {
        self.original.cleanup()
    }
    fn egglog_declarations(&self) -> Vec<String> {
        self.original.egglog_declarations()
    }
    fn ir_defs(&self) -> Vec<String> {
        self.original.ir_defs()
    }
    fn n_inputs(&self) -> usize {
        self.original.n_inputs()
    }
    fn rewrites(&self) -> Vec<Rule> {
        let mut rules = self.original.rewrites();
        assert_eq!(
            rules.len(),
            2,
            "baseline needs dimension declarations and one rule asset"
        );
        rules[1] = Rule::raw(&self.source);
        rules
    }
}

fn operation_counts(graph: &SerializedEGraph) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for (sort, nodes) in graph.eclasses.values() {
        if sort == "OpKind" {
            for node in nodes {
                *counts.entry(graph.enodes[node].0.clone()).or_default() += 1;
            }
        }
    }
    counts
}

fn dense_graph() -> Graph {
    const LAYERS: usize = 8;
    const WIDTH: usize = 32;
    let mut graph = Graph::default();
    let mut x = graph.tensor((SEQ, WIDTH));
    for _ in 0..LAYERS {
        let weight = graph.tensor((WIDTH, WIDTH));
        let residual = x;
        x = x.matmul(weight).silu() + residual;
    }
    x.output();
    graph
}

#[test]
#[ignore = "CPU saturation benchmark; optional ORBITKV_RULE_BASELINE_DIR supplies recorded rules"]
#[allow(clippy::arc_with_non_send_sync)] // Matches the compiler's single-threaded operation table ABI.
fn moe_rule_compilation_workloads() {
    let baseline = std::env::var_os("ORBITKV_RULE_BASELINE_DIR").map(PathBuf::from);
    let mut ops = <CudaRuntime as Runtime>::Ops::into_vec();
    if let Some(directory) = &baseline {
        for op in &mut ops {
            let filename = match op.sort().name.as_str() {
                "GLUMoE" => "glumoe_rewrite.egg",
                "FusedMoE" => "fused_moe_rewrite.egg",
                _ => continue,
            };
            *op = Arc::new(Box::new(BaselineRules {
                original: op.clone(),
                source: std::fs::read_to_string(directory.join(filename)).unwrap(),
            }));
        }
    }
    ops.extend(HLIROps::into_vec());
    let workloads = if let Some(path) = std::env::var_os("ORBITKV_RULE_FIXTURE") {
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        vec![(
            "decoder",
            fixture["program"].as_str().unwrap().to_owned(),
            fixture["root"].as_str().unwrap().to_owned(),
            fixture["extra"].as_str().unwrap().to_owned(),
            fixture["interval_analysis"].as_bool().unwrap(),
            false,
        )]
    } else {
        [
            ("normalized_swiglu", build_qwen_moe_graph().graph, true),
            ("gemma_gelu", build_gemma_moe_graph().graph, true),
            ("dense", dense_graph(), false),
        ]
        .into_iter()
        .map(|(name, mut graph, routed)| {
            graph.set_dim('s', SEQ);
            let (program, root) = hlir_to_egglog(&graph);
            (
                name,
                program,
                root,
                crate::target::CudaTarget { major: 9, minor: 0 }.compiler_facts(),
                false,
                routed,
            )
        })
        .collect()
    };
    for (workload, program, root, extra, interval_analysis, routed) in workloads {
        let program_sha256 = format!("{:x}", Sha256::digest(program.as_bytes()));
        let started = Instant::now();
        let (space, report) = run_egglog_with_report_late_passes_interval_analysis_and_log(
            &program,
            &root,
            &ops,
            false,
            &[],
            &extra,
            interval_analysis,
            false,
        )
        .unwrap();
        let elapsed = started.elapsed();
        let counts = operation_counts(&space);
        assert_eq!(
            counts.get("GLUMoE").copied().unwrap_or_default() > 0,
            routed
        );
        println!(
            "MOE_RULE_BENCH {}",
            serde_json::json!({
                "workload": workload,
                "baseline": baseline.is_some(),
                "program_sha256": program_sha256,
                "wall_seconds": elapsed.as_secs_f64(),
                "egglog_seconds": report.total_time.as_secs_f64(),
                "operation_counts": counts,
            })
        );
    }
}
