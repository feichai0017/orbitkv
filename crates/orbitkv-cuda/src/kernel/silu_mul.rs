//! Vectorized SiLU × up, adapted from SGLang's activation kernel.
//!
//! The rewrite preserves the frontend's F32 primitive chain, including its
//! BF16 activation materialization. It is deliberately distinct from both a
//! store-once mathematical SiLU and the all-BF16 fused-projection SwiGLU.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{DType, ENodeId, Expression, FxHashMap, SerializedEGraph, Symbol},
    shape::kernel_const_name,
};

use super::{KernelOp, hlir::generate_dyn_dims_defines};
use crate::compile_module_image_for_current_device;

// SGLang's pre-Blackwell vector width and activation launch geometry.
const VECTOR_ELEMENTS: usize = 16 / std::mem::size_of::<half::bf16>();
const THREADS: usize = 256;

#[derive(Debug, Clone, Default)]
pub struct KernelSiluMul {
    size: Expression,
}

impl EgglogOp for KernelSiluMul {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelSiluMul", &[("size", EXPRESSION)])
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        [
            ("symbolic rows", "(= ?span (MMul ?rows ?width_expr))"),
            (
                "static rows",
                "(= ?rows (MNum ?row_count))
                 (= ?span (MNum ?element_count))
                 (= ?element_count (* ?row_count ?width))",
            ),
            (
                "physical span",
                "(= ?row_lower (lower ?rows)) (> ?row_lower 0)
                 (= ?last_column (- ?width 1))
                 (= ?span (MAdd (MMax
                    (MAdd (MMul (MSub ?rows (MNum 1)) ?width_expr) (MNum ?last_column))
                    (MNum 0)) (MNum 1)))",
            ),
            (
                "positive physical span",
                "(= ?row_lower (lower ?rows)) (> ?row_lower 0)
                 (= ?span (MAdd (MMul (MSub ?rows (MNum 1)) ?width_expr) ?width_expr))",
            ),
        ]
        .into_iter()
        .map(|(variant, size_proof)| {
            Rule::raw(format!(
                include_str!("silu_mul/rewrite.egg.in"),
                vector_elements = VECTOR_ELEMENTS,
                size_proof = size_proof,
                variant = variant,
            ))
        })
        .collect()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                size: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
            })),
            input_enodes,
        )
    }
}

impl KernelSiluMul {
    fn source(&self) -> String {
        let vars = self.all_dyn_vars();
        let (dyn_defines, _) = generate_dyn_dims_defines(&vars);
        format!(
            include_str!("silu_mul/kernel.cu.in"),
            vector_elements = VECTOR_ELEMENTS,
            num_vectors = (self.size / VECTOR_ELEMENTS).to_kernel_with("const_z", &|dim| {
                format!("static_cast<long long>({})", kernel_const_name(dim))
            }),
            dyn_defines = dyn_defines,
            dyn_parameter = if vars.is_empty() {
                ""
            } else {
                ", const int* dyn_dims"
            },
        )
    }
}

impl KernelOp for KernelSiluMul {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let source = self.source();
        let (module, function) = if let Some((module, function)) = compile_cache.get(&source) {
            (Arc::clone(module), function.clone())
        } else {
            let image = compile_module_image_for_current_device(stream.context(), &source).unwrap();
            let module = stream.context().load_module(image).unwrap();
            let function = module.load_function("silu_mul").unwrap();
            compile_cache.insert(source.clone(), (Arc::clone(&module), function.clone()));
            (module, function)
        };
        (
            function,
            module,
            source,
            (
                self.size.ceil_div(VECTOR_ELEMENTS * THREADS),
                1.into(),
                1.into(),
            ),
            (THREADS.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.size
    }
    fn output_dtype(&self) -> DType {
        DType::Bf16
    }
    fn output_bytes(&self) -> Expression {
        self.size * std::mem::size_of::<half::bf16>()
    }
    fn bytes_loaded(&self) -> Expression {
        self.output_bytes() * 2
    }
    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }
    fn kernel_name(&self) -> &'static str {
        "SiluMul"
    }
}

#[cfg(test)]
#[path = "../../tests/unit/kernel/silu_mul/mod.rs"]
mod tests;
