// =========================================================================
// Generic CUDA elementwise ops used inside FusionStart/FusionEnd regions.
//
// CUDA elementwise execution is represented as a FusionEnd-rooted region even
// for a single op. These ops are therefore region-internal only; standalone
// compilation is intentionally unsupported.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, OP_KIND, STRING},
        extract_dtype, extract_expr_list,
    },
    op::*,
    prelude::*,
};

use crate::kernel::KernelOp;

pub type Ops = (CudaUnaryElementwise, CudaBinaryElementwise);

type CompileOut = (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<Symbol, CudaSlice<u8>>,
);

fn extract_string_label(egraph: &SerializedEGraph, node: &ENodeId) -> String {
    egraph.enodes[node].0.trim_matches('"').to_string()
}

#[derive(Default, Debug, Clone)]
pub struct CudaUnaryElementwise {
    pub(crate) op: String,
    pub(crate) shape: Vec<Expression>,
    pub(crate) in_strides: Vec<Expression>,
    pub(crate) out_strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl EgglogOp for CudaUnaryElementwise {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaUnaryElementwise",
            &[
                ("op", STRING),
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        let mut rules = Vec::new();
        for (hlir, opcode) in [
            ("Sin", "Sin"),
            ("Sqrt", "Sqrt"),
            ("Exp2", "Exp2"),
            ("Log2", "Log2"),
            ("Recip", "Recip"),
        ] {
            // Each FusionStart is stamped with ITS input's dtype — it
            // describes how the external producer's buffer is loaded, and the
            // codegen reads every local at its producer's dtype. Stamping the
            // consumer's dtype instead would violate the construction
            // invariant whenever input and output dtypes differ.
            rules.push(Rule::raw(format!(
                include_str!("elementwise/binary_elementwise_rewrite.egg.in"),
                hlir = hlir,
                opcode = opcode,
            )));
        }

        // The rsqrt / direct-exp / direct-sigmoid substitutions are removed:
        // they replaced exact HLIR chains with approximate single
        // instructions (`rsqrtf`, `expf`, fused sigmoid) matched through
        // constant tolerance windows — V2 violations under the exact-HLIR
        // reproduction ruling. See the exact-HLIR reproduction ruling.

        // Give a flat Cast its own well-formed singleton region. This makes
        // adjacent elementwise work eligible for the safe forward-growth
        // rules without rewriting an existing FusionStart boundary.
        rules.push(Rule::raw(include_str!("elementwise/binary_transforms.egg")));

        rules
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                op: extract_string_label(egraph, kind_children[0]),
                shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for CudaUnaryElementwise {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("CudaUnaryElementwise must be compiled through fusion region codegen")
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CudaUnaryElementwise"
    }
}

#[derive(Default, Debug, Clone)]
pub struct CudaBinaryElementwise {
    pub(crate) op: String,
    pub(crate) out_shape: Vec<Expression>,
    pub(crate) a_stride: Vec<Expression>,
    pub(crate) b_stride: Vec<Expression>,
    pub(crate) out_stride: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl EgglogOp for CudaBinaryElementwise {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaBinaryElementwise",
            &[
                ("op", STRING),
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                // FusionStart dtypes follow each input's producer; see the
                // unary rules for why anything else is construction-illegal.
                include_str!("elementwise/binary_producers.egg"),
            ),
            Rule::raw(
                // The op and FusionEnd carry the RESULT dtype (?bin, not ?a —
                // Mul(bf16, f32) stamped bf16 would round the product); the
                // FusionStarts carry their producers' dtypes as above.
                include_str!("elementwise/reversed_binary_producers.egg"),
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        // Preserve every extracted metadata list verbatim. Singleton fusion
        // rules derive these lists from one matched HLIR operation, so their
        // rank/layout contract is established by construction; extraction
        // must not mutate that metadata.
        let out_shape =
            extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap();
        let a_stride = extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap();
        let b_stride = extract_expr_list(egraph, kind_children[3], list_cache, expr_cache).unwrap();
        let out_stride =
            extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                op: extract_string_label(egraph, kind_children[0]),
                out_shape,
                a_stride,
                b_stride,
                out_stride,
                dtype: extract_dtype(egraph, kind_children[5]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for CudaBinaryElementwise {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("CudaBinaryElementwise must be compiled through fusion region codegen")
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes() * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CudaBinaryElementwise"
    }
}
