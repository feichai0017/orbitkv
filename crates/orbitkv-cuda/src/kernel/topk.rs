//! Fused stable-argsort ranks: one kernel for the O(E²) pairwise rank
//! computation that `stable_argsort(descending)` spells as ~8 HLIR kernels
//! (two broadcast value views, three LessThans, the index tie-break chain,
//! and the comparison sum).
//!
//! `rank[s, j] = Σ_i (x[s,i] > x[s,j]) || (x[s,i] == x[s,j] && i < j)`
//!
//! The kernel reproduces the chain's exact count-based semantics (including
//! the index tie-break) and the rank→index scatter in one launch: element j
//! lands at sorted position rank(j) — bit-identical to the decomposed
//! spelling. The rewrite matches all the way through the scatter tail and
//! unions into the sorted-indices eclass emitted by
//! `stable_argsort(descending)` / `topk_indexes`.

use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device,
    kernel::{KernelOp, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, OP_KIND},
        extract_expr_list,
    },
    op::*,
    prelude::*,
};

#[derive(Default, Debug, Clone)]
pub struct KernelStableSortIdx {
    /// Output shape `(rows, E)`; rows may be dynamic.
    out_shape: Vec<Expression>,
}

impl EgglogOp for KernelStableSortIdx {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelStableSortIdx", &[("out_shape", ELIST)])
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Mirrors the exact emission of stable_argsort(axis=1, descending):
        //   a_val = x·1.0 viewed (rows, E_j, E_i→0)   strides (z·E, z, 0)
        //   b_val = x·1.0 viewed (rows, 0, E_i)       strides (z·E, 0, z)
        //   primary = b_val < a_val                   (descending: a > b)
        //   not_lt  = (a_val < b_val)·(−1) + 1
        //   not_gt  = (b_val < a_val)·(−1) + 1
        //   val_eq  = not_lt · not_gt
        //   idx_cmp = iota_i < iota_j (broadcast casts)
        //   cmp     = cast(primary) + val_eq·cast(idx_cmp)
        //   ranks   = Cast(Int)(Sum_axis1(cmp))
        //
        // Three relation-staged rules (pre → late → late): one monolithic
        // ~25-atom join explodes combinatorially on rolled bodies with
        // several distinct layer instances (every loosely-paired binary atom
        // multiplies the join by the instance count). Each stage walks a
        // chain from a strongly-pinned anchor; later stages key every atom
        // off relation-bound variables.
        vec![Rule::raw(include_str!("topk/topk_rewrite.egg"))]
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
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelStableSortIdx {
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
        let rows = self.out_shape[0];
        let e = self.out_shape[1].to_usize().expect("ranks E is static");
        assert!(e <= 1024, "stable ranks kernel supports E <= 1024");

        let vars: FxHashSet<Symbol> = rows.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            include_str!("topk/topk.cu.in"),
            dyn_defines = dyn_defines,
            dyn_dims_param = dyn_dims_param,
            e = e,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("stable_sort_idx_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = e.next_multiple_of(32).min(1024);
        (
            func,
            module,
            kernel,
            (rows, Expression::from(1usize), Expression::from(1usize)),
            (
                Expression::from(tpb),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.out_shape[0].collect_dyn_vars_into(vars);
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        let e = self.out_shape[1];
        self.out_shape[0] * e * e * 2
    }

    fn kernel_name(&self) -> &'static str {
        "StableSortIdx"
    }
}
