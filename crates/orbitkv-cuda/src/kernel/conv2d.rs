//! CUDA conv2d-with-bias backend rewrite.
//!
//! `KernelConv2D` is selected by egglog from pure HLIR conv graphs and lowers
//! to a one-thread-per-output CUDA kernel. It avoids materializing unfold/im2col
//! intermediates while keeping model code free of custom ops.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use orbitkv_compiler::prelude::{FxHashMap, Symbol};
use orbitkv_compiler::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::{EgglogOp, LLIROp},
    prelude::FxHashSet,
    shape::{Expression, flatten_strides},
};

use crate::compile_module_image_for_current_device;
use crate::kernel::{KernelOp, hlir::generate_dyn_dims_defines};

#[derive(Default, Debug, Clone)]
pub struct KernelConv2D {
    out_shape: Vec<Expression>,
    input_shape: Vec<Expression>,
    input_stride: Vec<Expression>,
    weight_co_stride: Expression,
    weight_inner_stride: Expression,
    bias_c_stride: Expression,
    out_stride: Vec<Expression>,
    kernel_h: Expression,
    kernel_w: Expression,
    stride_h: Expression,
    stride_w: Expression,
    dilation_h: Expression,
    dilation_w: Expression,
    pad_h: Expression,
    pad_w: Expression,
    dtype: DType,
}

impl EgglogOp for KernelConv2D {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelConv2D",
            &[
                ("out_shape", ELIST),
                ("input_shape", ELIST),
                ("input_stride", ELIST),
                ("weight_co_stride", EXPRESSION),
                ("weight_inner_stride", EXPRESSION),
                ("bias_c_stride", EXPRESSION),
                ("out_stride", ELIST),
                ("kernel_h", EXPRESSION),
                ("kernel_w", EXPRESSION),
                ("stride_h", EXPRESSION),
                ("stride_w", EXPRESSION),
                ("dilation_h", EXPRESSION),
                ("dilation_w", EXPRESSION),
                ("pad_h", EXPRESSION),
                ("pad_w", EXPRESSION),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("conv2d/conv2d_rewrite.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a orbitkv_compiler::egglog_utils::SerializedEGraph,
        kind_children: &[&'a orbitkv_compiler::egglog_utils::NodeId],
        input_enodes: Vec<&'a orbitkv_compiler::egglog_utils::NodeId>,
        list_cache: &mut FxHashMap<&'a orbitkv_compiler::egglog_utils::NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a orbitkv_compiler::egglog_utils::NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a orbitkv_compiler::egglog_utils::NodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                input_shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                input_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                weight_co_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                weight_inner_stride: extract_expr(egraph, kind_children[4], expr_cache).unwrap(),
                bias_c_stride: extract_expr(egraph, kind_children[5], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[6], list_cache, expr_cache)
                    .unwrap(),
                kernel_h: extract_expr(egraph, kind_children[7], expr_cache).unwrap(),
                kernel_w: extract_expr(egraph, kind_children[8], expr_cache).unwrap(),
                stride_h: extract_expr(egraph, kind_children[9], expr_cache).unwrap(),
                stride_w: extract_expr(egraph, kind_children[10], expr_cache).unwrap(),
                dilation_h: extract_expr(egraph, kind_children[11], expr_cache).unwrap(),
                dilation_w: extract_expr(egraph, kind_children[12], expr_cache).unwrap(),
                pad_h: extract_expr(egraph, kind_children[13], expr_cache).unwrap(),
                pad_w: extract_expr(egraph, kind_children[14], expr_cache).unwrap(),
                dtype: extract_dtype(egraph, kind_children[15]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelConv2D {
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
        assert_eq!(self.dtype, DType::F32, "KernelConv2D currently emits F32");

        let vars: FxHashSet<Symbol> = self
            .out_shape
            .iter()
            .chain(&self.input_shape)
            .chain(&self.input_stride)
            .chain(&self.out_stride)
            .flat_map(|e| e.dyn_vars())
            .chain(self.weight_co_stride.dyn_vars())
            .chain(self.weight_inner_stride.dyn_vars())
            .chain(self.bias_c_stride.dyn_vars())
            .chain(self.kernel_h.dyn_vars())
            .chain(self.kernel_w.dyn_vars())
            .chain(self.stride_h.dyn_vars())
            .chain(self.stride_w.dyn_vars())
            .chain(self.dilation_h.dyn_vars())
            .chain(self.dilation_w.dyn_vars())
            .chain(self.pad_h.dyn_vars())
            .chain(self.pad_w.dyn_vars())
            .collect();

        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let c_out = self.out_shape[0].to_kernel();
        let h_out = self.out_shape[1].to_kernel();
        let w_out = self.out_shape[2].to_kernel();
        let c_in = self.input_shape[0].to_kernel();
        let h_in = self.input_shape[1].to_kernel();
        let w_in = self.input_shape[2].to_kernel();
        let weight_co_stride = self
            .weight_co_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let weight_inner_stride = self
            .weight_inner_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let bias_c_stride = self
            .bias_c_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let kh = self.kernel_h.to_kernel();
        let kw = self.kernel_w.to_kernel();
        let stride_h = self.stride_h.to_kernel();
        let stride_w = self.stride_w.to_kernel();
        let dilation_h = self.dilation_h.to_kernel();
        let dilation_w = self.dilation_w.to_kernel();
        let pad_h = self.pad_h.to_kernel();
        let pad_w = self.pad_w.to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let input_idx = flatten_strides(&self.input_shape, &self.input_stride)
            .to_kernel_with_index("input_linear");
        let n_outputs: Expression = self.out_shape.iter().copied().product();

        let kernel = format!(
            include_str!("conv2d/conv2d.cu.in"),
            total = n_outputs.to_kernel(),
            bias_c_stride = bias_c_stride,
            c_in = c_in,
            c_out = c_out,
            dilation_h = dilation_h,
            dilation_w = dilation_w,
            dyn_defines = dyn_defines,
            dyn_dims_param = dyn_dims_param,
            h_in = h_in,
            h_out = h_out,
            input_idx = input_idx,
            kh = kh,
            kw = kw,
            out_idx = out_idx,
            pad_h = pad_h,
            pad_w = pad_w,
            stride_h = stride_h,
            stride_w = stride_w,
            w_in = w_in,
            w_out = w_out,
            weight_co_stride = weight_co_stride,
            weight_inner_stride = weight_inner_stride,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("generic_conv2d_bias").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs.ceil_div(256), 1.into(), 1.into()),
            (n_outputs.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        for expression in self
            .out_shape
            .iter()
            .chain(&self.input_shape)
            .chain(&self.input_stride)
            .chain(&self.out_stride)
            .chain(std::iter::once(&self.weight_co_stride))
            .chain(std::iter::once(&self.weight_inner_stride))
            .chain(std::iter::once(&self.bias_c_stride))
            .chain(std::iter::once(&self.kernel_h))
            .chain(std::iter::once(&self.kernel_w))
            .chain(std::iter::once(&self.stride_h))
            .chain(std::iter::once(&self.stride_w))
            .chain(std::iter::once(&self.dilation_h))
            .chain(std::iter::once(&self.dilation_w))
            .chain(std::iter::once(&self.pad_h))
            .chain(std::iter::once(&self.pad_w))
        {
            expression.collect_dyn_vars_into(vars);
        }
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        let c_in = self.input_shape[0];
        self.output_size() * self.kernel_h * self.kernel_w * c_in * 2 * 4 + self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        let c_in = self.input_shape[0];
        self.output_size() * self.kernel_h * self.kernel_w * c_in * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GenericConv2D"
    }
}
