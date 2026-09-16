use super::*;
use crate::kernel::fusion::elementwise::{CudaBinaryElementwise, CudaUnaryElementwise};
use orbitkv_compiler::op::LLIROp;
use orbitkv_compiler::prelude::{DType, petgraph::algo::toposort};

/// Helper: wrap a `KernelOp` in an `LLIROp` of the kernel dialect.
fn llir_of(op: impl KernelOp + 'static) -> LLIROp {
    LLIROp::new::<dyn KernelOp>(Box::new(op) as Box<dyn KernelOp>)
}

/// Build the test region used by the canonicalization tests:
///
///   P_sqrt → FS_a (f32) ──┐
///   P_sin  → FS_b (bf16) ─┤→ Mul → Add → FE (f32, shape [8])
///   P_exp  → FS_c (f32) ──┘         ↑
///            (FS_c feeds Add's second operand)
///
/// `order` permutes node insertion and `flip_edges` reverses the
/// operand-edge insertion order, so the two graphs differ in every
/// NodeIndex and edge id while being structurally identical.
fn build_test_region(reversed: bool) -> (LLIRGraph, Vec<NodeIndex>) {
    let shape: Vec<Expression> = vec![8.into()];
    let z: Vec<Expression> = vec![Expression::from('z')];
    let fs = |dt: DType| FusionStart {
        shape: shape.clone(),
        strides: z.clone(),
        dtype: dt,
    };
    let bin = |op: &str, dt: DType| CudaBinaryElementwise {
        op: op.to_string(),
        out_shape: shape.clone(),
        a_stride: z.clone(),
        b_stride: z.clone(),
        out_stride: z.clone(),
        dtype: dt,
    };
    let unary = |op: &str| CudaUnaryElementwise {
        op: op.to_string(),
        shape: shape.clone(),
        in_strides: z.clone(),
        out_strides: z.clone(),
        dtype: DType::F32,
    };
    let fe = FusionEnd {
        shape: shape.clone(),
        strides: z.clone(),
        dtype: DType::F32,
    };

    let mut g: LLIRGraph = LLIRGraph::default();
    let add_nodes = |g: &mut LLIRGraph| {
        let p_sqrt = g.add_node(llir_of(unary("Sqrt")));
        let p_sin = g.add_node(llir_of(unary("Sin")));
        let p_exp = g.add_node(llir_of(unary("Exp")));
        let fs_a = g.add_node(llir_of(fs(DType::F32)));
        let fs_b = g.add_node(llir_of(fs(DType::Bf16)));
        let fs_c = g.add_node(llir_of(fs(DType::F32)));
        let mul = g.add_node(llir_of(bin("Mul", DType::F32)));
        let add = g.add_node(llir_of(bin("Add", DType::F32)));
        let fe_n = g.add_node(llir_of(fe.clone()));
        vec![p_sqrt, p_sin, p_exp, fs_a, fs_b, fs_c, mul, add, fe_n]
    };
    // Insert nodes in reverse for the permuted graph so every
    // NodeIndex differs. (StableGraph indices follow insertion order.)
    let nodes = if reversed {
        let p_exp = g.add_node(llir_of(unary("Exp")));
        let fe_n = g.add_node(llir_of(fe.clone()));
        let add = g.add_node(llir_of(bin("Add", DType::F32)));
        let fs_c = g.add_node(llir_of(fs(DType::F32)));
        let mul = g.add_node(llir_of(bin("Mul", DType::F32)));
        let fs_b = g.add_node(llir_of(fs(DType::Bf16)));
        let fs_a = g.add_node(llir_of(fs(DType::F32)));
        let p_sin = g.add_node(llir_of(unary("Sin")));
        let p_sqrt = g.add_node(llir_of(unary("Sqrt")));
        vec![p_sqrt, p_sin, p_exp, fs_a, fs_b, fs_c, mul, add, fe_n]
    } else {
        add_nodes(&mut g)
    };
    let [p_sqrt, p_sin, p_exp, fs_a, fs_b, fs_c, mul, add, fe_n]: [NodeIndex; 9] =
        nodes.clone().try_into().unwrap();

    let mut edges: Vec<(NodeIndex, NodeIndex)> = vec![
        (p_sqrt, fs_a),
        (p_sin, fs_b),
        (p_exp, fs_c),
        (fs_a, mul),
        (fs_b, mul),
        (mul, add),
        (fs_c, add),
        (add, fe_n),
    ];
    if reversed {
        edges.reverse();
    }
    for (a, b) in edges {
        g.add_edge(a, b, ());
    }
    (g, nodes)
}

fn region_source_and_producers(g: &LLIRGraph) -> (String, Vec<String>) {
    let topo = toposort(g, None).unwrap();
    let (units, _) = build_compile_units(&topo, g);
    let region = units
        .iter()
        .find_map(|u| match u {
            CompileUnit::Region(r) => Some(r),
            _ => None,
        })
        .expect("no region built");
    let (kernel, _) = region_kernel_source(region, g);
    // Producer identity per input slot, via the producer's unary op
    // name (Sqrt / Sin / Exp).
    let producers = region
        .external_inputs
        .iter()
        .map(|&p| {
            (***g[p].to_dialect::<dyn KernelOp>().unwrap())
                .downcast_ref::<CudaUnaryElementwise>()
                .unwrap()
                .op
                .clone()
        })
        .collect();
    (kernel, producers)
}

#[test]
fn prepared_fusion_plan_preserves_compile_units_and_sources() {
    let (graph, _) = build_test_region(false);
    let topo = toposort(&graph, None).unwrap();
    let (expected_units, absorbed) = build_compile_units(&topo, &graph);
    let all_nodes = topo.iter().copied().collect();
    let mut prepared = PreparedFusionPlan::discover(&topo, &graph);
    let mut source_cache = RegionSourceCache::default();
    prepared.prepare_region_kernels_for(&all_nodes, &graph, &mut source_cache, &[]);

    assert_eq!(prepared.compile_units(), expected_units);
    assert_eq!(prepared.absorbed_markers(), &absorbed);
    assert_eq!(
        prepared
            .compile_units_for(&all_nodes)
            .cloned()
            .collect::<Vec<_>>(),
        expected_units,
    );
    for unit in prepared.compile_units() {
        let CompileUnit::Region(region) = unit else {
            continue;
        };
        let (source, output_size) = region_kernel_source(region, &graph);
        let prepared_kernel = prepared.region_kernel(region.fe_node).unwrap();
        assert_eq!(prepared_kernel.source.as_ref(), source);
        assert_eq!(prepared_kernel.output_size, output_size);
    }
}

#[test]
fn singleton_region_deduplicates_repeated_fusion_start_inputs() {
    let shape = vec![8.into()];
    let strides = vec![Expression::from('z')];
    let mut graph = LLIRGraph::default();
    let producer = graph.add_node(llir_of(CudaUnaryElementwise {
        op: "Sqrt".to_string(),
        shape: shape.clone(),
        in_strides: strides.clone(),
        out_strides: strides.clone(),
        dtype: DType::F32,
    }));
    let start = graph.add_node(llir_of(FusionStart {
        shape: shape.clone(),
        strides: strides.clone(),
        dtype: DType::F32,
    }));
    let add = graph.add_node(llir_of(CudaBinaryElementwise {
        op: "Add".to_string(),
        out_shape: shape.clone(),
        a_stride: strides.clone(),
        b_stride: strides.clone(),
        out_stride: strides.clone(),
        dtype: DType::F32,
    }));
    let end = graph.add_node(llir_of(FusionEnd {
        shape,
        strides,
        dtype: DType::F32,
    }));
    graph.add_edge(producer, start, ());
    graph.add_edge(start, add, ());
    graph.add_edge(start, add, ());
    graph.add_edge(add, end, ());

    let topo = toposort(&graph, None).unwrap();
    let (units, _) = build_compile_units(&topo, &graph);
    let region = units
        .iter()
        .find_map(|unit| match unit {
            CompileUnit::Region(region) => Some(region),
            CompileUnit::Single(_) => None,
        })
        .unwrap();
    assert_eq!(region.fs_nodes, vec![start]);
    assert_eq!(region.external_inputs, vec![producer]);

    let (source, _) = region_kernel_source(region, &graph);
    assert!(source.contains("const float *in0"));
    assert!(!source.contains("in1"));
    assert!(source.contains("__fadd_rn(v_0, v_0)"));
}

#[test]
fn singleton_region_orders_inputs_by_structure() {
    let shape = vec![8.into()];
    let mut graph = LLIRGraph::default();
    let indexed = graph.add_node(llir_of(FusionStart {
        shape: shape.clone(),
        strides: vec![Expression::from('z')],
        dtype: DType::F32,
    }));
    let broadcast = graph.add_node(llir_of(FusionStart {
        shape,
        strides: vec![0.into()],
        dtype: DType::F32,
    }));

    assert_eq!(
        compare_fusion_starts(&graph, broadcast, indexed),
        std::cmp::Ordering::Less,
    );
    assert_eq!(
        compare_fusion_starts(&graph, indexed, broadcast),
        std::cmp::Ordering::Greater,
    );
}

#[test]
fn singleton_source_cache_separates_dynamic_dimension_abis() {
    let shape = vec![Expression::from('b')];
    let strides = vec![Expression::from('z')];
    let mut graph = LLIRGraph::default();
    let producer = graph.add_node(llir_of(CudaUnaryElementwise {
        op: "Sqrt".to_string(),
        shape: shape.clone(),
        in_strides: strides.clone(),
        out_strides: strides.clone(),
        dtype: DType::F32,
    }));
    let start = graph.add_node(llir_of(FusionStart {
        shape: shape.clone(),
        strides: strides.clone(),
        dtype: DType::F32,
    }));
    let unary = graph.add_node(llir_of(CudaUnaryElementwise {
        op: "Sin".to_string(),
        shape: shape.clone(),
        in_strides: strides.clone(),
        out_strides: strides.clone(),
        dtype: DType::F32,
    }));
    let end = graph.add_node(llir_of(FusionEnd {
        shape,
        strides,
        dtype: DType::F32,
    }));
    graph.add_edge(producer, start, ());
    graph.add_edge(start, unary, ());
    graph.add_edge(unary, end, ());

    let topo = toposort(&graph, None).unwrap();
    let all_nodes = topo.iter().copied().collect();
    let mut cache = RegionSourceCache::default();
    let ab = [Symbol::from('a'), Symbol::from('b')];
    let ba = [Symbol::from('b'), Symbol::from('a')];
    let prepare = |dims: &[Symbol], cache: &mut RegionSourceCache| {
        crate::kernel::hlir::set_global_dyn_dims(dims.to_vec());
        let mut plan = PreparedFusionPlan::discover(&topo, &graph);
        plan.prepare_region_kernels_for(&all_nodes, &graph, cache, dims);
        plan.region_kernel(end).unwrap().source.clone()
    };
    let source_ab = prepare(&ab, &mut cache);
    let source_ba = prepare(&ba, &mut cache);
    let source_ab_again = prepare(&ab, &mut cache);
    crate::kernel::hlir::clear_global_dyn_dims();

    assert!(source_ab.contains("dyn_dims[1]"));
    assert!(source_ba.contains("dyn_dims[0]"));
    assert_eq!(source_ab, source_ab_again);
    assert_eq!(cache.counters(), (1, 2));
}

/// Structurally identical regions must emit byte-identical kernel
/// sources (the compile-cache key) and bind the same producers to the
/// same input slots, regardless of NodeIndex / edge-id churn.
#[test]
fn region_kernel_source_is_nodeindex_invariant() {
    let (g1, _) = build_test_region(false);
    let (g2, _) = build_test_region(true);
    let (k1, p1) = region_source_and_producers(&g1);
    let (k2, p2) = region_source_and_producers(&g2);
    assert_eq!(k1, k2, "kernel source must not depend on NodeIndexes");
    assert_eq!(p1, p2, "input-slot → producer binding must match");
    assert!(
        k1.contains("__fmul_rn("),
        "F32 fusion must preserve the standalone Mul rounding boundary"
    );
    assert!(
        k1.contains("__fadd_rn("),
        "F32 fusion must preserve the standalone Add rounding boundary"
    );
}

#[test]
#[ignore = "requires CUDA; validates exact F32 fusion semantics"]
fn f32_mul_add_region_is_bit_exact_to_two_kernel_path() {
    use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
    use orbitkv_compiler::hlir::Input;

    const N: usize = 256;
    let shape = vec![Expression::from(N)];
    let strides = vec![Expression::from('z')];
    let mut graph = LLIRGraph::default();
    let inputs = [
        graph.add_node(LLIROp::new::<Input>(Box::default())),
        graph.add_node(LLIROp::new::<Input>(Box::default())),
        graph.add_node(LLIROp::new::<Input>(Box::default())),
    ];
    let starts = inputs.map(|input| {
        let start = graph.add_node(llir_of(FusionStart {
            shape: shape.clone(),
            strides: strides.clone(),
            dtype: DType::F32,
        }));
        graph.add_edge(input, start, ());
        start
    });
    let binary = |op: &str| CudaBinaryElementwise {
        op: op.to_string(),
        out_shape: shape.clone(),
        a_stride: strides.clone(),
        b_stride: strides.clone(),
        out_stride: strides.clone(),
        dtype: DType::F32,
    };
    let mul = graph.add_node(llir_of(binary("Mul")));
    let add = graph.add_node(llir_of(binary("Add")));
    let end = graph.add_node(llir_of(FusionEnd {
        shape,
        strides,
        dtype: DType::F32,
    }));
    graph.add_edge(starts[0], mul, ());
    graph.add_edge(starts[1], mul, ());
    graph.add_edge(mul, add, ());
    graph.add_edge(starts[2], add, ());
    graph.add_edge(add, end, ());

    let topo = toposort(&graph, None).unwrap();
    let (units, _) = build_compile_units(&topo, &graph);
    let region = units
        .iter()
        .find_map(|unit| match unit {
            CompileUnit::Region(region) => Some(region),
            CompileUnit::Single(_) => None,
        })
        .expect("Mul -> Add region must be fused");
    let (fused_source, _) = region_kernel_source(region, &graph);
    assert!(fused_source.contains("__fmul_rn("));
    assert!(fused_source.contains("__fadd_rn("));

    let context = CudaContext::new(0).expect("CUDA device is required");
    let stream = context.new_stream().unwrap();
    let compile = |source: &str, name: &str| {
        let image = compile_module_image_for_current_device(stream.context(), source).unwrap();
        let module = stream.context().load_module(image).unwrap();
        module.load_function(name).unwrap()
    };
    let fused = compile(&fused_source, "fused_region_k");
    let mul_kernel = compile(
        r#"
extern "C" __global__ void mul_k(float *out, const float *a, const float *b) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 256) out[i] = __fmul_rn(a[i], b[i]);
}
"#,
        "mul_k",
    );
    let add_kernel = compile(
        r#"
extern "C" __global__ void add_k(float *out, const float *product, const float *c) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 256) out[i] = __fadd_rn(product[i], c[i]);
}
"#,
        "add_k",
    );
    let fma_kernel = compile(
        r#"
extern "C" __global__ void fma_k(float *out, const float *a, const float *b, const float *c) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 256) out[i] = __fmaf_rn(a[i], b[i], c[i]);
}
"#,
        "fma_k",
    );

    // The first lane is a known contraction discriminator:
    // round((1 + 2^-23) * (1 - 2^-23)) - 1 == +0, while FMA yields -2^-46.
    let mut host = [vec![0.0_f32; N], vec![0.0_f32; N], vec![0.0_f32; N]];
    host[0][0] = f32::from_bits(1.0_f32.to_bits() + 1);
    host[1][0] = f32::from_bits(1.0_f32.to_bits() - 1);
    host[2][0] = -1.0;
    let [left, right, offset] = &mut host;
    for (i, ((left, right), offset)) in left.iter_mut().zip(right).zip(offset).enumerate().skip(1) {
        let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
        *left = sign * (1.0 + (i % 31) as f32 / 32.0);
        *right = (i % 17) as f32 / 16.0 - 0.5;
        *offset = sign * (i % 13) as f32 / 64.0;
    }
    let device_inputs = host
        .iter()
        .map(|values| stream.clone_htod(values).unwrap())
        .collect::<Vec<_>>();
    let input_ptrs = region
        .external_inputs
        .iter()
        .map(|producer| {
            let slot = inputs.iter().position(|input| input == producer).unwrap();
            device_inputs[slot].device_ptr(&stream).0
        })
        .collect::<Vec<_>>();
    let fused_output = stream.alloc_zeros::<f32>(N).unwrap();
    let product = stream.alloc_zeros::<f32>(N).unwrap();
    let split_output = stream.alloc_zeros::<f32>(N).unwrap();
    let fma_output = stream.alloc_zeros::<f32>(N).unwrap();
    let cfg = LaunchConfig::for_num_elems(N as u32);
    let fused_output_ptr = fused_output.device_ptr(&stream).0;
    let product_ptr = product.device_ptr(&stream).0;
    let split_output_ptr = split_output.device_ptr(&stream).0;
    let fma_output_ptr = fma_output.device_ptr(&stream).0;
    let a_ptr = device_inputs[0].device_ptr(&stream).0;
    let b_ptr = device_inputs[1].device_ptr(&stream).0;
    let c_ptr = device_inputs[2].device_ptr(&stream).0;

    unsafe {
        let mut launch = stream.launch_builder(&fused);
        launch.arg(&fused_output_ptr);
        for pointer in &input_ptrs {
            launch.arg(pointer);
        }
        launch.launch(cfg).unwrap();

        stream
            .launch_builder(&mul_kernel)
            .arg(&product_ptr)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .launch(cfg)
            .unwrap();
        stream
            .launch_builder(&add_kernel)
            .arg(&split_output_ptr)
            .arg(&product_ptr)
            .arg(&c_ptr)
            .launch(cfg)
            .unwrap();
        stream
            .launch_builder(&fma_kernel)
            .arg(&fma_output_ptr)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .launch(cfg)
            .unwrap();
    }

    let fused_bits = stream
        .clone_dtoh(&fused_output)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let split_bits = stream
        .clone_dtoh(&split_output)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let fma_bits = stream
        .clone_dtoh(&fma_output)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(fused_bits, split_bits);
    assert_ne!(
        fma_bits[0], split_bits[0],
        "fixture must detect contraction"
    );
}
