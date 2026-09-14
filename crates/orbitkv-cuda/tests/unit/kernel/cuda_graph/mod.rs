use super::*;
use cudarc::driver::CudaContext;
use orbitkv_compiler::prelude::*;

use crate::runtime::CudaRuntime;
use crate::tests::utilities::*;

#[test]
fn test_create_empty_graph() {
    let Ok(ctx) = CudaContext::new(0) else { return };
    assert!(CudaGraphHandle::new(ctx).is_ok());
}

#[test]
fn test_kernel_params() {
    let mut params = KernelParams::new(0x1000, &[0x2000, 0x3000]);
    assert!(!params.as_cuda_params().is_null());
    params.update_output(0x4000);
    params.update_input(0, 0x5000);
}

#[test]
fn test_cuda_function_size() {
    assert_eq!(
        std::mem::size_of::<CudaFunction>(),
        std::mem::size_of::<CUfunction>() + std::mem::size_of::<usize>()
    );
}

#[test]
fn test_raw_function_extraction() {
    let Ok(ctx) = CudaContext::new(0) else { return };
    let kernel_src = r#"extern "C" __global__ void test_kernel(float* out) { out[0] = 1.0f; }"#;
    let Ok(ptx) = crate::compile_module_image_for_current_device(&ctx, kernel_src) else {
        return;
    };
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("test_kernel").unwrap();
    let cu_func = unsafe { func.raw_function() };
    assert!(!cu_func.is_null());
    let mut max_threads: i32 = 0;
    let result = unsafe {
        sys::cuFuncGetAttribute(
            &mut max_threads,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            cu_func,
        )
    };
    assert!(result == sys::cudaError_enum::CUDA_SUCCESS);
}

#[test]
fn test_graph_with_kernel() {
    use cudarc::driver::{CudaSlice, DevicePtr};
    let Ok(ctx) = CudaContext::new(0) else { return };
    let kernel_src = r#"extern "C" __global__ void test_kernel(float* out, float* in1) { if (threadIdx.x == 0) out[0] = in1[0] + 1.0f; }"#;
    let Ok(ptx) = crate::compile_module_image_for_current_device(&ctx, kernel_src) else {
        return;
    };
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("test_kernel").unwrap();
    let stream = ctx.default_stream();
    let output: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
    let mut input: CudaSlice<f32> = unsafe { stream.alloc(1) }.unwrap();
    stream.memcpy_htod(&[5.0f32], &mut input).unwrap();
    let cu_func = unsafe { func.raw_function() };
    let mut graph = CudaGraphHandle::new(ctx.clone()).unwrap();
    let mut params =
        KernelParams::new(output.device_ptr(&stream).0, &[input.device_ptr(&stream).0]);
    let _node = unsafe {
        graph.add_kernel_node(
            &[],
            cu_func,
            (1, 1, 1),
            (1, 1, 1),
            0,
            params.as_cuda_params(),
        )
    }
    .unwrap();
    let exec = graph.instantiate().unwrap();
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let mut result = [0.0f32];
    stream.memcpy_dtoh(&output, &mut result).unwrap();
    assert_eq!(result[0], 6.0f32);
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_copy_node_copies_after_child_graph() {
    use cudarc::driver::{CudaSlice, DevicePtr};

    let ctx = CudaContext::new(0).expect("CUDA device is required");
    let stream = ctx.new_stream().unwrap();
    let source: CudaSlice<u32> = stream.clone_htod(&[7_u32, 11]).unwrap();
    let destination: CudaSlice<u32> = stream.clone_htod(&[0_u32, 0]).unwrap();

    let mut child = CudaGraphHandle::new(ctx.clone()).unwrap();
    child.add_empty_node(&[]).unwrap();
    let mut parent = CudaGraphHandle::new(ctx).unwrap();
    let child_node = parent.add_child_graph_node(&[], &child).unwrap();
    parent
        .add_device_copy_node(
            &[child_node],
            source.device_ptr(&stream).0,
            destination.device_ptr(&stream).0,
            2 * size_of::<u32>(),
        )
        .unwrap();
    let executable = parent.instantiate().unwrap();
    executable.launch(&stream).unwrap();
    let result = stream.clone_dtoh(&destination).unwrap();
    assert_eq!(result, vec![7_u32, 11]);
}

#[test]
fn test_graph_empty_node_dependency_reconnect() {
    let Ok(ctx) = CudaContext::new(0) else { return };
    let mut graph = CudaGraphHandle::new(ctx).unwrap();

    let entry = graph.add_empty_node(&[]).unwrap();
    let middle = graph.add_empty_node(&[entry]).unwrap();
    let exit = graph.add_empty_node(&[middle]).unwrap();

    let nodes = graph.nodes().unwrap();
    assert!(nodes.contains(&entry));
    assert!(nodes.contains(&middle));
    assert!(nodes.contains(&exit));
    assert_eq!(graph.dependencies(middle).unwrap(), vec![entry]);
    assert_eq!(graph.dependent_nodes(middle).unwrap(), vec![exit]);

    graph.add_dependencies(&[entry], &[exit]).unwrap();
    let exit_deps = graph.dependencies(exit).unwrap();
    assert!(exit_deps.contains(&entry));
    assert!(exit_deps.contains(&middle));

    graph.remove_dependencies(&[middle], &[exit]).unwrap();
    let exit_deps = graph.dependencies(exit).unwrap();
    assert_eq!(exit_deps.len(), 1);
    assert!(exit_deps.contains(&entry));

    unsafe {
        graph.destroy_node(middle).unwrap();
    }
    assert!(!graph.nodes().unwrap().contains(&middle));
}

#[test]
fn device_copy_node_rejects_empty_or_null_ranges_before_cuda() {
    let Ok(ctx) = CudaContext::new(0) else { return };
    let mut graph = CudaGraphHandle::new(ctx).unwrap();
    assert!(graph.add_device_copy_node(&[], 0, 1, 4).is_err());
    assert!(graph.add_device_copy_node(&[], 1, 0, 4).is_err());
    assert!(graph.add_device_copy_node(&[], 1, 2, 0).is_err());
}

// CUDA Graph Tests

#[test]
fn test_cuda_graph_basic_execution() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let size = 1024;
    let mut cx = Graph::default();
    let a = cx.tensor(size).persist();
    let b = cx.tensor(size).persist();
    let c = ((a + b) * a + b).output();

    let mut rt = CudaRuntime::initialize(stream);
    let data_a = random_f32_vec(size, 42, -0.5, 0.5);
    let data_b = random_f32_vec(size, 43, -0.5, 0.5);
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let result1 = rt.get_f32(c);
    rt.execute(&cx.dyn_map);
    let eps = dtype_epsilon(orbitkv_compiler::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result1, &rt.get_f32(c), tol, tol);
    let expected: Vec<f32> = data_a
        .iter()
        .zip(&data_b)
        .map(|(a, b)| (a + b) * a + b)
        .collect();
    assert_close(&result1, &expected, tol, tol);
}

#[test]
fn test_cuda_graph_multiple_executions() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let size = 2048;
    let mut cx = Graph::default();
    let a = cx.tensor(size).persist();
    let b = cx.tensor(size).persist();
    let c = (a + b + a + b).output();

    let mut rt = CudaRuntime::initialize(stream);
    let data_a = random_f32_vec(size, 42, -0.5, 0.5);
    let data_b = random_f32_vec(size, 43, -0.5, 0.5);
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    let mut results = Vec::new();
    for _ in 0..5 {
        rt.execute(&cx.dyn_map);
        results.push(rt.get_f32(c));
    }
    let eps = dtype_epsilon(orbitkv_compiler::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    for result in &results {
        assert_close(result, &results[0], tol, tol);
    }
    let expected: Vec<f32> = data_a
        .iter()
        .zip(&data_b)
        .map(|(a, b)| a + b + a + b)
        .collect();
    assert_close(&results[0], &expected, tol, tol);
}

#[test]
fn test_cuda_graph_dyn_dims_surgical_update() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let size = 512;
    let mut cx = Graph::default();
    let a = cx.tensor('s');
    let b = cx.tensor('s');
    let c = (a + b).output();
    let d = (c * a).output();

    let mut rt = CudaRuntime::initialize(stream);
    let data_a = random_f32_vec(size, 42, -0.5, 0.5);
    let data_b = random_f32_vec(size, 43, -0.5, 0.5);
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    cx.set_dim('s', size);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let expected: Vec<f32> = data_a
        .iter()
        .zip(&data_b)
        .map(|(a, b)| (a + b) * a)
        .collect();
    let eps = dtype_epsilon(orbitkv_compiler::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    assert_close(&rt.get_f32(d), &expected, tol, tol);
    let size = 1024;
    let data_a2 = random_f32_vec(size, 44, -0.5, 0.5);
    let data_b2 = random_f32_vec(size, 45, -0.5, 0.5);
    rt.set_data(a, data_a2.clone());
    rt.set_data(b, data_b2.clone());
    cx.set_dim('s', size);
    rt.execute(&cx.dyn_map);
    let expected2: Vec<f32> = data_a2
        .iter()
        .zip(&data_b2)
        .map(|(a, b)| (a + b) * a)
        .collect();
    assert_close(&rt.get_f32(d), &expected2, tol, tol);
}

#[test]
fn test_single_kernel_in_graph() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let size = 1024;
    let mut cx = Graph::default();
    let a = cx.tensor(size);
    let b = cx.tensor(size);
    let c = (a + b).output();

    let mut rt = CudaRuntime::initialize(stream);
    let data_a = random_f32_vec(size, 42, -0.5, 0.5);
    let data_b = random_f32_vec(size, 43, -0.5, 0.5);
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);
    let expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(a, b)| a + b).collect();
    let eps = dtype_epsilon(orbitkv_compiler::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    assert_close(&rt.get_f32(c), &expected, tol, tol);
    assert!(rt.last_kernel_stats.iter().any(|s| s.name == "CudaGraph"));
}

#[test]
fn test_cuda_graph_chain_performance() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let size = 4096;
    let mut cx = Graph::default();
    let a = cx.tensor(size).persist();
    let b = cx.tensor(size).persist();
    let mut result = a + b;
    for _ in 0..5 {
        result += a;
        result *= b;
    }
    let output = result.output();

    let mut rt = CudaRuntime::initialize(stream);
    let data_a = random_f32_vec(size, 42, -0.5, 0.5);
    let data_b = random_f32_vec(size, 43, -0.5, 0.5);
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    for _ in 0..10 {
        rt.execute(&cx.dyn_map);
    }
    let mut expected: Vec<f32> = data_a.iter().zip(&data_b).map(|(a, b)| a + b).collect();
    for _ in 0..5 {
        expected = expected.iter().zip(&data_a).map(|(r, a)| r + a).collect();
        expected = expected.iter().zip(&data_b).map(|(r, b)| r * b).collect();
    }
    assert_close(&rt.get_f32(output), &expected, 1e-2, 1e-2);
}

/// Test that CUDA graphs produce correct results when dynamic dimensions
/// change incrementally across many executions (simulating a decode loop
/// where position offset increments each step).
#[test]
fn test_cuda_graph_incremental_dim_changes() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let mut cx = Graph::default();
    let a = cx.tensor('s');
    let b = cx.tensor('s');
    let c = ((a + b) * a).output();

    let initial_size = 128;
    cx.set_dim('s', initial_size);
    let mut rt = CudaRuntime::initialize(stream);
    let data_a = random_f32_vec(initial_size, 42, -0.5, 0.5);
    let data_b = random_f32_vec(initial_size, 43, -0.5, 0.5);
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));

    // Initial execution
    rt.execute(&cx.dyn_map);
    let eps = dtype_epsilon(orbitkv_compiler::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    let expected: Vec<f32> = data_a
        .iter()
        .zip(&data_b)
        .map(|(a, b)| (a + b) * a)
        .collect();
    assert_close(&rt.get_f32(c), &expected, tol, tol);

    // Incrementally change the dynamic dimension 10 times,
    // simulating decode steps where position offset grows.
    for step in 1..=10usize {
        let size = initial_size + step;
        cx.set_dim('s', size);
        let da = random_f32_vec(size, 100 + step as u64, -0.5, 0.5);
        let db = random_f32_vec(size, 200 + step as u64, -0.5, 0.5);
        rt.set_data(a, da.clone());
        rt.set_data(b, db.clone());
        rt.execute(&cx.dyn_map);
        let expected: Vec<f32> = da.iter().zip(&db).map(|(a, b)| (a + b) * a).collect();
        assert_close(&rt.get_f32(c), &expected, tol, tol);
    }
}
