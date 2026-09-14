use super::*;

#[test]
#[ignore = "requires CUDA; queries only providers declared by a selected program"]
fn execution_context_and_selected_libraries_are_validated() {
    let context = CudaContext::new(0).expect("CUDA device required");
    let generated = CudaExecutionEnvironment::capture(&context, []).unwrap();
    assert!(generated.providers.is_empty());
    assert!(generated.native_compiler.is_none());
    assert_eq!(
        generated.target,
        CudaTarget::from_context(&context).unwrap()
    );
    generated.validate_for_device(&context).unwrap();
    let saved = CudaExecutionEnvironment::capture(&context, [ProviderId::CublasLt]).unwrap();
    saved.validate_for_device(&context).unwrap();
    let mut changed = saved.clone();
    let ProviderIdentity::CudaToolkit { version } =
        changed.providers.get_mut(&ProviderId::CublasLt).unwrap()
    else {
        panic!("toolkit provider")
    };
    *version += 1;
    let error = changed.validate_for_device(&context).unwrap_err();
    assert!(
        error.contains("provider.cublaslt requires recompilation"),
        "{error}"
    );
}
