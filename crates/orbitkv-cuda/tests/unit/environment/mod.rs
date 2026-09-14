use super::*;

mod gpu;

type EnvironmentMutation = fn(&mut CudaExecutionEnvironment);

fn fixture() -> CudaExecutionEnvironment {
    CudaExecutionEnvironment {
        target: CudaTarget { major: 9, minor: 0 },
        device: DeviceTuningIdentity {
            name: "fixture".into(),
            multiprocessors: 100,
            total_memory_bytes: 1 << 30,
        },
        driver_api_version: 13000,
        nvrtc: NvrtcIdentity {
            version: 13000,
            options: vec!["--gpu-architecture=sm_90".into()],
        },
        provider_lock_digest: "inventory".into(),
        providers: BTreeMap::from([
            (
                ProviderId::CublasLt,
                ProviderIdentity::CudaToolkit { version: 130000 },
            ),
            (
                ProviderId::DeepGemm,
                ProviderIdentity::NativeSource {
                    digest: "headers".into(),
                },
            ),
        ]),
        native_compiler: Some(NativeCompilerIdentity {
            executable_digest: "nvcc".into(),
            environment_digest: "environment".into(),
        }),
        cublaslt_autotune: Some(false),
    }
}

#[test]
fn environment_round_trip_and_identical_replay() {
    let environment = fixture();
    let bytes = serde_json::to_vec(&environment).unwrap();
    let loaded: CudaExecutionEnvironment = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(environment, loaded);
    assert_eq!(bytes, serde_json::to_vec(&loaded).unwrap());
    environment.validate_against(&loaded).unwrap();
}

#[test]
fn executable_changes_require_recompilation_with_component_diagnostics() {
    let saved = fixture();
    let changes: [(&str, EnvironmentMutation); 7] = [
        ("target", |e| e.target.major = 8),
        ("nvrtc", |e| e.nvrtc.version += 10),
        ("nvrtc", |e| e.nvrtc.options.push("--use_fast_math".into())),
        ("native_compiler", |e| {
            e.native_compiler.as_mut().unwrap().executable_digest = "changed".into()
        }),
        ("native_compiler", |e| {
            e.native_compiler.as_mut().unwrap().environment_digest = "changed".into()
        }),
        ("provider.cublaslt", |e| {
            e.providers.insert(
                ProviderId::CublasLt,
                ProviderIdentity::CudaToolkit { version: 130100 },
            );
        }),
        ("provider.deepgemm", |e| {
            e.providers.insert(
                ProviderId::DeepGemm,
                ProviderIdentity::NativeSource {
                    digest: "edited header".into(),
                },
            );
        }),
    ];
    for (component, change) in changes {
        let mut current = saved.clone();
        change(&mut current);
        let mismatch = saved.validate_against(&current).unwrap_err();
        assert_eq!(mismatch.changes.len(), 1, "{mismatch}");
        assert_eq!(mismatch.changes[0].component, component);
        assert_eq!(mismatch.changes[0].recovery, EnvironmentRecovery::Recompile);
        assert!(mismatch.to_string().contains("requires recompilation"));
    }
}

#[test]
fn timing_changes_require_retuning_without_claiming_binary_incompatibility() {
    let saved = fixture();
    let changes: [fn(&mut CudaExecutionEnvironment); 6] = [
        |e| e.device.name = "another device".into(),
        |e| e.device.multiprocessors -= 1,
        |e| e.device.total_memory_bytes /= 2,
        |e| e.driver_api_version += 10,
        |e| e.provider_lock_digest = "new inventory".into(),
        |e| e.cublaslt_autotune = Some(true),
    ];
    for change in changes {
        let mut current = saved.clone();
        change(&mut current);
        let mismatch = saved.validate_against(&current).unwrap_err();
        assert_eq!(mismatch.changes.len(), 1, "{mismatch}");
        assert_eq!(mismatch.changes[0].recovery, EnvironmentRecovery::Retune);
        assert!(mismatch.to_string().contains("requires retuning"));
    }
}

#[test]
fn every_difference_is_reported_and_provider_omissions_fail_closed() {
    let saved = fixture();
    let mut current = saved.clone();
    current.providers.remove(&ProviderId::CublasLt);
    current.driver_api_version += 10;
    let mismatch = saved.validate_against(&current).unwrap_err();
    assert_eq!(mismatch.changes.len(), 2);
    assert!(mismatch.to_string().contains("provider.cublaslt"));
    assert!(mismatch.to_string().contains("driver_api_version"));
    assert!(current.validate_against(&saved).is_err());
}
