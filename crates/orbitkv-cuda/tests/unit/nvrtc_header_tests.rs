use super::*;

#[test]
fn parses_cuda_version_define() {
    assert_eq!(
        parse_cuda_header_version("#pragma once\n#define CUDA_VERSION 12080\n"),
        Some(12080)
    );
    assert_eq!(
        parse_cuda_header_version("#define SOMETHING_ELSE 1\n"),
        None
    );
}

#[test]
fn nvrtc_version_encoding_is_checked() {
    assert_eq!(encode_nvrtc_version(12, 1), Some(12010));
    assert_eq!(encode_nvrtc_version(13, 3), Some(13030));
    assert_eq!(encode_nvrtc_version(-1, 3), None);
    assert_eq!(encode_nvrtc_version(13, -1), None);
    assert_eq!(encode_nvrtc_version(c_int::MAX, 0), None);
}

#[test]
fn missing_nvrtc_library_is_optional() {
    let missing = std::env::temp_dir().join(format!(
        "orbitkv-missing-nvrtc-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("unnamed")
    ));
    assert_eq!(query_nvrtc_version([missing]), None);
}

#[test]
fn nvrtc_probe_matches_cudarc_loader_order() {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let pointer_width = if cfg!(target_pointer_width = "32") {
        "32"
    } else {
        "64"
    };
    let major = driver_sys::CUDA_VERSION / 1000;
    let minor = (driver_sys::CUDA_VERSION % 1000) / 10;
    let expected = [
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}{minor}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}{minor}_0{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}0_{minor}{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_10{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_11{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_12{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_{major}0_0{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{pointer_width}_9{DLL_SUFFIX}"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.{major}"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.12"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.11"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.10"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.9"),
        format!("{DLL_PREFIX}nvrtc{DLL_SUFFIX}.1"),
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect::<Vec<_>>();

    assert_eq!(nvrtc_library_candidates(), expected);
}
