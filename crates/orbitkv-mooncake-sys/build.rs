use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const RUNTIME_LIBRARIES: [(&str, &str); 3] = [
    ("src/libtransfer_engine.so", "libtransfer_engine.so"),
    (
        "mooncake-common-src/libmooncake_common.so",
        "libmooncake_common.so",
    ),
    ("mooncake-common/libasio.so", "libasio.so"),
];
fn main() {
    println!("cargo:rerun-if-env-changed=ORBITKV_MOONCAKE_BUILD_JOBS");
    println!("cargo:rerun-if-env-changed=ORBITKV_MOONCAKE_CMAKE");
    println!("cargo:rerun-if-env-changed=ORBITKV_MOONCAKE_LIB_DIR");

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let workspace = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("sys crate lives under <workspace>/crates");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let build_dir = out_dir.join("native");
    let variant = if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        "cuda"
    } else {
        "cpu"
    };
    let runtime_dir = workspace
        .join(".orbitkv/mooncake")
        .join(variant)
        .join("lib");
    fs::create_dir_all(&runtime_dir).expect("create Mooncake runtime directory");

    if let Some(prebuilt_dir) = env::var_os("ORBITKV_MOONCAKE_LIB_DIR") {
        stage_prebuilt_libraries(Path::new(&prebuilt_dir), &runtime_dir);
        set_origin_runpaths(&runtime_dir);
        return;
    }

    let source = workspace.join("third-party/mooncake/mooncake-transfer-engine");
    let pybind = workspace.join("third-party/mooncake/extern/pybind11/CMakeLists.txt");
    assert!(
        source.join("CMakeLists.txt").is_file() && pybind.is_file(),
        "Mooncake source is incomplete; run `git submodule update --init --recursive \
         third-party/mooncake`"
    );
    println!(
        "cargo:rerun-if-changed={}",
        workspace.join("third-party/mooncake").display()
    );

    configure(&source, &build_dir);
    build(&build_dir);
    stage_libraries(&build_dir, &runtime_dir);
    set_origin_runpaths(&runtime_dir);
}

fn configure(source: &Path, build_dir: &Path) {
    let cmake = env::var("ORBITKV_MOONCAKE_CMAKE").unwrap_or_else(|_| "cmake".to_string());
    let cuda = if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        "ON"
    } else {
        "OFF"
    };
    let status = Command::new(&cmake)
        .args(["-S"])
        .arg(source)
        .args(["-B"])
        .arg(build_dir)
        .args([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_BUILD_RPATH=$ORIGIN",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DBUILD_UNIT_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_BENCHMARK=OFF",
            "-DWITH_RUST_EXAMPLE=OFF",
            "-DUSE_HTTP=OFF",
            "-DUSE_ETCD=OFF",
            "-DUSE_REDIS=OFF",
            "-DUSE_TENT=OFF",
            "-DWITH_METRICS=OFF",
        ])
        .arg(format!("-DUSE_CUDA={cuda}"))
        .status()
        .unwrap_or_else(|error| panic!("failed to run {cmake}: {error}"));
    assert!(
        status.success(),
        "Mooncake CMake configure failed; install its documented build dependencies"
    );
}

fn build(build_dir: &Path) {
    let cmake = env::var("ORBITKV_MOONCAKE_CMAKE").unwrap_or_else(|_| "cmake".to_string());
    let jobs = env::var("ORBITKV_MOONCAKE_BUILD_JOBS").unwrap_or_else(|_| "8".to_string());
    let status = Command::new(&cmake)
        .args(["--build"])
        .arg(build_dir)
        .args(["--target", "transfer_engine", "--parallel"])
        .arg(&jobs)
        .status()
        .unwrap_or_else(|error| panic!("failed to build Mooncake Transfer Engine: {error}"));
    assert!(status.success(), "Mooncake Transfer Engine build failed");
}

fn stage_libraries(build_dir: &Path, link_dir: &Path) {
    for (relative, name) in RUNTIME_LIBRARIES {
        let source = build_dir.join(relative);
        assert!(
            source.is_file(),
            "Mooncake build did not produce {}",
            source.display()
        );
        fs::copy(&source, link_dir.join(name))
            .unwrap_or_else(|error| panic!("failed to stage {}: {error}", source.display()));
    }
}

fn stage_prebuilt_libraries(source_dir: &Path, runtime_dir: &Path) {
    for (_, name) in RUNTIME_LIBRARIES {
        let source = source_dir.join(name);
        assert!(
            source.is_file(),
            "ORBITKV_MOONCAKE_LIB_DIR is missing {}",
            source.display()
        );
        let destination = runtime_dir.join(name);
        if source != destination {
            fs::copy(&source, &destination)
                .unwrap_or_else(|error| panic!("failed to stage {}: {error}", source.display()));
        }
    }
}

fn set_origin_runpaths(runtime_dir: &Path) {
    let patchelf = Command::new("patchelf").arg("--version").status();
    if !patchelf.is_ok_and(|status| status.success()) {
        println!(
            "cargo:warning=patchelf not found; staged Mooncake libraries may retain build-directory RUNPATH entries"
        );
        return;
    }
    for (_, name) in RUNTIME_LIBRARIES {
        let library = runtime_dir.join(name);
        let status = Command::new("patchelf")
            .args(["--set-rpath", "$ORIGIN"])
            .arg(&library)
            .status()
            .unwrap_or_else(|error| {
                panic!("failed to run patchelf for {}: {error}", library.display())
            });
        assert!(
            status.success(),
            "failed to make {} relocatable",
            library.display()
        );
    }
}
