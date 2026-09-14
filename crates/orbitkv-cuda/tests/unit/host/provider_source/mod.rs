use super::*;

struct SourceTree(PathBuf);

impl SourceTree {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "orbitkv-provider-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }

    fn write(&self, relative: &str, content: &str) {
        let path = self.0.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
}

impl Drop for SourceTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn test_source() -> ProviderSource {
    ProviderSource {
        name: "test",
        environment_variable: "ORBITKV_UNUSED_TEST_PROVIDER_DIR",
        cache_name: "test",
        url: "unused",
        revision: "123456789012abcdef",
        markers: &["include/provider"],
        source_paths: &["include", "dependency/include"],
        dependencies: &[],
    }
}

#[test]
fn existing_provider_directory_prevents_a_fetch() {
    let root = std::env::temp_dir().join(format!("orbitkv-provider-source-{}", std::process::id()));
    let marker = root.join("include/provider");
    std::fs::create_dir_all(&marker).unwrap();
    let source = ProviderSource {
        name: "test",
        environment_variable: "ORBITKV_UNUSED_TEST_PROVIDER_DIR",
        cache_name: "test",
        url: "unused",
        revision: "unused",
        markers: &["include/provider"],
        source_paths: &["include"],
        dependencies: &[],
    };
    assert_eq!(
        source.resolve(std::slice::from_ref(&root)),
        Ok(root.clone())
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn prefetched_source_resolves_without_environment_or_network() {
    let cache = SourceTree::new();
    let source = test_source();
    let pinned = source.pinned_cache_root(&cache.0);
    std::fs::create_dir_all(pinned.join("include/provider")).unwrap();
    assert_eq!(
        source.resolve_with_cache_root(&[], &cache.0).unwrap(),
        pinned
    );
}

#[test]
fn header_and_dependency_edits_invalidate_cached_library_and_schedule() {
    let tree = SourceTree::new();
    tree.write("include/provider/main.h", "provider original");
    tree.write("dependency/include/math.h", "dependency original");
    let paths = test_source().source_paths;
    let original = source_digest(&tree.0, paths).unwrap();
    let args = vec!["-arch=sm_90a".to_owned(), "-O3".to_owned()];
    let old_key = compilation_key(&original, "nvcc", "wrapper", &args);
    tree.write(&format!("{old_key}.so"), "previously compiled library");
    tree.write("include/provider/main.h", "provider modified");
    let edited = source_digest(&tree.0, paths).unwrap();
    let new_key = compilation_key(&edited, "nvcc", "wrapper", &args);
    assert_ne!(old_key, new_key);
    assert!(!tree.0.join(format!("{new_key}.so")).exists());
    assert!(
        validate_provider_identity(&original, &edited)
            .unwrap_err()
            .contains("rebuild")
    );
    tree.write("dependency/include/math.h", "dependency modified");
    let dependency_edit = source_digest(&tree.0, paths).unwrap();
    assert_ne!(edited, dependency_edit);
    assert!(validate_provider_identity(&edited, &dependency_edit).is_err());
    assert!(validate_provider_identity(&dependency_edit, &dependency_edit).is_ok());
}

#[test]
fn source_identity_is_content_based_and_covers_file_addition_and_removal() {
    let first = SourceTree::new();
    let second = SourceTree::new();
    first.write("include/z.h", "z");
    first.write("include/a.h", "a");
    second.write("include/a.h", "a");
    second.write("include/z.h", "z");
    let original = source_digest(&first.0, &["include"]).unwrap();
    assert_eq!(original, source_digest(&second.0, &["include"]).unwrap());
    second.write("include/new.h", "new");
    assert_ne!(original, source_digest(&second.0, &["include"]).unwrap());
    std::fs::remove_file(second.0.join("include/new.h")).unwrap();
    assert_eq!(original, source_digest(&second.0, &["include"]).unwrap());
    std::fs::remove_file(second.0.join("include/z.h")).unwrap();
    assert_ne!(original, source_digest(&second.0, &["include"]).unwrap());
}

#[test]
fn source_identity_is_memoized_with_an_explicit_process_lifetime_contract() {
    let tree = SourceTree::new();
    tree.write("include/provider/main.h", "first");
    let source = test_source();
    let first = source
        .resolve_identity(std::slice::from_ref(&tree.0))
        .unwrap();
    tree.write("include/provider/main.h", "changed after first resolution");
    let same_process = source
        .resolve_identity(std::slice::from_ref(&tree.0))
        .unwrap();
    assert_eq!(first.digest, same_process.digest);
    assert_ne!(
        first.digest,
        source_digest(&tree.0, source.source_paths).unwrap()
    );
}

#[test]
fn compilation_key_separates_wrapper_compiler_flags_and_target() {
    let args = vec!["-arch=sm_90a".to_owned(), "-O3".to_owned()];
    let original = compilation_key("provider", "nvcc-1", "wrapper-1", &args);
    assert_ne!(
        original,
        compilation_key("provider", "nvcc-2", "wrapper-1", &args)
    );
    assert_ne!(
        original,
        compilation_key("provider", "nvcc-1", "wrapper-2", &args)
    );
    let mut other_target = args.clone();
    other_target[0] = "-arch=sm_80".to_owned();
    assert_ne!(
        original,
        compilation_key("provider", "nvcc-1", "wrapper-1", &other_target)
    );
    let mut other_flags = args.clone();
    other_flags.push("--use_fast_math".to_owned());
    assert_ne!(
        original,
        compilation_key("provider", "nvcc-1", "wrapper-1", &other_flags)
    );
    assert_eq!(
        original,
        compilation_key("provider", "nvcc-1", "wrapper-1", &args)
    );
}

#[cfg(unix)]
#[test]
fn symlinked_header_contents_are_hashed_and_directory_cycles_are_rejected() {
    let tree = SourceTree::new();
    tree.write("outside/header.h", "first");
    std::fs::create_dir_all(tree.0.join("include")).unwrap();
    std::os::unix::fs::symlink("../outside/header.h", tree.0.join("include/header.h")).unwrap();
    let first = source_digest(&tree.0, &["include"]).unwrap();
    tree.write("outside/header.h", "second");
    assert_ne!(first, source_digest(&tree.0, &["include"]).unwrap());
    std::os::unix::fs::symlink(".", tree.0.join("include/loop")).unwrap();
    assert!(
        source_digest(&tree.0, &["include"])
            .unwrap_err()
            .contains("cycle")
    );
}
