//! Reproducible source discovery for header-only/JIT CUDA providers.

use std::{
    collections::HashMap,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
};

use sha2::{Digest, Sha256};

pub(crate) struct GitDependency {
    pub relative_path: &'static str,
    pub url: &'static str,
    pub revision: &'static str,
}

pub(crate) struct ProviderSource {
    pub name: &'static str,
    pub environment_variable: &'static str,
    pub cache_name: &'static str,
    pub url: &'static str,
    pub revision: &'static str,
    pub markers: &'static [&'static str],
    /// All provider/dependency include trees used by the compiler, relative to
    /// the checkout. Optional trees are hashed as absent when not installed.
    pub source_paths: &'static [&'static str],
    pub dependencies: &'static [GitDependency],
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedProviderSource {
    pub root: PathBuf,
    pub digest: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedCompiler {
    pub executable: PathBuf,
    pub digest: String,
}

impl ProviderSource {
    /// Resolve without fetching, then identify the actual headers, including
    /// local uncommitted changes and dependency trees. Sources are fixed after
    /// first resolution in this process: they MUST remain unchanged until the
    /// process exits; restart compilation after editing them. This is a cache
    /// contract, not a filesystem snapshot. It avoids re-reading large header
    /// trees for every shape/kernel.
    pub(crate) fn resolve_identity(
        &self,
        defaults: &[PathBuf],
    ) -> Result<ResolvedProviderSource, String> {
        type SourceCache = HashMap<(PathBuf, Vec<&'static str>), String>;
        static IDENTITIES: OnceLock<Mutex<SourceCache>> = OnceLock::new();
        let root = self
            .resolve(defaults)?
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let key = (root.clone(), self.source_paths.to_vec());
        let mut cache = IDENTITIES.get_or_init(Default::default).lock().unwrap();
        let digest = match cache.get(&key) {
            Some(digest) => digest.clone(),
            None => {
                let digest = source_digest(&root, self.source_paths)?;
                cache.insert(key, digest.clone());
                digest
            }
        };
        Ok(ResolvedProviderSource { root, digest })
    }

    pub(crate) fn resolve(&self, defaults: &[PathBuf]) -> Result<PathBuf, String> {
        self.resolve_with_cache_root(defaults, &cache_directory())
    }

    fn resolve_with_cache_root(
        &self,
        defaults: &[PathBuf],
        cache_root: &Path,
    ) -> Result<PathBuf, String> {
        if let Some(root) = std::env::var_os(self.environment_variable)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        {
            if self.valid(&root) {
                return Ok(root);
            }
            eprintln!(
                "{}: {}={} is missing required headers; falling back to pinned sources",
                self.name,
                self.environment_variable,
                root.display()
            );
        }

        if let Some(root) = defaults.iter().find(|root| self.valid(root)) {
            return Ok(root.clone());
        }
        let cached = self.pinned_cache_root(cache_root);
        if self.valid(&cached) {
            return Ok(cached);
        }

        Err(format!(
            "{} sources are unavailable; set {} to a pinned checkout or run `cargo run -p orbitkv-cuda --bin fetch-provider -- {}` before compiling models",
            self.name, self.environment_variable, self.cache_name
        ))
    }

    pub(crate) fn prefetch(&self) -> Result<PathBuf, String> {
        if let Ok(root) = self.resolve(&[]) {
            return Ok(root);
        }
        self.fetch()
    }

    fn valid(&self, root: &Path) -> bool {
        self.markers.iter().all(|marker| root.join(marker).exists())
    }

    fn fetch(&self) -> Result<PathBuf, String> {
        let short_revision = self.revision.get(..12).unwrap_or(self.revision);
        let cache_root = self.pinned_cache_root(&cache_directory());
        if self.valid(&cache_root) {
            return Ok(cache_root);
        }

        let parent = cache_root
            .parent()
            .ok_or_else(|| format!("invalid {} provider cache path", self.name))?;
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        let staging = parent.join(format!(
            ".staging-{}-{short_revision}-{}",
            self.cache_name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&staging);

        eprintln!(
            "{}: fetching {} @ {} into {}",
            self.name,
            self.url,
            short_revision,
            cache_root.display()
        );
        clone_at(self.url, self.revision, &staging)?;
        for dependency in self.dependencies {
            let destination = staging.join(dependency.relative_path);
            let _ = std::fs::remove_dir_all(&destination);
            clone_at(dependency.url, dependency.revision, &destination)?;
        }
        if !self.valid(&staging) {
            return Err(format!(
                "{} source fetch completed without the required headers in {}",
                self.name,
                staging.display()
            ));
        }

        match std::fs::rename(&staging, &cache_root) {
            Ok(()) => {}
            Err(_) if self.valid(&cache_root) => {
                let _ = std::fs::remove_dir_all(&staging);
            }
            Err(error) => {
                return Err(format!(
                    "failed to install {} sources at {}: {error}",
                    self.name,
                    cache_root.display()
                ));
            }
        }
        Ok(cache_root)
    }

    fn pinned_cache_root(&self, cache_root: &Path) -> PathBuf {
        cache_root
            .join(self.cache_name)
            .join(self.revision.get(..12).unwrap_or(self.revision))
    }
}

fn hash_part(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

pub(crate) fn content_digest(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hash_part(&mut hasher, part);
    }
    format!("{:x}", hasher.finalize())
}

fn source_digest(root: &Path, paths: &[&str]) -> Result<String, String> {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, b"orbitkv-provider-source-v1");
    let mut paths = paths.to_vec();
    paths.sort_unstable();
    paths.dedup();
    for path in paths {
        hash_source_path(root, Path::new(path), &mut hasher, &mut Vec::new())?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_source_path(
    root: &Path,
    relative: &Path,
    hasher: &mut Sha256,
    ancestors: &mut Vec<PathBuf>,
) -> Result<(), String> {
    hash_part(hasher, relative.as_os_str().as_encoded_bytes());
    let path = root.join(relative);
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Include optional-tree absence in the identity. Dangling links
            // and disappearing required headers must not silently look absent.
            if std::fs::symlink_metadata(&path).is_ok() || !ancestors.is_empty() {
                return Err(format!("provider source disappeared: {}", path.display()));
            }
            hash_part(hasher, b"missing");
            return Ok(());
        }
        Err(error) => return Err(format!("cannot inspect {}: {error}", path.display())),
    };
    if metadata.is_file() {
        hash_part(hasher, b"file");
        hash_part(hasher, &file_digest(&path)?);
    } else if metadata.is_dir() {
        hash_part(hasher, b"directory");
        let canonical = path.canonicalize().map_err(|e| e.to_string())?;
        if ancestors.contains(&canonical) {
            return Err(format!("provider source symlink cycle: {}", path.display()));
        }
        ancestors.push(canonical);
        let mut entries = std::fs::read_dir(&path)
            .map_err(|e| e.to_string())?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        entries.sort_unstable();
        for name in entries {
            hash_source_path(root, &relative.join(name), hasher, ancestors)?;
        }
        ancestors.pop();
    } else {
        return Err(format!(
            "unsupported provider source file: {}",
            path.display()
        ));
    }
    Ok(())
}

fn file_digest(path: &Path) -> Result<Vec<u8>, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_vec())
}

/// Resolve the compiler once per executable. Like provider sources, toolchains
/// and compiler environment must remain fixed for the lifetime of the process.
pub(crate) fn resolve_compiler(compiler: &Path) -> Result<ResolvedCompiler, String> {
    static COMPILERS: OnceLock<Mutex<HashMap<PathBuf, ResolvedCompiler>>> = OnceLock::new();
    let executable = if compiler.components().count() == 1 {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|directory| directory.join(compiler))
            .find(|path| path.is_file())
            .ok_or_else(|| format!("compiler {} was not found in PATH", compiler.display()))?
    } else {
        compiler.to_path_buf()
    }
    .canonicalize()
    .map_err(|e| format!("cannot resolve compiler {}: {e}", compiler.display()))?;
    let mut cache = COMPILERS.get_or_init(Default::default).lock().unwrap();
    if let Some(identity) = cache.get(&executable) {
        return Ok(identity.clone());
    }
    let version = Command::new(&executable)
        .arg("--version")
        .output()
        .map_err(|e| format!("cannot identify compiler {}: {e}", executable.display()))?;
    if !version.status.success() {
        return Err(format!(
            "compiler --version failed: {}",
            executable.display()
        ));
    }
    let digest = content_digest(&[
        b"orbitkv-compiler-v1",
        executable.as_os_str().as_encoded_bytes(),
        &file_digest(&executable)?,
        &version.stdout,
        &version.stderr,
    ]);
    let identity = ResolvedCompiler { executable, digest };
    cache.insert(identity.executable.clone(), identity.clone());
    Ok(identity)
}

/// Declared compilation inputs, excluding the output filename. Target and
/// include/link/compiler flags are supplied in `arguments`. The compiler
/// identity covers nvcc's executable and version, not a hermetic snapshot of
/// its host compiler, ptxas, CUDA headers, or system libraries.
pub(crate) fn compilation_key(
    provider: &str,
    compiler: &str,
    wrapper: &str,
    arguments: &[String],
) -> String {
    let mut hasher = Sha256::new();
    for part in ["orbitkv-provider-library-v1", provider, compiler, wrapper] {
        hash_part(&mut hasher, part.as_bytes());
    }
    for argument in arguments {
        hash_part(&mut hasher, argument.as_bytes());
    }
    // nvcc also accepts flags and include/library paths from its environment.
    for name in [
        "NVCC_CCBIN",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ] {
        hash_part(&mut hasher, name.as_bytes());
        if let Some(value) = std::env::var_os(name) {
            hash_part(&mut hasher, b"set");
            hash_part(&mut hasher, value.as_encoded_bytes());
        } else {
            hash_part(&mut hasher, b"unset");
        }
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn validate_provider_identity(selected: &str, current: &str) -> Result<(), String> {
    if selected == current {
        Ok(())
    } else {
        Err(format!(
            "selected provider identity changed: recorded {selected}, current {current}; rebuild the selected schedule"
        ))
    }
}

fn cache_directory() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into()))
        .join(".cache/orbitkv/providers")
}

fn clone_at(url: &str, revision: &str, destination: &Path) -> Result<(), String> {
    run_git(&[
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        url,
        destination
            .to_str()
            .ok_or_else(|| "provider source path is not UTF-8".to_owned())?,
    ])?;
    run_git_in(destination, &["checkout", revision])
}

fn run_git(arguments: &[&str]) -> Result<(), String> {
    run_command(Command::new("git").args(arguments))
}

fn run_git_in(directory: &Path, arguments: &[&str]) -> Result<(), String> {
    run_command(Command::new("git").current_dir(directory).args(arguments))
}

fn run_command(command: &mut Command) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("failed to run git: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "git command failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/host/provider_source/mod.rs"]
mod tests;
