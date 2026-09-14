//! One cache root for source checkouts and compiled provider libraries.

use std::path::PathBuf;

use super::registry::ProviderId;

pub fn cache_root() -> PathBuf {
    if let Some(path) = std::env::var_os("ORBITKV_CACHE_DIR").filter(|path| !path.is_empty()) {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME").filter(|path| !path.is_empty()) {
        return PathBuf::from(path).join("orbitkv");
    }
    std::env::var_os("HOME")
        .filter(|path| !path.is_empty())
        .map(|path| PathBuf::from(path).join(".cache/orbitkv"))
        .unwrap_or_else(|| std::env::temp_dir().join("orbitkv"))
}

pub(crate) fn source_cache() -> PathBuf {
    cache_root().join("providers")
}

pub(crate) fn library_cache(provider: ProviderId) -> PathBuf {
    cache_root().join("libraries").join(provider.as_str())
}
