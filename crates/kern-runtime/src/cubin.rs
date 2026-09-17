//! Getting CUDA modules out of kernel artifacts: local cubins, host shared
//! libraries with embedded fatbins, and registry refs (`hf:...`) fetched
//! into a content-addressed cache. Also the SASS scan that keeps multicast
//! TMA away from peer memory.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;

use cudarc::driver::{result as cu, sys};
use kern_manifest::types::{Manifest, RegistryRef};
use sha2::Digest;

use crate::error::{bail, cuda_check, Error, Result};

/// Parameter byte sizes of a loaded function, per `cuFuncGetParamInfo`.
pub(crate) fn param_sizes(func: sys::CUfunction) -> Result<Vec<usize>> {
    let mut sizes = Vec::new();
    loop {
        let (mut off, mut size) = (0usize, 0usize);
        let r = unsafe { sys::cuFuncGetParamInfo(func, sizes.len(), &mut off, &mut size) };
        if r == sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            return Ok(sizes);
        }
        cuda_check(r, "cuFuncGetParamInfo")?;
        sizes.push(size);
    }
}

/// Materialize every registry ref in the manifest's `modules` table into the
/// local cache. The returned map is keyed by the full ref string, so module
/// pinning matches registry artifacts like any local file name.
pub(crate) fn fetch_registry_cubins(manifest: &Manifest) -> Result<BTreeMap<String, PathBuf>> {
    let mut remote = BTreeMap::new();
    for (name, md) in &manifest.modules {
        let Some(reg) = RegistryRef::parse(&md.source) else {
            continue;
        };
        let reg = reg.map_err(|e| Error::Manifest(format!("module `{name}`: {e}")))?;
        if remote.contains_key(&md.source) {
            continue;
        }
        let path = fetch_registry_cubin(&reg, &md.sha256)?;
        remote.insert(md.source.clone(), path);
    }
    Ok(remote)
}

/// One loaded device module: the hash that identifies it (what a manifest
/// `modules` entry pins), a label for humans (the file name it came from, or
/// the registry ref), and the driver handle.
pub(crate) struct LoadedModule {
    pub sha: String,
    pub label: String,
    /// The artifact on disk, for the SASS scan.
    pub path: PathBuf,
    pub module: sys::CUmodule,
}

/// Load the modules the manifest pins, keyed by sha256 of the file. Every
/// `.cubin` in the directory is hashed; only artifacts whose hash the
/// manifest's `modules` table names are loaded — the manifest is the
/// complete dependency list, the directory may hold every version ever
/// built (`gemm8-3f9a1c2d4e5b.cubin`, `gemm8-9b0c…`) and each manifest
/// resolves to the one it pins. Two files with the same bytes load once.
pub(crate) fn load_pinned_modules(
    kernels_dir: &Path,
    remote: &BTreeMap<String, PathBuf>,
    wanted: &BTreeSet<String>,
) -> Result<Vec<LoadedModule>> {
    let mut cubins: Vec<_> = std::fs::read_dir(kernels_dir)
        .map_err(|e| Error::KernelArtifact(format!("kernel dir {}: {e}", kernels_dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "cubin"))
        .collect();
    cubins.sort();
    if cubins.is_empty() && remote.is_empty() && !wanted.is_empty() {
        bail!(KernelArtifact, "no .cubin files in {}", kernels_dir.display());
    }
    let mut modules: Vec<LoadedModule> = Vec::new();
    let local = cubins.iter().map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p.clone()));
    for (label, path) in local.chain(remote.iter().map(|(r, p)| (r.clone(), p.clone()))) {
        let bytes =
            std::fs::read(&path).map_err(|e| Error::KernelArtifact(format!("reading {}: {e}", path.display())))?;
        let sha = hex::encode(sha2::Sha256::digest(&bytes));
        if !wanted.contains(&sha) {
            tracing::debug!("{label} ({}): not pinned by this manifest, not loaded", &sha[..12]);
            continue;
        }
        if let Some(prev) = modules.iter().find(|m| m.sha == sha) {
            tracing::debug!("{label}: same bytes as {} ({}), not loaded twice", prev.label, &sha[..12]);
            continue;
        }
        for module in load_device_modules(&path)? {
            modules.push(LoadedModule { sha: sha.clone(), label: label.clone(), path: path.clone(), module });
        }
    }
    Ok(modules)
}

/// Materialize a registry cubin into the content-addressed cache
/// (`$KERN_CACHE_DIR` or `~/.cache/kern`, `blobs/<sha256>`) and return its
/// local path. A cache hit is re-hashed (corruption re-fetches); a download
/// is hash-checked before it lands in the cache, so the transport — the
/// Hugging Face `resolve/` endpoint or whatever fronts it — is untrusted.
fn fetch_registry_cubin(reg: &RegistryRef, sha256: &str) -> Result<PathBuf> {
    let sha = sha256.to_lowercase();
    let cache_root = std::env::var_os("KERN_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/kern")))
        .ok_or_else(|| {
            Error::KernelArtifact("registry cubin cache: neither $KERN_CACHE_DIR nor $HOME is set".into())
        })?;
    let blobs = cache_root.join("blobs");
    let cached = blobs.join(&sha);
    if let Ok(data) = std::fs::read(&cached) {
        if hex::encode(sha2::Sha256::digest(&data)) == sha {
            return Ok(cached);
        }
    }

    let url = format!("https://huggingface.co/{}/{}/resolve/{}/{}", reg.org, reg.repo, reg.revision, reg.path);
    tracing::info!("fetching {url}");
    let mut req = ureq::get(&url);
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", format!("Bearer {tok}"));
    }
    let resp = req.call().map_err(|e| Error::KernelArtifact(format!("fetching {url}: {e}")))?;
    let mut data = Vec::new();
    std::io::Read::read_to_end(&mut resp.into_body().into_reader(), &mut data)
        .map_err(|e| Error::KernelArtifact(format!("reading {url}: {e}")))?;
    let got = hex::encode(sha2::Sha256::digest(&data));
    if got != sha {
        bail!(
            KernelArtifact,
            "registry cubin hf:{}/{}/{}@{}: sha256 mismatch: manifest declares {sha}, \
             fetched bytes are {got}",
            reg.org,
            reg.repo,
            reg.path,
            reg.revision
        );
    }
    std::fs::create_dir_all(&blobs)?;
    let tmp = blobs.join(format!("{sha}.tmp.{}", std::process::id()));
    std::fs::write(&tmp, &data)?;
    std::fs::rename(&tmp, &cached)?;
    Ok(cached)
}

fn le16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn le32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn le64(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}

/// Find a named section in a 64-bit little-endian ELF.
fn elf_section<'a>(bytes: &'a [u8], name: &str) -> Option<&'a [u8]> {
    if bytes.get(..4)? != b"\x7fELF" || *bytes.get(4)? != 2 {
        return None;
    }
    let shoff = le64(bytes, 0x28)? as usize;
    let shentsize = le16(bytes, 0x3a)? as usize;
    let shnum = le16(bytes, 0x3c)? as usize;
    let shstrndx = le16(bytes, 0x3e)? as usize;
    let sh = |i: usize| {
        let base = shoff.checked_add(i.checked_mul(shentsize)?)?;
        Some((le32(bytes, base)? as usize, le64(bytes, base + 0x18)? as usize, le64(bytes, base + 0x20)? as usize))
    };
    let (_, stroff, strsz) = sh(shstrndx)?;
    let strtab = bytes.get(stroff..stroff.checked_add(strsz)?)?;
    for i in 0..shnum {
        let (n, off, size) = sh(i)?;
        let nm = strtab.get(n..)?.split(|&c| c == 0).next()?;
        if nm == name.as_bytes() {
            return bytes.get(off..off.checked_add(size)?);
        }
    }
    None
}

const FATBIN_MAGIC: u32 = 0xba55_ed50;

/// Split an `.nv_fatbin` section into its fatbin containers (one per
/// translation unit): each is a 16-byte header (magic, version, header size,
/// payload size) followed by the payload, 8-aligned with zero padding.
fn split_fatbins(sec: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 16 <= sec.len() && le32(sec, off) == Some(FATBIN_MAGIC) {
        let hsize = le16(sec, off + 6).unwrap() as usize;
        let fatsize = le64(sec, off + 8).unwrap() as usize;
        let Some(end) = hsize.checked_add(fatsize).and_then(|t| off.checked_add(t)) else {
            break;
        };
        let Some(chunk) = sec.get(off..end) else { break };
        out.push(chunk);
        off = end;
        while off < sec.len() && sec[off] == 0 {
            off += 1;
        }
    }
    out
}

/// ELF `e_machine` value for device-code objects: a raw `.cubin` is itself
/// an ELF, but for the CUDA architecture.
const EM_CUDA: u16 = 190;

/// Load every device-code module an artifact carries. A raw cubin loads
/// directly. A host shared library — e.g. a torch-extension kernel package
/// from the HF kernel hub — has its device code embedded in `.nv_fatbin`;
/// the host half of the .so (torch/python bindings) is dead weight to kern,
/// so each embedded fatbin container is loaded on its own and the usual
/// entry + param-layout resolution picks the right function out.
fn load_device_modules(path: &Path) -> Result<Vec<sys::CUmodule>> {
    let bytes = std::fs::read(path).map_err(|e| Error::KernelArtifact(format!("reading {}: {e}", path.display())))?;
    let host_elf = bytes.get(..4) == Some(b"\x7fELF".as_slice()) && le16(&bytes, 0x12) != Some(EM_CUDA);
    if !host_elf {
        let cpath = CString::new(path.to_str().unwrap())
            .map_err(|e| Error::KernelArtifact(format!("cubin path {}: {e}", path.display())))?;
        let cmod =
            cu::module::load(cpath).map_err(|e| Error::KernelArtifact(format!("loading {}: {e:?}", path.display())))?;
        return Ok(vec![cmod]);
    }
    let sec = elf_section(&bytes, ".nv_fatbin")
        .ok_or_else(|| Error::KernelArtifact(format!("{}: host ELF without a .nv_fatbin section", path.display())))?;
    let chunks = split_fatbins(sec);
    if chunks.is_empty() {
        bail!(KernelArtifact, "{}: .nv_fatbin holds no fatbin containers", path.display());
    }
    let mut mods = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        // Copy to an 8-aligned buffer; the slice into the ELF has no
        // alignment guarantee.
        let mut buf = vec![0u64; chunk.len().div_ceil(8)];
        unsafe {
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), buf.as_mut_ptr() as *mut u8, chunk.len());
        }
        let mut m: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleLoadData(&mut m, buf.as_ptr() as *const c_void) };
        match r {
            sys::CUresult::CUDA_SUCCESS => mods.push(m),
            // A container may hold only device code this GPU can't use
            // (e.g. relocatable-only); resolution below just won't see it.
            _ => tracing::warn!("{}: fatbin container #{i} not loadable: {r:?}", path.display()),
        }
    }
    if mods.is_empty() {
        bail!(KernelArtifact, "{}: no loadable fatbin container", path.display());
    }
    Ok(mods)
}

/// The one thing a kernel may not do to peer memory: multicast TMA
/// (`cp.async.bulk{.tensor}...multicast::cluster`, SASS `UTMALDG.*.MULTICAST`
/// and friends) issued at a fabric or peer address wedges the issuing GPU
/// until a node reboot (GB300, driver 580: "Route Unhealthy: GPU requires
/// reset"). Plain loads/stores, atomics, non-multicast TMA and 1-D bulk
/// copies are fine. The verifier cannot see SASS, so the runtime disassembles
/// every entry that receives a peer-derived pointer and refuses any
/// multicast instruction in it. No `cuobjdump`, no proof, no launch.
pub(crate) struct MulticastScan {
    tool: Option<PathBuf>,
    /// artifact -> function -> offending instructions
    cache: BTreeMap<PathBuf, BTreeMap<String, Vec<String>>>,
}

impl MulticastScan {
    pub(crate) fn new() -> MulticastScan {
        MulticastScan { tool: find_cuobjdump(), cache: BTreeMap::new() }
    }

    /// Multicast instructions in `entry` of the artifact at `path`; an
    /// error when the artifact cannot be disassembled or `entry` is not in
    /// the disassembly (fail closed).
    pub(crate) fn offending(&mut self, path: &Path, entry: &str) -> Result<Vec<String>> {
        if !self.cache.contains_key(path) {
            let Some(tool) = &self.tool else {
                bail!(
                    KernelArtifact,
                    "{}: no cuobjdump to scan `{entry}` for multicast TMA before it touches peer memory \
                     (set KERN_CUOBJDUMP or CUDA_HOME, or put cuobjdump on PATH)",
                    path.display()
                );
            };
            let out = Command::new(tool)
                .arg("-sass")
                .arg(path)
                .output()
                .map_err(|e| Error::KernelArtifact(format!("running {}: {e}", tool.display())))?;
            if !out.status.success() {
                bail!(
                    KernelArtifact,
                    "{} -sass {}: {}\n{}",
                    tool.display(),
                    path.display(),
                    out.status,
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            let text = String::from_utf8_lossy(&out.stdout);
            self.cache.insert(path.to_path_buf(), multicast_by_function(&text));
        }
        match self.cache[path].get(entry) {
            Some(v) => Ok(v.clone()),
            None => bail!(
                KernelArtifact,
                "{}: `{entry}` not found in the cuobjdump disassembly, cannot prove it free of multicast TMA",
                path.display()
            ),
        }
    }
}

fn find_cuobjdump() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("KERN_CUOBJDUMP") {
        return Some(PathBuf::from(p));
    }
    for root in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(r) = std::env::var_os(root) {
            let p = PathBuf::from(r).join("bin/cuobjdump");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let p = dir.join("cuobjdump");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let p = PathBuf::from("/usr/local/cuda/bin/cuobjdump");
    p.is_file().then_some(p)
}

/// Parse `cuobjdump -sass` output into function -> instructions whose
/// opcode carries `.MULTICAST`. Every `Function : <sym>` block is a
/// function; a symbol appearing in several containers (same-name Triton
/// instances) accumulates, so one bad instance taints the name.
fn multicast_by_function(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("Function :") {
            let name = name.trim().to_string();
            out.entry(name.clone()).or_default();
            cur = Some(name);
            continue;
        }
        let Some(f) = &cur else { continue };
        // `/*0120*/  @P0 UTMALDG.2D.MULTICAST [UR4], [UR8] ;  /* 0x... */`
        let Some(rest) = t.strip_prefix("/*") else { continue };
        let Some(end) = rest.find("*/") else { continue };
        let body = rest[end + 2..].trim();
        let opcode = body.split_whitespace().find(|tok| !tok.starts_with('@')).unwrap_or("").trim_end_matches(';');
        // Only memory-moving multicast wedges the GPU at a peer address:
        // TMA (UTMALDG/UTMASTG/UTMAREDG/UTMACCTL) and bulk copies
        // (UBLKCP/UBLKRED). A cluster barrier's multicast arrive
        // (UTCBAR.2CTA.MULTICAST) never touches global memory.
        let is_copy = opcode.starts_with("UTMA") || opcode.starts_with("UBLK");
        if is_copy && opcode.contains(".MULTICAST") {
            out.get_mut(f).unwrap().push(body.split("/*").next().unwrap_or(body).trim().to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SASS: &str = r#"
	code for sm_103a
		Function : k_tma_mcast
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM103 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM103)"
        /*0000*/                   LDC R1, c[0x0][0x28] ;                       /* 0x00000a00ff017b82 */
        /*0010*/              @!P0 UTMALDG.2D.MULTICAST [UR4], [UR8] ;          /* 0x0000000004007fab */
        /*0020*/                   UBLKCP.S.G [UR4], [UR8] ;                    /* 0x0000000004007fab */
        /*0030*/                   UTCBAR.2CTA.MULTICAST.ARRIVE UR4 ;           /* 0x0000000004007fab */
        /*0040*/                   EXIT ;                                       /* 0x000000000000794d */
		Function : k_bulk
        /*0000*/                   UBLKCP.S.G [UR4], [UR8] ;
        /*0010*/                   UTMALDG.2D [UR4], [UR8] ;
        /*0020*/                   EXIT ;
		Function : _Z7k_plainPf
        /*0000*/                   STG.E [R2.64], R5 ;
        /*0010*/                   EXIT ;
"#;

    #[test]
    fn multicast_only_flags_multicast() {
        let m = multicast_by_function(SASS);
        assert_eq!(m["k_tma_mcast"], vec!["@!P0 UTMALDG.2D.MULTICAST [UR4], [UR8] ;"]);
        assert!(m["k_bulk"].is_empty(), "{:?}", m["k_bulk"]);
        assert!(m["_Z7k_plainPf"].is_empty());
        assert!(!m.contains_key("ghost"));
    }
}
