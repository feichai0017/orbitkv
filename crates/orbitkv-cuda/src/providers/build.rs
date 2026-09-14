//! Shared native build, cache publication, and bounded compiler diagnostics.

use std::{
    io::Read,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};

use super::{
    cache::library_cache,
    provider_source::{ResolvedProviderSource, compilation_key, resolve_compiler},
    registry::ProviderId,
};

const DEFAULT_COMPILE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const MAX_DIAGNOSTIC_BYTES: usize = 1024 * 1024;
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Every flag that affects the translation unit is part of the cache identity.
pub(crate) struct NativeBuild<'a> {
    pub provider: ProviderId,
    pub sources: &'a ResolvedProviderSource,
    pub source: &'a str,
    pub arguments: Vec<String>,
    pub link_arguments: Vec<String>,
}

impl NativeBuild<'_> {
    pub fn compile(self) -> Result<PathBuf> {
        let stage = tracing::info_span!(target: "orbitkv::stage", "cuda.provider.jit", provider = %self.provider, cache_hit = tracing::field::Empty);
        let _entered = stage.enter();
        let compiler = native_compiler()?;
        let mut identity_arguments = self.arguments.clone();
        identity_arguments.push("<translation-unit>".into());
        identity_arguments.extend(self.link_arguments.iter().cloned());
        let key = compilation_key(
            &self.sources.digest,
            &compiler.digest,
            self.source,
            &identity_arguments,
        );
        let cache = library_cache(self.provider);
        std::fs::create_dir_all(&cache)?;
        let directory = cache.join(&key);
        let library = directory.join("provider.so");
        if library.is_file() && library.metadata()?.len() > 0 {
            stage.record("cache_hit", true);
            return Ok(library);
        }
        stage.record("cache_hit", false);
        let staging = BuildDirectory(cache.join(format!(".staging-{}", uuid::Uuid::new_v4())));
        std::fs::create_dir(&staging.0)?;
        let translation_unit = staging.0.join("provider.cu");
        let output_library = staging.0.join("provider.so");
        std::fs::write(&translation_unit, self.source)?;
        let mut command = Command::new(&compiler.executable);
        command
            .args(&self.arguments)
            .arg(&translation_unit)
            .args(&self.link_arguments)
            .arg("-o")
            .arg(&output_library);
        run_compiler(&mut command, compile_timeout()?)
            .with_context(|| format!("{} native compilation failed", self.provider))?;
        ensure!(
            output_library
                .metadata()
                .is_ok_and(|metadata| metadata.len() > 0),
            "{} compiler did not produce a shared library",
            self.provider
        );
        match std::fs::rename(&staging.0, &directory) {
            Ok(()) => {}
            Err(_) if library.is_file() && library.metadata()?.len() > 0 => {}
            Err(error) => return Err(error).context("publish compiled provider library"),
        }
        Ok(library)
    }
}

fn compile_timeout() -> Result<Duration> {
    match std::env::var("ORBITKV_NVCC_TIMEOUT_SECONDS") {
        Ok(value) => {
            let seconds: u64 = value
                .parse()
                .context("invalid ORBITKV_NVCC_TIMEOUT_SECONDS")?;
            ensure!(seconds > 0, "ORBITKV_NVCC_TIMEOUT_SECONDS must be positive");
            Ok(Duration::from_secs(seconds))
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_COMPILE_TIMEOUT),
        Err(error) => Err(error).context("invalid ORBITKV_NVCC_TIMEOUT_SECONDS"),
    }
}

pub(crate) fn native_compiler() -> Result<super::provider_source::ResolvedCompiler> {
    resolve_compiler(&nvcc_path()?).map_err(anyhow::Error::msg)
}

fn nvcc_path() -> Result<PathBuf> {
    for name in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(name).filter(|value| !value.is_empty()) {
            let compiler = PathBuf::from(root).join("bin/nvcc");
            ensure!(compiler.is_file(), "{name} does not contain bin/nvcc");
            return Ok(compiler);
        }
    }
    Ok(PathBuf::from("nvcc"))
}

fn run_compiler(command: &mut Command, timeout: Duration) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = CompilerProcess(
        command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?,
    );
    let stdout = child.0.stdout.take().expect("piped stdout");
    let stderr = child.0.stderr.take().expect("piped stderr");
    let stdout = thread::spawn(move || diagnostic_tail(stdout));
    let stderr = thread::spawn(move || diagnostic_tail(stderr));
    let started = Instant::now();
    let mut timed_out = false;
    let mut status = None;
    let status = loop {
        if status.is_none() {
            status = child.0.try_wait()?;
        }
        // A compiler can exit while a subprocess still owns its diagnostic
        // pipes. Keep that case inside the same deadline as compilation.
        if stdout.is_finished()
            && stderr.is_finished()
            && let Some(status) = status
        {
            break status;
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            child.terminate();
            break child.0.wait()?;
        }
        thread::sleep(PROCESS_POLL_INTERVAL);
    };
    let stdout = stdout
        .join()
        .map_err(|_| anyhow::anyhow!("compiler stdout reader panicked"))??;
    let stderr = stderr
        .join()
        .map_err(|_| anyhow::anyhow!("compiler stderr reader panicked"))??;
    if timed_out || !status.success() {
        bail!(
            "compiler {} after {:.3}s (limit {:.3}s):\nstdout:\n{}\nstderr:\n{}",
            if timed_out {
                "timed out".to_owned()
            } else {
                format!("exited with {status}")
            },
            started.elapsed().as_secs_f64(),
            timeout.as_secs_f64(),
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr),
        );
    }
    Ok(())
}

/// Reap the compiler on every exit path, including diagnostic I/O failures.
struct CompilerProcess(Child);

impl CompilerProcess {
    fn terminate(&mut self) {
        #[cfg(unix)]
        unsafe {
            // This invocation created and exclusively owns the process group.
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        let _ = self.0.kill();
    }
}

impl Drop for CompilerProcess {
    fn drop(&mut self) {
        self.terminate();
        let _ = self.0.wait();
    }
}

fn diagnostic_tail(mut stream: impl Read) -> std::io::Result<Vec<u8>> {
    let mut tail = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Ok(tail);
        }
        let discard = (tail.len() + count).saturating_sub(MAX_DIAGNOSTIC_BYTES);
        tail.drain(..discard);
        tail.extend_from_slice(&buffer[..count]);
    }
}

struct BuildDirectory(PathBuf);

impl Drop for BuildDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/providers/build.rs"]
mod tests;
