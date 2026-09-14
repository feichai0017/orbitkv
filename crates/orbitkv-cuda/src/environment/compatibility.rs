use std::fmt;

use super::CudaExecutionEnvironment;

/// Required recovery for a changed artifact environment. Both require a new
/// selected program. This classification explains rejection; it does not
/// automatically recompile, retune or bypass strict replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnvironmentRecovery {
    Recompile,
    Retune,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvironmentChange {
    pub component: String,
    pub recovery: EnvironmentRecovery,
    pub saved: String,
    pub current: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvironmentMismatch {
    pub changes: Vec<EnvironmentChange>,
}

impl fmt::Display for EnvironmentMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA execution environment changed; regenerate the selected artifact"
        )?;
        for change in &self.changes {
            write!(
                f,
                "; {} requires {} (saved {}, current {})",
                change.component,
                match change.recovery {
                    EnvironmentRecovery::Recompile => "recompilation",
                    EnvironmentRecovery::Retune => "retuning",
                },
                change.saved,
                change.current,
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for EnvironmentMismatch {}

impl CudaExecutionEnvironment {
    /// Pure comparison, also usable by deployment tooling without CUDA probing.
    pub fn validate_against(&self, current: &Self) -> Result<(), EnvironmentMismatch> {
        use EnvironmentRecovery::{Recompile, Retune};
        let mut changes = Vec::new();
        compare(
            &mut changes,
            "target",
            Recompile,
            &self.target,
            &current.target,
        );
        compare(
            &mut changes,
            "nvrtc",
            Recompile,
            &self.nvrtc,
            &current.nvrtc,
        );
        compare(
            &mut changes,
            "native_compiler",
            Recompile,
            &self.native_compiler,
            &current.native_compiler,
        );
        compare(
            &mut changes,
            "device",
            Retune,
            &self.device,
            &current.device,
        );
        compare(
            &mut changes,
            "driver_api_version",
            Retune,
            &self.driver_api_version,
            &current.driver_api_version,
        );
        compare(
            &mut changes,
            "provider_inventory",
            Retune,
            &self.provider_lock_digest,
            &current.provider_lock_digest,
        );
        compare(
            &mut changes,
            "cublaslt_autotune",
            Retune,
            &self.cublaslt_autotune,
            &current.cublaslt_autotune,
        );
        let providers = self
            .providers
            .keys()
            .chain(current.providers.keys())
            .collect::<std::collections::BTreeSet<_>>();
        for provider in providers {
            compare(
                &mut changes,
                &format!("provider.{provider}"),
                Recompile,
                &self.providers.get(provider),
                &current.providers.get(provider),
            );
        }
        if changes.is_empty() {
            Ok(())
        } else {
            Err(EnvironmentMismatch { changes })
        }
    }
}

fn compare<T: PartialEq + fmt::Debug>(
    changes: &mut Vec<EnvironmentChange>,
    component: &str,
    recovery: EnvironmentRecovery,
    saved: &T,
    current: &T,
) {
    if saved != current {
        changes.push(EnvironmentChange {
            component: component.into(),
            recovery,
            saved: format!("{saved:?}"),
            current: format!("{current:?}"),
        });
    }
}
