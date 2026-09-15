//! The actual execution device, carried explicitly into compilation.

use cudarc::driver::{CudaContext, DriverError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct CudaTarget {
    pub major: i32,
    pub minor: i32,
}

impl CudaTarget {
    /// Query the context that owns the execution stream.
    pub fn from_context(context: &CudaContext) -> Result<Self, DriverError> {
        let (major, minor) = context.compute_capability()?;
        Ok(Self { major, minor })
    }

    pub fn architecture(self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }

    pub fn compiler_facts(self) -> String {
        format!(
            "(set (cuda-target-major) {})\n(set (cuda-target-minor) {})",
            self.major, self.minor
        )
    }

    /// Full device facts for shape-dependent kernel choices. Architecture alone
    /// does not describe occupancy; no SM count is inferred from a GPU name.
    pub fn compiler_facts_from_context(context: &CudaContext) -> Result<String, DriverError> {
        let target = Self::from_context(context)?;
        let sm_count = context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )?;
        Ok(format!(
            "{}\n(set (cuda-target-sm-count) {sm_count})",
            target.compiler_facts()
        ))
    }

    pub(crate) fn hopper_architecture(self) -> anyhow::Result<&'static str> {
        anyhow::ensure!(
            self == Self { major: 9, minor: 0 },
            "this provider adapter requires Hopper; execution target is {}",
            self.architecture()
        );
        Ok("sm_90a")
    }
}

pub(crate) const DECLARATIONS: &str = include_str!("target.egg");

/// A declaration carrier, with no executable graph node of its own.
#[derive(Debug, Default)]
pub struct CudaTargetFacts;

impl orbitkv_compiler::op::EgglogOp for CudaTargetFacts {
    fn cleanup(&self) -> bool {
        false
    }

    fn n_inputs(&self) -> usize {
        0
    }
    fn sort(&self) -> orbitkv_compiler::egglog_utils::api::SortDef {
        orbitkv_compiler::egglog_utils::api::sort(
            orbitkv_compiler::egglog_utils::base::OP_KIND,
            "CudaTargetFacts",
            &[],
        )
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![DECLARATIONS.to_owned()]
    }
}

#[cfg(test)]
#[path = "../tests/unit/target.rs"]
mod tests;
