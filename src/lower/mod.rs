//! Verified lowering boundary to the external `kern` execution substrate.

mod deepgemm_projection;
mod kern;
mod qwen38_skeleton;
mod qwen38_weights;

pub use deepgemm_projection::lower_deepgemm_projection_probe;
pub use kern::{KernArtifact, LoweringError};
pub use qwen38_skeleton::{ProgramCall, ProgramSkeleton, Qwen38Declarations, Qwen38ProgramSkeletons};
pub use qwen38_weights::{Qwen38Fp8WeightPlan, WeightPlanError, WeightTransform, WeightTransformKind};
