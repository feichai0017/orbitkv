// Allow files path-included into this crate's tests (e.g. the llama example
// model in search_equivalence_fuzz.rs) to keep their normal
// `use orbitkv_cuda::...` imports.
extern crate self as orbitkv_cuda;

mod artifact;
pub use artifact::CudaModuleArtifact;
mod compilation;
pub mod dyn_backend;
pub mod environment;
pub mod kernel;
pub mod providers;
mod resource;
pub mod runtime;
mod search;
pub mod target;
pub(crate) use compilation::with_kernel_source_limit;
pub use compilation::{
    CudaModuleImageCompileError, CudaModuleImageCompileFailure,
    compile_module_image_for_current_device, cuda_dtype,
};
pub use cudarc;

#[cfg(test)]
#[path = "../tests/unit/mod.rs"]
mod tests;
