use super::CudaRuntimeImpl;
use crate::environment::CudaExecutionEnvironment;

impl<O> CudaRuntimeImpl<O> {
    /// Include every retained bucket and nested CUDA Graph, not just the active
    /// bucket. Library dependencies are contracts supplied by the selected ops.
    pub fn execution_environment(&self) -> anyhow::Result<CudaExecutionEnvironment> {
        anyhow::ensure!(
            !self.compiled_buckets.is_empty(),
            "no selected CUDA program"
        );
        let dependencies = self.compiled_buckets.iter().flat_map(|bucket| {
            bucket
                .exec_graph
                .node_weights()
                .flat_map(|op| op.internal.provider_dependencies())
        });
        CudaExecutionEnvironment::capture(self.cuda_stream.context(), dependencies)
    }
}
