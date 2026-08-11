pub mod bf16_gemm;
pub mod oxide_sm90_simt_gemv;
pub mod paged_batch_decode;
pub mod paged_prefill;
pub mod ragged_prefill;
pub mod rms_norm;
pub mod rope;
pub mod single_decode;

pub(super) const VALID_GRAPH_REPLAYS: u64 = 2;
pub(super) const VALID_GRAPH_STAGES: usize = 3;

pub(super) const fn valid_graph_commands(commands_per_stage: usize) -> usize {
    commands_per_stage * VALID_GRAPH_STAGES
}

#[cfg(test)]
mod tests {
    use super::{VALID_GRAPH_REPLAYS, VALID_GRAPH_STAGES, valid_graph_commands};

    #[test]
    fn valid_graph_accounting_covers_observer_poison_and_real_stages() {
        assert_eq!(VALID_GRAPH_REPLAYS, 2);
        assert_eq!(VALID_GRAPH_STAGES, 3);
        assert_eq!(valid_graph_commands(1), 3);
        assert_eq!(valid_graph_commands(2), 6);
        assert_eq!(valid_graph_commands(3), 9);
    }
}
