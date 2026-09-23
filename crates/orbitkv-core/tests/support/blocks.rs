use super::SealedBlock;

impl SealedBlock {
    /// Metadata-only page for policy tests; no GPU or storage I/O may use it.
    pub(crate) fn for_policy_test(bytes: u64) -> Self {
        let mut block = Self::from_slots(Vec::new());
        block.footprint = bytes;
        block
    }
}
