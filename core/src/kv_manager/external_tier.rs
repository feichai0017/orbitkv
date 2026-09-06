use std::collections::BTreeSet;

use super::{
    CanonicalKvManager, KvManagerError, PageLease, PagePhase, RequestLease, SnapshotLease,
};

/// One immutable, generation-checked page pinned for an external read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PinnedSnapshotPage {
    pub page: PageLease,
    pub class_id: u16,
    pub backend_domain: u16,
    pub backend_index: u64,
    pub logical_ordinal: u64,
    pub valid_token_count: u32,
    pub visible_token_offset: u32,
    pub visible_token_count: u32,
    pub payload_bytes: u64,
}

impl CanonicalKvManager {
    pub(crate) const fn plan_fingerprint(&self) -> [u8; 32] {
        self.plan_fingerprint
    }

    pub(crate) fn external_payload_bytes(
        &self,
        class_id: u16,
        valid_token_count: u32,
    ) -> Result<u64, KvManagerError> {
        if valid_token_count == 0 || u64::from(valid_token_count) > self.page_tokens {
            return Err(KvManagerError::InvalidBatchRange);
        }
        self.runtime_class(class_id)?
            .page_payload_bytes
            .checked_mul(u64::from(valid_token_count))
            .and_then(|bytes| bytes.checked_div(self.page_tokens))
            .ok_or(KvManagerError::ArithmeticOverflow(
                "external page payload bytes",
            ))
    }

    pub(crate) fn validate_external_completion(
        &self,
        completion_domain: u64,
        completion_value: u64,
    ) -> Result<(), KvManagerError> {
        self.validate_completion_frontier(completion_domain, completion_value)
    }

    pub(crate) fn commit_external_completion(
        &mut self,
        completion_domain: u64,
        completion_value: u64,
    ) {
        self.commit_completion_frontier(completion_domain, completion_value);
    }

    /// Pins every page in one immutable request snapshot for an external read.
    ///
    /// The complete page set is validated before any pin changes. The caller
    /// must later release exactly this returned set after durable completion,
    /// or retain the pins when transfer completion is ambiguous.
    pub(crate) fn pin_snapshot_for_external_read(
        &mut self,
        request: RequestLease,
        expected: SnapshotLease,
    ) -> Result<Box<[PinnedSnapshotPage]>, KvManagerError> {
        if self
            .classes
            .iter()
            .copied()
            .any(|class| !class.uses_compiled_residence())
        {
            return Err(KvManagerError::UnsupportedProfile(
                "external export requires compiled physical residence",
            ));
        }
        let state = self.request(request)?;
        if state.released || state.quarantined {
            return Err(KvManagerError::RequestUnavailable);
        }
        if state.head != expected {
            return Err(KvManagerError::StaleView);
        }
        let snapshot = self.request_snapshot(request)?;
        let pages = self.materialize_snapshot_roots(snapshot.boundary, &snapshot.roots)?;
        if pages.is_empty() {
            return Err(KvManagerError::InvalidBatchRange);
        }

        let mut seen = BTreeSet::new();
        let mut pinned = Vec::with_capacity(pages.len());
        for page in &pages {
            if !seen.insert(page.page) {
                return Err(KvManagerError::DuplicatePage);
            }
            let class = self.runtime_class(page.class_id)?;
            self.validate_page_lease(class, page.page)?;
            let state = self.page(page.page.page_id)?;
            if state.generation != page.page.generation
                || state.phase != PagePhase::Live
                || state.writer.is_some()
                || state.request_refs == 0
            {
                return Err(KvManagerError::StalePage);
            }
            if state.reader_pins == u32::MAX {
                return Err(KvManagerError::ReaderCountOverflow(page.page.page_id));
            }
            let payload_bytes =
                self.external_payload_bytes(page.class_id, page.valid_token_count)?;
            pinned.push(PinnedSnapshotPage {
                page: page.page,
                class_id: page.class_id,
                backend_domain: page.backend_domain,
                backend_index: page.backend_index,
                logical_ordinal: page.logical_ordinal,
                valid_token_count: page.valid_token_count,
                visible_token_offset: page.visible_token_offset,
                visible_token_count: page.visible_token_count,
                payload_bytes,
            });
        }
        for page in &pinned {
            self.page_mut(page.page.page_id)?.reader_pins += 1;
        }
        Ok(pinned.into_boxed_slice())
    }

    pub(crate) fn validate_external_read_pins(
        &self,
        pages: &[PinnedSnapshotPage],
    ) -> Result<(), KvManagerError> {
        let mut seen = BTreeSet::new();
        for page in pages {
            if !seen.insert(page.page) {
                return Err(KvManagerError::DuplicatePage);
            }
            let class = self.runtime_class(page.class_id)?;
            self.validate_page_lease(class, page.page)?;
            let state = self.page(page.page.page_id)?;
            if state.generation != page.page.generation
                || state.phase != PagePhase::Live
                || state.reader_pins == 0
            {
                return Err(KvManagerError::StalePage);
            }
        }
        Ok(())
    }

    /// Releases one already validated external-read pin set.
    pub(crate) fn commit_external_read_pin_release(&mut self, pages: &[PinnedSnapshotPage]) {
        for page in pages {
            let mut state = self
                .page_mut(page.page.page_id)
                .expect("validated external pin page remains present");
            debug_assert_eq!(state.generation, page.page.generation);
            debug_assert!(state.reader_pins > 0);
            state.reader_pins -= 1;
        }
    }

    pub(crate) fn release_external_read_pins(
        &mut self,
        pages: &[PinnedSnapshotPage],
    ) -> Result<(), KvManagerError> {
        self.validate_external_read_pins(pages)?;
        self.commit_external_read_pin_release(pages);
        Ok(())
    }
}
