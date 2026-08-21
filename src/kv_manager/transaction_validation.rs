use super::{
    BTreeMap, BTreeSet, BackendBindReceipt, BackendCopyReceipt, CanonicalKvManager, KvManagerError,
    PagePhase, PreparedState, RequestLease, RootEntry, StepLease, SubmissionLease, SubmittedState,
    TailActionKind,
};

impl CanonicalKvManager {
    pub(super) fn validate_bind_receipts(
        step: StepLease,
        prepared: &PreparedState,
        receipts: &[BackendBindReceipt],
    ) -> Result<(), KvManagerError> {
        let mut expected = BTreeMap::new();
        for entry in prepared
            .delta
            .classes
            .iter()
            .flat_map(|class| class.tail_destination.iter().chain(class.writes.iter()))
        {
            if expected.insert(entry.page.page_id, entry).is_some() {
                return Err(KvManagerError::DuplicatePage);
            }
        }
        if receipts.len() != expected.len() {
            return Err(KvManagerError::BindingReceiptMismatch);
        }
        let mut seen = BTreeSet::new();
        for receipt in receipts {
            if receipt.reserved != 0 {
                return Err(KvManagerError::ReservedFieldNonZero);
            }
            if receipt.step != step || receipt.mapped != 1 || receipt.writable != 1 {
                return Err(KvManagerError::BindingReceiptMismatch);
            }
            if !seen.insert(receipt.page.page_id) {
                return Err(KvManagerError::DuplicateBindingReceipt);
            }
            let entry = expected
                .get(&receipt.page.page_id)
                .copied()
                .ok_or(KvManagerError::BindingReceiptMismatch)?;
            if receipt.page != entry.page
                || receipt.backend_domain != entry.backend_domain
                || receipt.backend_index != entry.backend_index
            {
                return Err(KvManagerError::BindingReceiptMismatch);
            }
        }
        Ok(())
    }

    pub(super) fn validate_copy_receipts(
        step: StepLease,
        prepared: &PreparedState,
        receipts: &[BackendCopyReceipt],
    ) -> Result<(), KvManagerError> {
        let expected = prepared
            .delta
            .classes
            .iter()
            .filter_map(|class| class.copy_intent)
            .map(|intent| (intent.destination.page_id, intent))
            .collect::<BTreeMap<_, _>>();
        if receipts.len() != expected.len() {
            return Err(KvManagerError::CopyReceiptMismatch);
        }
        let mut seen = BTreeSet::new();
        for receipt in receipts {
            if receipt.reserved8 != 0 || receipt.reserved32 != 0 {
                return Err(KvManagerError::ReservedFieldNonZero);
            }
            if receipt.step != step {
                return Err(KvManagerError::CopyReceiptMismatch);
            }
            if receipt.observed != 1 {
                return Err(KvManagerError::CopyObservationUnknown);
            }
            if receipt.ordered_before_writes != 1 {
                return Err(KvManagerError::CopyOrderingUnknown);
            }
            if receipt.copied != 1 || !seen.insert(receipt.destination.page_id) {
                return Err(KvManagerError::CopyReceiptMismatch);
            }
            let intent = expected
                .get(&receipt.destination.page_id)
                .ok_or(KvManagerError::CopyReceiptMismatch)?;
            if receipt.class_id != intent.class_id
                || receipt.backend_domain != intent.backend_domain
                || receipt.token_count != intent.token_count
                || receipt.source_token_offset != intent.source_token_offset
                || receipt.destination_token_offset != intent.destination_token_offset
                || receipt.source != intent.source
                || receipt.destination != intent.destination
                || receipt.source_backend_index != intent.source_backend_index
                || receipt.destination_backend_index != intent.destination_backend_index
            {
                return Err(KvManagerError::CopyReceiptMismatch);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn preflight_prepared_delta(
        &self,
        prepared: &PreparedState,
        step: StepLease,
    ) -> Result<(), KvManagerError> {
        let mut seen = BTreeSet::new();
        let delta = &prepared.delta;
        self.check_snapshot_epoch(delta.base_snapshot)?;
        self.check_snapshot_epoch(delta.target_snapshot)?;
        let request = self.request(delta.request)?;
        if request.head != delta.base_snapshot {
            return Err(KvManagerError::StaleView);
        }
        let reserved_snapshot = self
            .snapshots
            .get(delta.target_snapshot.slot, delta.target_snapshot.generation)?;
        if reserved_snapshot.boundary != delta.target_boundary
            || reserved_snapshot.view_version != delta.target_view_version
            || !reserved_snapshot.is_empty()
        {
            return Err(KvManagerError::StaleView);
        }
        let snapshot = self.request_snapshot(delta.request)?;
        if delta.classes.len() != self.classes.len() || snapshot.roots.len() != self.classes.len() {
            return Err(KvManagerError::StaleView);
        }
        let first_new = delta.previous_boundary.div_ceil(self.page_tokens);
        let new_end = delta.target_boundary.div_ceil(self.page_tokens);
        let joint_cow = delta
            .classes
            .iter()
            .any(|class| class.tail_action == TailActionKind::CopyOnWrite);
        for (((class, class_delta), root), class_index) in self
            .classes
            .iter()
            .copied()
            .zip(delta.classes.iter())
            .zip(snapshot.roots.iter())
            .zip(0_usize..)
        {
            if class_delta.class_id != class.class_id
                || class_index != usize::from(class.class_id)
                || class_delta.writes.len()
                    != usize::try_from(new_end - first_new)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("class write count"))?
            {
                return Err(KvManagerError::Invariant("delta class shape"));
            }
            let expected_tail = if delta.previous_boundary.is_multiple_of(self.page_tokens) {
                None
            } else {
                root.entries.back().copied()
            };
            if class_delta.tail_source != expected_tail
                || class_delta.tail_source.is_some_and(|entry| {
                    entry.logical_ordinal != delta.previous_boundary / self.page_tokens
                        || entry.class_id != class.class_id
                })
            {
                return Err(KvManagerError::StaleView);
            }
            let expected_action = if delta.previous_boundary.is_multiple_of(self.page_tokens) {
                TailActionKind::None
            } else if expected_tail.is_none() {
                TailActionKind::Fresh
            } else if joint_cow {
                TailActionKind::CopyOnWrite
            } else {
                TailActionKind::InPlace
            };
            if class_delta.tail_action != expected_action {
                return Err(KvManagerError::Invariant("joint tail action"));
            }
            if let Some(entry) = class_delta.tail_source {
                if !seen.insert(entry.page.page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation
                    || page.request_refs == 0
                    || page.reader_pins != 0
                    || page.writer.is_some()
                    || page.phase != PagePhase::Live
                {
                    return Err(KvManagerError::StalePage);
                }
                if page.reader_pins == u32::MAX {
                    return Err(KvManagerError::ReaderCountOverflow(entry.page.page_id));
                }
                if class_delta.tail_action == TailActionKind::InPlace
                    && (page.request_refs != 1 || page.prefix_refs != 0)
                {
                    return Err(KvManagerError::StalePage);
                }
            }
            if let Some(entry) = class_delta.tail_destination {
                if !seen.insert(entry.page.page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let expected_ordinal = delta.previous_boundary / self.page_tokens;
                if self.root_entry_for_page(class, expected_ordinal, entry.page)? != entry {
                    return Err(KvManagerError::StaleView);
                }
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation
                    || page.request_refs != 0
                    || page.prefix_refs != 0
                    || page.reader_pins != 0
                    || page.writer.is_some()
                    || page.phase != (PagePhase::Reserved { step })
                {
                    return Err(KvManagerError::StalePage);
                }
                if class_delta.tail_action == TailActionKind::CopyOnWrite {
                    let intent = class_delta
                        .copy_intent
                        .ok_or(KvManagerError::Invariant("missing copy intent"))?;
                    if Some(intent.source) != class_delta.tail_source.map(|source| source.page)
                        || intent.destination != entry.page
                        || intent.class_id != class.class_id
                    {
                        return Err(KvManagerError::Invariant("copy intent shape"));
                    }
                } else if class_delta.copy_intent.is_some() {
                    return Err(KvManagerError::Invariant("unexpected copy intent"));
                }
            } else if class_delta.copy_intent.is_some()
                || matches!(
                    class_delta.tail_action,
                    TailActionKind::CopyOnWrite | TailActionKind::Fresh
                )
            {
                return Err(KvManagerError::Invariant("missing tail destination"));
            }
            for (offset, entry) in class_delta.writes.iter().enumerate() {
                if !seen.insert(entry.page.page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let expected_ordinal = first_new
                    .checked_add(offset as u64)
                    .ok_or(KvManagerError::ArithmeticOverflow("write ordinal"))?;
                if self.root_entry_for_page(class, expected_ordinal, entry.page)? != *entry {
                    return Err(KvManagerError::StaleView);
                }
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation
                    || page.request_refs != 0
                    || page.prefix_refs != 0
                    || page.reader_pins != 0
                    || page.writer.is_some()
                    || page.phase != (PagePhase::Reserved { step })
                {
                    return Err(KvManagerError::StalePage);
                }
            }
        }
        Ok(())
    }

    pub(super) fn preflight_submitted_delta(
        &self,
        submission: SubmissionLease,
        submitted: &SubmittedState,
    ) -> Result<(), KvManagerError> {
        let delta = &submitted.delta;
        self.check_snapshot_epoch(delta.base_snapshot)?;
        self.check_snapshot_epoch(delta.target_snapshot)?;
        if self.request(delta.request)?.head != delta.base_snapshot {
            return Err(KvManagerError::StaleView);
        }
        let reserved_snapshot = self
            .snapshots
            .get(delta.target_snapshot.slot, delta.target_snapshot.generation)?;
        if reserved_snapshot.boundary != delta.target_boundary
            || reserved_snapshot.view_version != delta.target_view_version
            || !reserved_snapshot.is_empty()
        {
            return Err(KvManagerError::StaleView);
        }
        if delta.classes.len() != self.classes.len() {
            return Err(KvManagerError::StaleView);
        }
        for (class, class_delta) in self.classes.iter().copied().zip(delta.classes.iter()) {
            if class_delta.class_id != class.class_id {
                return Err(KvManagerError::Invariant("delta class ordering"));
            }
            if let Some(entry) = class_delta.tail_source {
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                let witness_valid = match class_delta.tail_action {
                    TailActionKind::InPlace => {
                        page.reader_pins == 1 && page.writer == Some(submission)
                    }
                    TailActionKind::CopyOnWrite => page.reader_pins != 0 && page.writer.is_none(),
                    TailActionKind::None | TailActionKind::Fresh => false,
                };
                if page.generation != entry.page.generation
                    || page.request_refs == 0
                    || !witness_valid
                    || page.phase != PagePhase::Live
                {
                    return Err(KvManagerError::StalePage);
                }
            }
            if let Some(entry) = class_delta.tail_destination {
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation
                    || page.request_refs != 0
                    || page.prefix_refs != 0
                    || page.reader_pins != 1
                    || page.writer != Some(submission)
                    || page.phase != PagePhase::Live
                {
                    return Err(KvManagerError::StalePage);
                }
            }
            for entry in class_delta.writes.iter() {
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation
                    || page.request_refs != 0
                    || page.prefix_refs != 0
                    || page.reader_pins != 1
                    || page.writer != Some(submission)
                    || page.phase != PagePhase::Live
                {
                    return Err(KvManagerError::StalePage);
                }
            }
        }
        Ok(())
    }

    pub(super) fn preflight_active_root_entry(
        &self,
        _request: RequestLease,
        entry: RootEntry,
    ) -> Result<(), KvManagerError> {
        let class = self.runtime_class(entry.class_id)?;
        self.validate_page_lease(class, entry.page)?;
        let page = self.page(entry.page.page_id)?;
        if page.generation != entry.page.generation
            || page.request_refs == 0
            || page.reader_pins != 0
            || page.writer.is_some()
            || page.phase != PagePhase::Live
        {
            return Err(KvManagerError::StalePage);
        }
        Ok(())
    }
}
