use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::plan::RetentionKind;

use super::persistent_snapshot::{ClassRoot, RootLayout};
use super::relocation_policy::RelocationPlanningContext;
use super::token_virtualization::plan_token_relocation;
use super::{
    BatchCompletionReceipt, CanonicalKvManager, CompletedRelocationBatch, KvManagerError,
    PagePhase, PersistentRootEntries, PrepareRelocationItem, PreparedRelocation, ReclamationState,
    RelocationCopyReceipt, RelocationDestination, RelocationLease, RelocationPageState,
    RelocationPlan, RelocationUnobservedReceipt, RequestSnapshot, SnapshotLease,
    SubmittedRelocation, TokenView, ViewVersion, apply_token_relocation, validate_token_view,
};

#[derive(Clone, Debug)]
pub(super) struct RelocationState {
    request: super::RequestLease,
    base_snapshot: SnapshotLease,
    target_snapshot: SnapshotLease,
    plan: RelocationPlan,
    submitted: bool,
}

impl CanonicalKvManager {
    fn validate_relocation_source_page(
        &self,
        class_id: u16,
        source: super::PageLease,
    ) -> Result<(), KvManagerError> {
        let class = self.runtime_class(class_id)?;
        self.validate_page_lease(class, source)?;
        let page = self.page(source.page_id)?;
        if page.generation != source.generation
            || page.phase != PagePhase::Live
            || page.request_refs != 1
            || page.prefix_refs != 0
            || page.reader_pins != 0
            || page.writer.is_some()
        {
            return Err(KvManagerError::StalePage);
        }
        Ok(())
    }

    fn validate_reserved_relocation_destination(
        &self,
        class_id: u16,
        relocation: RelocationLease,
        destination: super::PageLease,
    ) -> Result<(), KvManagerError> {
        let class = self.runtime_class(class_id)?;
        self.validate_page_lease(class, destination)?;
        let page = self.page(destination.page_id)?;
        if page.generation != destination.generation
            || page.phase != (PagePhase::ReservedRelocation { relocation })
            || page.request_refs != 0
            || page.prefix_refs != 0
            || page.reader_pins != 0
            || page.writer.is_some()
        {
            return Err(KvManagerError::StalePage);
        }
        Ok(())
    }

    /// Validates the exact correspondence between a Full-class root, its
    /// absolute token history, and its current physical layout.
    ///
    /// A source root may contain dead tokens that still occupy physical slots:
    /// semantic death alone does not release storage. A freshly compacted
    /// target is stricter and must contain only retained physical placements.
    fn validate_relocation_root(
        &self,
        class_id: u16,
        snapshot_boundary: u64,
        view_version: ViewVersion,
        root: &ClassRoot,
        require_compacted: bool,
    ) -> Result<TokenView, KvManagerError> {
        if root.tokens.len() != snapshot_boundary {
            return Err(KvManagerError::Invariant("token table boundary"));
        }
        let class = self.runtime_class(class_id)?;
        let page_tokens = u32::try_from(self.page_tokens)
            .map_err(|_| KvManagerError::ArithmeticOverflow("page tokens"))?;
        let view = TokenView {
            class_id,
            version: view_version,
            page_tokens,
            placements: root.tokens.materialize()?,
        };
        validate_token_view(&view)?;
        for (token_id, placement) in view.placements.iter().enumerate() {
            if placement.token_id
                != u64::try_from(token_id)
                    .map_err(|_| KvManagerError::ArithmeticOverflow("token id"))?
            {
                return Err(KvManagerError::InvalidTokenView);
            }
        }

        let physical_boundary = root.mirror_boundary(snapshot_boundary);
        if root.is_dense() && root.resident_tokens != snapshot_boundary {
            return Err(KvManagerError::Invariant("dense resident token boundary"));
        }
        let expected_entries = usize::try_from(physical_boundary.div_ceil(self.page_tokens))
            .map_err(|_| KvManagerError::ArithmeticOverflow("root entry count"))?;
        if root.entries.len() != expected_entries {
            return Err(KvManagerError::Invariant("root physical span"));
        }

        let mut entries = BTreeMap::new();
        for (ordinal, entry) in root.entries.iter().copied().enumerate() {
            let ordinal = u64::try_from(ordinal)
                .map_err(|_| KvManagerError::ArithmeticOverflow("root ordinal"))?;
            if self.root_entry_for_page(class, ordinal, entry.page)? != entry
                || entries
                    .insert(entry.page, (ordinal, entry.backend_index))
                    .is_some()
            {
                return Err(KvManagerError::Invariant("root physical identity"));
            }
        }

        let mut occupied = BTreeSet::new();
        let mut retained = 0_u64;
        for placement in &view.placements {
            retained = retained
                .checked_add(u64::from(placement.disposition.retained()))
                .ok_or(KvManagerError::ArithmeticOverflow("retained tokens"))?;
            let Some(location) = placement.location else {
                if root.is_dense() {
                    return Err(KvManagerError::TokenPlacementMismatch);
                }
                continue;
            };
            if require_compacted && !placement.disposition.retained() {
                return Err(KvManagerError::TokenPlacementMismatch);
            }
            let (ordinal, backend_index) = entries
                .get(&location.page)
                .copied()
                .ok_or(KvManagerError::TokenPlacementMismatch)?;
            if backend_index != location.backend_index {
                return Err(KvManagerError::TokenPlacementMismatch);
            }
            let physical_slot = ordinal
                .checked_mul(self.page_tokens)
                .and_then(|begin| begin.checked_add(u64::from(location.offset)))
                .ok_or(KvManagerError::ArithmeticOverflow("physical token slot"))?;
            if physical_slot >= physical_boundary
                || root.is_dense() && physical_slot != placement.token_id
                || !occupied.insert(physical_slot)
            {
                return Err(KvManagerError::TokenPlacementMismatch);
            }
        }
        if u64::try_from(occupied.len())
            .map_err(|_| KvManagerError::ArithmeticOverflow("occupied token slots"))?
            != physical_boundary
            || !occupied.iter().copied().eq(0..physical_boundary)
            || require_compacted && retained != physical_boundary
        {
            return Err(KvManagerError::TokenPlacementMismatch);
        }
        Ok(view)
    }

    /// Atomically plans and reserves one full-evacuation relocation per request.
    ///
    /// # Errors
    ///
    /// Rejects stale/busy/shared/Prefix/pinned requests, non-Full classes,
    /// non-profitable plans, insufficient bounded headroom, or arena exhaustion
    /// before changing any request or page state.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive free-list state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn prepare_relocation_batch(
        &mut self,
        items: &[PrepareRelocationItem],
    ) -> Result<Box<[PreparedRelocation]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let planned_operations = self.relocations.plan_many(items.len())?;
        let planned_snapshots = self.snapshots.plan_many(items.len())?;
        let mut seen_requests = BTreeSet::new();
        let mut page_cursors = vec![0_usize; self.classes.len()];
        let mut plans = Vec::with_capacity(items.len());
        for ((item, operation_slot), snapshot_slot) in items
            .iter()
            .zip(planned_operations.iter().copied())
            .zip(planned_snapshots.iter().copied())
        {
            if !seen_requests.insert(item.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let state = self.request(item.request)?;
            if state.released || state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if state.head != item.expected_snapshot {
                return Err(KvManagerError::StaleTokenView);
            }
            if state.busy() {
                return Err(KvManagerError::RequestBusy);
            }
            let class = self.runtime_class(item.class_id)?;
            if class.retention != RetentionKind::Full
                || !class.uses_compiled_residence()
                || !item.policy.full_evacuation
            {
                return Err(KvManagerError::UnsupportedProfile(
                    "relocation requires compiled Full-class full evacuation",
                ));
            }
            let snapshot = self.request_snapshot(item.request)?;
            let root = snapshot
                .roots
                .get(usize::from(item.class_id))
                .ok_or(KvManagerError::InvalidClass(item.class_id))?;
            let view = self.validate_relocation_root(
                item.class_id,
                snapshot.boundary,
                snapshot.view_version,
                root,
                false,
            )?;
            let mut page_states = Vec::with_capacity(root.entries.len());
            for entry in root.entries.iter().copied() {
                self.validate_page_lease(class, entry.page)?;
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation || page.phase != PagePhase::Live {
                    return Err(KvManagerError::StalePage);
                }
                page_states.push(RelocationPageState {
                    page: entry.page,
                    backend_index: entry.backend_index,
                    request_refs: page.request_refs,
                    prefix_refs: page.prefix_refs,
                    reader_pins: page.reader_pins,
                    writer_present: page.writer.is_some(),
                });
            }
            let retained = view
                .placements
                .iter()
                .filter(|placement| placement.disposition.retained())
                .count();
            let page_tokens = usize::try_from(self.page_tokens)
                .map_err(|_| KvManagerError::ArithmeticOverflow("page tokens"))?;
            let destination_count = retained.div_ceil(page_tokens);
            let mut destination_leases = Vec::with_capacity(destination_count);
            let mut destinations = Vec::with_capacity(destination_count);
            for ordinal in 0..destination_count {
                let page = self.plan_free_page(class, &mut page_cursors)?;
                let root = self.root_entry_for_page(
                    class,
                    u64::try_from(ordinal)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("packed ordinal"))?,
                    page,
                )?;
                destination_leases.push(page);
                destinations.push(RelocationDestination {
                    page,
                    backend_index: root.backend_index,
                });
            }
            let plan = plan_token_relocation(
                &view,
                &page_states,
                &destinations,
                RelocationPlanningContext {
                    plan_fingerprint: self.plan_fingerprint,
                    page_tokens: u32::try_from(self.page_tokens).map_err(|_| {
                        KvManagerError::ArithmeticOverflow("relocation page tokens")
                    })?,
                    page_payload_bytes: class.page_payload_bytes,
                },
                &item.policy,
            )?
            .ok_or(KvManagerError::InvalidRelocationPlan)?;
            if plan.source_pages.len() != root.entries.len()
                || plan.destination_pages.len() != destination_count
            {
                return Err(KvManagerError::InvalidRelocationPlan);
            }
            let relocation = RelocationLease {
                engine_epoch: self.engine_epoch,
                slot: operation_slot.0,
                generation: operation_slot.1,
            };
            let target_snapshot = SnapshotLease {
                engine_epoch: self.engine_epoch,
                slot: snapshot_slot.0,
                generation: snapshot_slot.1,
            };
            let state = RelocationState {
                request: item.request,
                base_snapshot: item.expected_snapshot,
                target_snapshot,
                plan: plan.clone(),
                submitted: false,
            };
            plans.push((
                operation_slot,
                snapshot_slot,
                relocation,
                destination_leases,
                state,
                PreparedRelocation {
                    relocation,
                    request: item.request,
                    base_snapshot: item.expected_snapshot,
                    target_snapshot,
                    plan,
                },
            ));
        }
        for (_, _, relocation, pages, _, _) in &plans {
            for lease in pages {
                let class_id = self.page(lease.page_id)?.class_id;
                let popped = self.free_pages[usize::from(class_id)].pop();
                assert_eq!(popped, Some(lease.page_id));
                {
                    let mut page = self.page_mut(lease.page_id)?;
                    page.generation = lease.generation;
                }
                self.set_page_phase(
                    lease.page_id,
                    PagePhase::ReservedRelocation {
                        relocation: *relocation,
                    },
                )?;
            }
        }
        for (operation_slot, snapshot_slot, relocation, _, state, _) in &plans {
            self.snapshots.insert_planned(
                *snapshot_slot,
                RequestSnapshot {
                    boundary: self.request_snapshot(state.request)?.boundary,
                    view_version: state.plan.target_version,
                    roots: Arc::from([]),
                },
            );
            self.relocations
                .insert_planned(*operation_slot, state.clone());
            self.request_mut(state.request)?.pending_relocation = Some(*relocation);
        }
        Ok(plans
            .into_iter()
            .map(|(_, _, _, _, _, output)| output)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Validates exact token-copy receipts and transitions destinations to Relocating.
    ///
    /// # Errors
    ///
    /// Any stale, duplicate, unobserved, uncopied, or mismatched receipt rejects
    /// the operation. Semantic uncertainty quarantines every destination.
    ///
    /// # Panics
    ///
    /// Panics only if manager-owned relocation state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn submit_relocation_batch(
        &mut self,
        relocations: &[RelocationLease],
        receipts: &[RelocationCopyReceipt],
    ) -> Result<Box<[SubmittedRelocation]>, KvManagerError> {
        if relocations.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(relocations.len());
        let mut receipt_offset = 0_usize;
        for &relocation in relocations {
            if relocation.engine_epoch != self.engine_epoch {
                return Err(KvManagerError::WrongEngine);
            }
            if !seen.insert(relocation) {
                return Err(KvManagerError::DuplicateStep);
            }
            let state = self
                .relocations
                .get(relocation.slot, relocation.generation)?
                .clone();
            if state.submitted {
                return Err(KvManagerError::StepAlreadySubmitted);
            }
            if !seen_requests.insert(state.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(state.request)?;
            if request.head != state.base_snapshot || request.pending_relocation != Some(relocation)
            {
                return Err(KvManagerError::StaleTokenView);
            }
            self.snapshots
                .get(state.target_snapshot.slot, state.target_snapshot.generation)?;
            let end = receipt_offset
                .checked_add(state.plan.moves.len())
                .ok_or(KvManagerError::InvalidBatchRange)?;
            receipts
                .get(receipt_offset..end)
                .ok_or(KvManagerError::InvalidBatchRange)?;
            if state
                .plan
                .source_pages
                .iter()
                .chain(state.plan.destination_pages.iter())
                .any(|page| !seen_pages.insert(page.page_id))
            {
                return Err(KvManagerError::DuplicatePage);
            }
            for &source in &state.plan.source_pages {
                self.validate_relocation_source_page(state.plan.class_id, source)?;
            }
            for &destination in &state.plan.destination_pages {
                self.validate_reserved_relocation_destination(
                    state.plan.class_id,
                    relocation,
                    destination,
                )?;
            }
            plans.push((relocation, state, receipt_offset, end));
            receipt_offset = end;
        }
        if receipt_offset != receipts.len() {
            return Err(KvManagerError::InvalidBatchRange);
        }
        let semantic_result = (|| {
            for (relocation, state, begin, end) in &plans {
                for (receipt, movement) in
                    receipts[*begin..*end].iter().zip(state.plan.moves.iter())
                {
                    if receipt.reserved16 != 0 || receipt.reserved32 != 0 {
                        return Err(KvManagerError::ReservedFieldNonZero);
                    }
                    if receipt.observed != 1 {
                        return Err(KvManagerError::CopyObservationUnknown);
                    }
                    if receipt.copied != 1
                        || receipt.relocation != *relocation
                        || receipt.token_id != movement.token_id
                        || receipt.source != movement.source
                        || receipt.destination != movement.destination
                    {
                        return Err(KvManagerError::CopyReceiptMismatch);
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = semantic_result {
            for (relocation, state, _, _) in &plans {
                for destination in &state.plan.destination_pages {
                    self.set_page_phase(destination.page_id, PagePhase::Quarantined)
                        .expect("relocation submit preflight retained destination");
                }
                self.snapshots
                    .remove(state.target_snapshot.slot, state.target_snapshot.generation)
                    .expect("relocation submit preflight retained target snapshot");
                self.relocations
                    .remove(relocation.slot, relocation.generation)
                    .expect("relocation submit preflight retained relocation");
                let request = self
                    .request_mut(state.request)
                    .expect("relocation submit preflight retained request");
                request.pending_relocation = None;
                request.quarantined = true;
            }
            return Err(KvManagerError::BatchQuarantined(Box::new(error)));
        }
        for (relocation, state, _, _) in &plans {
            for source in &state.plan.source_pages {
                self.page_mut(source.page_id)
                    .expect("relocation submit preflight retained source")
                    .reader_pins += 1;
            }
            for destination in &state.plan.destination_pages {
                self.set_page_phase(
                    destination.page_id,
                    PagePhase::Relocating {
                        relocation: *relocation,
                    },
                )
                .expect("relocation submit preflight retained destination");
            }
            self.relocations
                .get_mut(relocation.slot, relocation.generation)
                .expect("relocation submit preflight retained relocation")
                .submitted = true;
        }
        Ok(plans
            .into_iter()
            .map(|(relocation, state, _, _)| SubmittedRelocation {
                relocation,
                request: state.request,
                target_snapshot: state.target_snapshot,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Publishes packed token views after one confirmed CUDA completion point.
    ///
    /// # Errors
    ///
    /// Rejects unconfirmed or stale completion, malformed plans, unavailable
    /// reclamation capacity, or changed page generations before mutation.
    #[allow(clippy::too_many_lines)]
    pub fn complete_relocation_batch(
        &mut self,
        receipt: BatchCompletionReceipt,
        relocations: &[RelocationLease],
    ) -> Result<CompletedRelocationBatch, KvManagerError> {
        if receipt.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        if receipt.confirmed != 1 || receipt.reserved != 0 {
            return Err(KvManagerError::CompletionNotConfirmed);
        }
        if relocations.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        self.validate_completion_frontier(receipt.completion_domain, receipt.completion_value)?;
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(relocations.len());
        let mut retiring = Vec::new();
        for &relocation in relocations {
            let state = self
                .relocations
                .get(relocation.slot, relocation.generation)?
                .clone();
            if !state.submitted || !seen_requests.insert(state.request) {
                return Err(KvManagerError::StepNotSubmitted);
            }
            let request = self.request(state.request)?;
            if request.head != state.base_snapshot || request.pending_relocation != Some(relocation)
            {
                return Err(KvManagerError::StaleTokenView);
            }
            let snapshot = self.request_snapshot(state.request)?;
            let class = self.runtime_class(state.plan.class_id)?;
            let root = &snapshot.roots[usize::from(state.plan.class_id)];
            let view = self.validate_relocation_root(
                state.plan.class_id,
                snapshot.boundary,
                snapshot.view_version,
                root,
                false,
            )?;
            let source_layout_boundary = root.mirror_boundary(snapshot.boundary);
            let target_view = apply_token_relocation(&view, &state.plan)?;
            let mut target_roots = snapshot.roots.iter().cloned().collect::<Vec<_>>();
            let target_root = &mut target_roots[usize::from(state.plan.class_id)];
            let mut entries = PersistentRootEntries::default();
            for (ordinal, destination) in state.plan.destination_pages.iter().copied().enumerate() {
                entries.push_back(
                    self.root_entry_for_page(
                        class,
                        u64::try_from(ordinal)
                            .map_err(|_| KvManagerError::ArithmeticOverflow("packed ordinal"))?,
                        destination,
                    )?,
                );
            }
            target_root.entries = entries;
            target_root.tokens =
                super::PersistentTokenTable::from_materialized(&target_view.placements)?;
            target_root.layout = RootLayout::Packed;
            target_root.resident_tokens = u64::try_from(
                target_view
                    .placements
                    .iter()
                    .filter(|placement| placement.disposition.retained())
                    .count(),
            )
            .map_err(|_| KvManagerError::ArithmeticOverflow("resident tokens"))?;
            self.validate_relocation_root(
                state.plan.class_id,
                snapshot.boundary,
                state.plan.target_version,
                target_root,
                true,
            )?;
            let mut retiring_for_request = Vec::with_capacity(state.plan.source_pages.len());
            for source in &state.plan.source_pages {
                if !seen_pages.insert(*source) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(source.page_id)?;
                if page.generation != source.generation
                    || page.phase != PagePhase::Live
                    || page.reader_pins != 1
                    || page.request_refs != 1
                    || page.prefix_refs != 0
                    || page.writer.is_some()
                {
                    return Err(KvManagerError::StalePage);
                }
                let entry = root
                    .entries
                    .iter()
                    .copied()
                    .find(|entry| entry.page == *source)
                    .ok_or(KvManagerError::InvalidRelocationPlan)?;
                retiring_for_request.push((entry, source_layout_boundary));
            }
            // The planner orders evacuation candidates by occupancy, which is
            // intentionally different from the canonical root order after a
            // packed append.  Keep each request's certificates contiguous and
            // publish them in physical logical-ordinal order.
            retiring_for_request
                .sort_unstable_by_key(|(entry, _)| (entry.logical_ordinal, entry.page));
            retiring.extend(retiring_for_request);
            for destination in &state.plan.destination_pages {
                if !seen_pages.insert(*destination) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(destination.page_id)?;
                if page.generation != destination.generation
                    || page.phase != (PagePhase::Relocating { relocation })
                    || page.reader_pins != 0
                    || page.request_refs != 0
                    || page.prefix_refs != 0
                    || page.writer.is_some()
                {
                    return Err(KvManagerError::StalePage);
                }
            }
            let resident_pages = target_roots.iter().try_fold(0_usize, |count, root| {
                count
                    .checked_add(root.entries.len())
                    .ok_or(KvManagerError::ArithmeticOverflow("resident pages"))
            })?;
            plans.push((relocation, state, target_roots, resident_pages));
        }
        let reclamation_slots = self.reclamations.plan_many(retiring.len())?;
        let certificates = retiring
            .iter()
            .zip(reclamation_slots.iter().copied())
            .map(|((entry, boundary), slot)| {
                self.certificate_for_root(
                    *entry,
                    *boundary,
                    slot,
                    receipt.completion_domain,
                    receipt.completion_value,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (_, state, roots, _) in &plans {
            for source in &state.plan.source_pages {
                let mut page = self.page_mut(source.page_id)?;
                page.reader_pins -= 1;
                page.request_refs = 0;
                page.completion_domain = receipt.completion_domain;
                page.completion_value = receipt.completion_value;
            }
            for destination in &state.plan.destination_pages {
                self.set_page_phase(destination.page_id, PagePhase::Live)?;
                let mut page = self.page_mut(destination.page_id)?;
                page.request_refs = 1;
                page.completion_domain = receipt.completion_domain;
                page.completion_value = receipt.completion_value;
            }
            let target = self
                .snapshots
                .get_mut(state.target_snapshot.slot, state.target_snapshot.generation)?;
            target.roots = roots.clone().into();
        }
        for (certificate, slot) in certificates.iter().zip(reclamation_slots) {
            self.set_page_phase(
                certificate.page.page_id,
                PagePhase::Retiring {
                    reclamation: certificate.reclamation,
                },
            )?;
            self.reclamations.insert_planned(
                slot,
                ReclamationState {
                    certificate: certificate.clone(),
                },
            );
        }
        let mut publications = Vec::with_capacity(plans.len());
        for (relocation, state, _, resident_pages) in plans {
            self.snapshots
                .remove(state.base_snapshot.slot, state.base_snapshot.generation)?;
            let request = self.request_mut(state.request)?;
            request.head = state.target_snapshot;
            request.pending_relocation = None;
            request.last_completion_domain = receipt.completion_domain;
            request.last_completion_value = receipt.completion_value;
            self.relocations
                .remove(relocation.slot, relocation.generation)?;
            publications.push(super::RequestView {
                request: state.request,
                snapshot: state.target_snapshot,
                view_version: state.plan.target_version,
                boundary: self
                    .snapshots
                    .get(state.target_snapshot.slot, state.target_snapshot.generation)?
                    .boundary,
                resident_count: u32::try_from(resident_pages)
                    .map_err(|_| KvManagerError::ArithmeticOverflow("resident pages"))?,
            });
        }
        self.commit_completion_frontier(receipt.completion_domain, receipt.completion_value);
        Ok(CompletedRelocationBatch {
            publications: publications.into_boxed_slice(),
            retirements: certificates.into_boxed_slice(),
        })
    }

    /// Aborts backend-unobserved prepared relocations and returns reservations.
    ///
    /// # Errors
    ///
    /// Rejects submitted, stale, duplicated, or unproven operations before mutation.
    ///
    /// # Panics
    ///
    /// Panics only if manager-owned relocation state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn abort_relocations_batch(
        &mut self,
        receipts: &[RelocationUnobservedReceipt],
    ) -> Result<(), KvManagerError> {
        if receipts.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_relocations = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut states = Vec::with_capacity(receipts.len());
        for receipt in receipts {
            if receipt.backend_unobserved != 1 || receipt.reserved != 0 {
                return Err(KvManagerError::BackendObservationUnknown);
            }
            let relocation = receipt.relocation;
            if relocation.engine_epoch != self.engine_epoch {
                return Err(KvManagerError::WrongEngine);
            }
            if !seen_relocations.insert(relocation) {
                return Err(KvManagerError::DuplicateStep);
            }
            let state = self
                .relocations
                .get(relocation.slot, relocation.generation)?;
            if state.submitted {
                return Err(KvManagerError::StepAlreadySubmitted);
            }
            if !seen_requests.insert(state.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(state.request)?;
            if request.head != state.base_snapshot || request.pending_relocation != Some(relocation)
            {
                return Err(KvManagerError::StaleTokenView);
            }
            self.snapshots
                .get(state.target_snapshot.slot, state.target_snapshot.generation)?;
            for &destination in &state.plan.destination_pages {
                if !seen_pages.insert(destination.page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                self.validate_reserved_relocation_destination(
                    state.plan.class_id,
                    relocation,
                    destination,
                )?;
            }
            states.push((relocation, state.clone()));
        }
        let mut recycled_by_class = vec![Vec::<u32>::new(); self.classes.len()];
        for (relocation, state) in states {
            for destination in &state.plan.destination_pages {
                if destination.generation == u64::MAX {
                    self.set_page_phase(destination.page_id, PagePhase::Exhausted)
                        .expect("relocation abort preflight retained destination");
                } else {
                    self.set_page_phase(destination.page_id, PagePhase::Free)
                        .expect("relocation abort preflight retained destination");
                    recycled_by_class[usize::from(state.plan.class_id)].push(destination.page_id);
                }
            }
            self.snapshots
                .remove(state.target_snapshot.slot, state.target_snapshot.generation)
                .expect("relocation abort preflight retained target snapshot");
            self.relocations
                .remove(relocation.slot, relocation.generation)
                .expect("relocation abort preflight retained relocation");
            self.request_mut(state.request)
                .expect("relocation abort preflight retained request")
                .pending_relocation = None;
        }
        for (free, mut recycled) in self.free_pages.iter_mut().zip(recycled_by_class) {
            recycled.sort_unstable_by(|left, right| right.cmp(left));
            free.extend(recycled);
        }
        Ok(())
    }
}
