use std::collections::BTreeSet;
use std::sync::Arc;

use crate::plan::RetentionKind;

use super::persistent_snapshot::RootLayout;
use super::{
    BatchCompletionReceipt, CanonicalKvManager, CompletedRelocationBatch, KvManagerError,
    PagePhase, PersistentRootEntries, PrepareRelocationItem, PreparedRelocation, ReclamationState,
    RelocationCopyReceipt, RelocationDestination, RelocationLease, RelocationPageState,
    RelocationPlan, RelocationUnobservedReceipt, RequestSnapshot, SnapshotLease,
    SubmittedRelocation, TokenView, apply_token_relocation, plan_token_relocation,
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
            if class.retention != RetentionKind::Full || !item.policy.full_evacuation {
                return Err(KvManagerError::UnsupportedProfile(
                    "first relocation transaction requires Full evacuation",
                ));
            }
            let snapshot = self.request_snapshot(item.request)?;
            let root = snapshot
                .roots
                .get(usize::from(item.class_id))
                .ok_or(KvManagerError::InvalidClass(item.class_id))?;
            if !root.is_dense() || root.tokens.len() != snapshot.boundary {
                return Err(KvManagerError::UnsupportedProfile(
                    "nested or append-after-relocation is not implemented",
                ));
            }
            let placements = root.tokens.materialize()?;
            let view = TokenView {
                class_id: item.class_id,
                version: snapshot.view_version,
                page_tokens: u32::try_from(self.page_tokens)
                    .map_err(|_| KvManagerError::ArithmeticOverflow("page tokens"))?,
                placements,
            };
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
            let plan = plan_token_relocation(&view, &page_states, &destinations, item.policy)?
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
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(relocations.len());
        let mut receipt_offset = 0_usize;
        for &relocation in relocations {
            if relocation.engine_epoch != self.engine_epoch || !seen.insert(relocation) {
                return Err(KvManagerError::WrongEngine);
            }
            let state = self
                .relocations
                .get(relocation.slot, relocation.generation)?
                .clone();
            if state.submitted {
                return Err(KvManagerError::StepAlreadySubmitted);
            }
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
                .any(|page| !seen_pages.insert(*page))
            {
                return Err(KvManagerError::DuplicatePage);
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
                    self.set_page_phase(destination.page_id, PagePhase::Quarantined)?;
                }
                self.snapshots
                    .remove(state.target_snapshot.slot, state.target_snapshot.generation)?;
                self.relocations
                    .remove(relocation.slot, relocation.generation)?;
                let request = self.request_mut(state.request)?;
                request.pending_relocation = None;
                request.quarantined = true;
            }
            return Err(KvManagerError::BatchQuarantined(Box::new(error)));
        }
        for (relocation, state, _, _) in &plans {
            for source in &state.plan.source_pages {
                self.page_mut(source.page_id)?.reader_pins += 1;
            }
            for destination in &state.plan.destination_pages {
                self.set_page_phase(
                    destination.page_id,
                    PagePhase::Relocating {
                        relocation: *relocation,
                    },
                )?;
            }
            self.relocations
                .get_mut(relocation.slot, relocation.generation)?
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
            let view = TokenView {
                class_id: state.plan.class_id,
                version: snapshot.view_version,
                page_tokens: u32::try_from(self.page_tokens)
                    .map_err(|_| KvManagerError::ArithmeticOverflow("page tokens"))?,
                placements: root.tokens.materialize()?,
            };
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
            for source in &state.plan.source_pages {
                if !seen_pages.insert(*source) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(source.page_id)?;
                if page.reader_pins == 0 || page.request_refs != 1 || page.prefix_refs != 0 {
                    return Err(KvManagerError::StalePage);
                }
                let entry = root
                    .entries
                    .iter()
                    .copied()
                    .find(|entry| entry.page == *source)
                    .ok_or(KvManagerError::InvalidRelocationPlan)?;
                retiring.push((entry, snapshot.boundary));
            }
            for destination in &state.plan.destination_pages {
                if !seen_pages.insert(*destination) {
                    return Err(KvManagerError::DuplicatePage);
                }
            }
            let resident_tokens = target_root.resident_tokens;
            let resident_pages = target_roots.iter().try_fold(0_usize, |count, root| {
                count
                    .checked_add(root.entries.len())
                    .ok_or(KvManagerError::ArithmeticOverflow("resident pages"))
            })?;
            plans.push((
                relocation,
                state,
                target_roots,
                resident_tokens,
                resident_pages,
            ));
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
        for (_, state, roots, _, _) in &plans {
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
        for (relocation, state, _, _resident_tokens, resident_pages) in plans {
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
    pub fn abort_relocations_batch(
        &mut self,
        receipts: &[RelocationUnobservedReceipt],
    ) -> Result<(), KvManagerError> {
        if receipts.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut states = Vec::with_capacity(receipts.len());
        for receipt in receipts {
            if receipt.backend_unobserved != 1 || receipt.reserved != 0 {
                return Err(KvManagerError::BackendObservationUnknown);
            }
            let relocation = receipt.relocation;
            let state = self
                .relocations
                .get(relocation.slot, relocation.generation)?;
            if state.submitted {
                return Err(KvManagerError::StepAlreadySubmitted);
            }
            states.push((relocation, state.clone()));
        }
        for (relocation, state) in states {
            for destination in state.plan.destination_pages.iter().rev() {
                self.set_page_phase(destination.page_id, PagePhase::Free)?;
                self.free_pages[usize::from(state.plan.class_id)].push(destination.page_id);
            }
            self.snapshots
                .remove(state.target_snapshot.slot, state.target_snapshot.generation)?;
            self.relocations
                .remove(relocation.slot, relocation.generation)?;
            self.request_mut(state.request)?.pending_relocation = None;
        }
        Ok(())
    }
}
