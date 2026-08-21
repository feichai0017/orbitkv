use super::{
    BTreeMap, BTreeSet, CanonicalKvManager, DetachedAction, DetachedBinding, DetachedReason,
    KvManagerError, PageLease, PagePhase, ReclamationCertificate, ReclamationLease,
    ReclamationReceipt, ReclamationState, ReleaseBatchCompletion, ReleaseBatchItem,
    ReleaseCompletion, RequestLease, RootEntry,
};

impl CanonicalKvManager {
    /// Atomically releases an ordered batch of quiescent requests.
    ///
    /// # Errors
    ///
    /// Any duplicate, unavailable, busy, stale, or under-provisioned item
    /// rejects the entire batch without changing a root or page phase.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn release_batch(
        &mut self,
        items: &[ReleaseBatchItem],
    ) -> Result<ReleaseBatchCompletion, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        let mut states = Vec::with_capacity(items.len());
        let mut page_detaches = BTreeMap::<PageLease, (u32, RootEntry, u64)>::new();
        for item in items {
            let request = item.request;
            if !seen_requests.insert(request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let state = self.request(request)?;
            if state.released || state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if state.head != item.expected_head {
                return Err(KvManagerError::StaleView);
            }
            if state.pending_step.is_some() || state.inflight_submission.is_some() {
                return Err(KvManagerError::RequestBusy);
            }
            let snapshot = self.request_snapshot(request)?;
            let entries = Self::root_entries(&snapshot.roots);
            for entry in &entries {
                let aggregate =
                    page_detaches
                        .entry(entry.page)
                        .or_insert((0, *entry, snapshot.boundary));
                aggregate.0 = aggregate
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ReferenceCountOverflow(entry.page.page_id))?;
                aggregate.2 = aggregate.2.max(snapshot.boundary);
            }
            let detached = entries
                .iter()
                .copied()
                .map(|entry| {
                    self.clear_detached_binding(
                        entry,
                        snapshot.boundary,
                        DetachedReason::RequestRelease,
                    )
                })
                .collect::<Result<Vec<_>, KvManagerError>>()?;
            states.push((request, state.head, detached));
        }
        let mut retiring = Vec::new();
        for (lease, (decrement, entry, boundary)) in &page_detaches {
            let page = self.page(lease.page_id)?;
            if page.generation != lease.generation
                || page.phase != PagePhase::Live
                || page.reader_pins != 0
                || page.writer.is_some()
                || page.request_refs < *decrement
            {
                return Err(KvManagerError::StalePage);
            }
            if page.request_refs == *decrement && page.prefix_refs == 0 {
                retiring.push((
                    *entry,
                    *boundary,
                    page.completion_domain,
                    page.completion_value,
                ));
            }
        }
        let planned = self.reclamations.plan_many(retiring.len())?;
        let certificates = retiring
            .iter()
            .zip(planned.iter().copied())
            .map(|((entry, boundary, domain, value), slot)| {
                self.certificate_for_root(*entry, *boundary, slot, *domain, *value)
            })
            .collect::<Result<Vec<_>, KvManagerError>>()?;
        for (lease, (decrement, _, _)) in page_detaches {
            self.page_mut(lease.page_id)
                .expect("batch release preflight retained page")
                .request_refs -= decrement;
        }
        for (certificate, slot) in certificates.iter().zip(planned) {
            self.set_page_phase(
                certificate.page.page_id,
                PagePhase::Retiring {
                    reclamation: certificate.reclamation,
                },
            )
            .expect("batch release preflight retained retiring page transition");
            self.reclamations.insert_planned(
                slot,
                ReclamationState {
                    certificate: certificate.clone(),
                },
            );
        }
        let mut releases = Vec::with_capacity(states.len());
        for (request, detached_snapshot, detached) in states {
            self.snapshots
                .remove(detached_snapshot.slot, detached_snapshot.generation)
                .expect("batch release preflight retained request snapshot");
            let request_state = self
                .request_mut(request)
                .expect("batch release preflight retained request");
            request_state.released = true;
            releases.push(ReleaseCompletion {
                request,
                detached_snapshot,
                detached: detached.into_boxed_slice(),
            });
        }
        Ok(ReleaseBatchCompletion {
            releases: releases.into_boxed_slice(),
            retirements: certificates.into_boxed_slice(),
        })
    }

    #[cfg(test)]
    pub(super) fn release_current_for_test(
        &mut self,
        requests: &[RequestLease],
    ) -> Result<ReleaseBatchCompletion, KvManagerError> {
        let items = requests
            .iter()
            .copied()
            .map(|request| {
                Ok(ReleaseBatchItem {
                    request,
                    expected_head: self.request(request)?.head,
                })
            })
            .collect::<Result<Vec<_>, KvManagerError>>()?;
        self.release_batch(&items)
    }

    /// Atomically acknowledges a complete reclamation batch.
    ///
    /// # Errors
    ///
    /// Returns an error before mutation when any receipt is stale, duplicated,
    /// unacknowledged, or does not exactly match its certificate.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn acknowledge_reclamations_batch(
        &mut self,
        receipts: &[ReclamationReceipt],
    ) -> Result<(), KvManagerError> {
        if receipts.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        let mut states = Vec::with_capacity(receipts.len());
        for receipt in receipts {
            if receipt.reserved8 != 0 || receipt.reserved32 != 0 {
                return Err(KvManagerError::ReservedFieldNonZero);
            }
            if receipt.acknowledged != 1 {
                return Err(KvManagerError::ReclamationNotAcknowledged);
            }
            self.check_reclamation_epoch(receipt.reclamation)?;
            if !seen.insert(receipt.reclamation) {
                return Err(KvManagerError::DuplicateReclamation);
            }
            let state = self
                .reclamations
                .get(receipt.reclamation.slot, receipt.reclamation.generation)?;
            let certificate = &state.certificate;
            if certificate.reclamation != receipt.reclamation
                || certificate.page != receipt.page
                || certificate.backend_domain != receipt.backend_domain
                || certificate.backend_index != receipt.backend_index
            {
                return Err(KvManagerError::ReclamationMismatch);
            }
            let page = self.page(receipt.page.page_id)?;
            if page.reader_pins != 0
                || page.writer.is_some()
                || page.request_refs != 0
                || page.prefix_refs != 0
                || page.generation != receipt.page.generation
                || page.phase
                    != (PagePhase::Retiring {
                        reclamation: receipt.reclamation,
                    })
            {
                return Err(KvManagerError::StalePage);
            }
            states.push(receipt.reclamation);
        }
        let mut recycled_by_class = vec![Vec::<u32>::new(); self.classes.len()];
        for reclamation in states {
            let certificate = self
                .reclamations
                .remove(reclamation.slot, reclamation.generation)
                .expect("reclamation preflight retained certificate")
                .certificate;
            let page_id = certificate.page.page_id;
            let (generation, class_id) = {
                let page = self
                    .page(page_id)
                    .expect("reclamation preflight retained page");
                debug_assert_eq!(page.reader_pins, 0);
                debug_assert_eq!(page.writer, None);
                debug_assert_eq!(page.request_refs, 0);
                debug_assert_eq!(page.prefix_refs, 0);
                debug_assert_eq!(page.phase, PagePhase::Retiring { reclamation });
                (page.generation, page.class_id)
            };
            if generation == u64::MAX {
                self.set_page_phase(page_id, PagePhase::Exhausted)
                    .expect("reclamation preflight retained page");
            } else {
                self.set_page_phase(page_id, PagePhase::Free)
                    .expect("reclamation preflight retained page");
                recycled_by_class[usize::from(class_id)].push(page_id);
            }
        }
        for (free, mut recycled) in self.free_pages.iter_mut().zip(recycled_by_class) {
            recycled.sort_unstable_by(|left, right| right.cmp(left));
            free.extend(recycled);
        }
        Ok(())
    }

    /// Atomically recycles a non-empty batch of fully released identities.
    ///
    /// # Errors
    ///
    /// Any duplicate, stale, or non-recyclable request rejects the whole
    /// batch without advancing any request generation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn recycle_requests_batch(
        &mut self,
        requests: &[RequestLease],
    ) -> Result<(), KvManagerError> {
        if requests.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        for &request in requests {
            if !seen.insert(request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let state = self.request(request)?;
            if !state.released
                || state.pending_step.is_some()
                || state.inflight_submission.is_some()
            {
                return Err(KvManagerError::RequestNotRecyclable);
            }
            if self
                .snapshots
                .get(state.head.slot, state.head.generation)
                .is_ok()
            {
                return Err(KvManagerError::RequestNotRecyclable);
            }
        }
        for &request in requests {
            self.requests
                .remove(request.slot, request.generation)
                .expect("batch recycle preflight retained request");
        }
        Ok(())
    }
    fn detached_binding_span(
        &self,
        entry: RootEntry,
        resident_boundary: u64,
    ) -> Result<(u64, u64), KvManagerError> {
        let token_begin = entry.logical_ordinal.checked_mul(self.page_tokens).ok_or(
            KvManagerError::ArithmeticOverflow("detached binding token begin"),
        )?;
        let token_end_exclusive = token_begin
            .checked_add(self.page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow(
                "detached binding token end",
            ))?
            .min(resident_boundary);
        if token_end_exclusive <= token_begin {
            return Err(KvManagerError::Invariant("empty detached binding span"));
        }
        Ok((token_begin, token_end_exclusive))
    }

    pub(super) fn clear_detached_binding(
        &self,
        old: RootEntry,
        resident_boundary: u64,
        reason: DetachedReason,
    ) -> Result<DetachedBinding, KvManagerError> {
        let (token_begin, token_end_exclusive) =
            self.detached_binding_span(old, resident_boundary)?;
        Ok(DetachedBinding {
            old: old.page,
            replacement: PageLease::default(),
            logical_ordinal: old.logical_ordinal,
            old_backend_index: old.backend_index,
            replacement_backend_index: 0,
            token_begin,
            token_end_exclusive,
            class_id: old.class_id,
            backend_domain: old.backend_domain,
            action: DetachedAction::Clear,
            reason,
            reserved: 0,
        })
    }

    pub(super) fn replace_detached_binding(
        &self,
        old: RootEntry,
        replacement: RootEntry,
        old_resident_boundary: u64,
    ) -> Result<DetachedBinding, KvManagerError> {
        if old.class_id != replacement.class_id
            || old.backend_domain != replacement.backend_domain
            || old.logical_ordinal != replacement.logical_ordinal
        {
            return Err(KvManagerError::Invariant("COW detached binding identity"));
        }
        let (token_begin, token_end_exclusive) =
            self.detached_binding_span(old, old_resident_boundary)?;
        Ok(DetachedBinding {
            old: old.page,
            replacement: replacement.page,
            logical_ordinal: old.logical_ordinal,
            old_backend_index: old.backend_index,
            replacement_backend_index: replacement.backend_index,
            token_begin,
            token_end_exclusive,
            class_id: old.class_id,
            backend_domain: old.backend_domain,
            action: DetachedAction::Replace,
            reason: DetachedReason::CopyOnWrite,
            reserved: 0,
        })
    }

    pub(super) fn certificate_for_root(
        &self,
        entry: RootEntry,
        target_boundary: u64,
        planned: (u32, u32),
        completion_domain: u64,
        completion_value: u64,
    ) -> Result<ReclamationCertificate, KvManagerError> {
        let token_begin = entry.logical_ordinal.checked_mul(self.page_tokens).ok_or(
            KvManagerError::ArithmeticOverflow("reclamation token begin"),
        )?;
        let page_end = token_begin
            .checked_add(self.page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow("reclamation token end"))?;
        let token_end_exclusive = target_boundary.min(page_end);
        if token_end_exclusive <= token_begin {
            return Err(KvManagerError::Invariant("empty reclamation token span"));
        }
        Ok(ReclamationCertificate {
            reclamation: ReclamationLease {
                engine_epoch: self.engine_epoch,
                slot: planned.0,
                generation: planned.1,
            },
            page: entry.page,
            class_id: entry.class_id,
            backend_domain: entry.backend_domain,
            logical_ordinal: entry.logical_ordinal,
            backend_index: entry.backend_index,
            token_begin,
            token_end_exclusive,
            completion_domain,
            completion_value,
        })
    }

    pub(super) fn insert_completion_candidate(
        &self,
        candidates: &mut BTreeMap<PageLease, (RootEntry, u64)>,
        entry: RootEntry,
        resident_boundary: u64,
    ) -> Result<(), KvManagerError> {
        let token_begin = entry
            .logical_ordinal
            .checked_mul(self.page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow("candidate token begin"))?;
        let token_end = token_begin
            .checked_add(self.page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow("candidate token end"))?
            .min(resident_boundary);
        if token_end <= token_begin {
            return Err(KvManagerError::Invariant("empty candidate resident span"));
        }
        if let Some((resident, existing_end)) = candidates.get(&entry.page) {
            if *resident != entry || *existing_end != token_end {
                return Err(KvManagerError::Invariant(
                    "shared page resident span mismatch",
                ));
            }
        } else {
            candidates.insert(entry.page, (entry, token_end));
        }
        Ok(())
    }
}
