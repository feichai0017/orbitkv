use super::{
    Arc, AttachedPrefix, BTreeMap, BTreeSet, CanonicalKvManager, DetachedReason, EvictedPrefix,
    KvManagerError, MaterializedRequestView, PageLease, PagePhase, PrefixAttachItem,
    PrefixEvictionBatch, PrefixLease, PrefixPublishItem, PrefixPublishRelease, PrefixSemanticKey,
    PrefixState, PublishedPrefix, ReclamationState, ReleaseCompletion, RequestSnapshot,
    RequestView, RootEntry, SnapshotLease, ViewVersion,
};

impl CanonicalKvManager {
    /// Publishes the complete Full/Hybrid root bundle under one page-aligned
    /// semantic key. The request retains its independent snapshot.
    ///
    /// # Errors
    ///
    /// Rejects empty/duplicate batches, unavailable requests, invalid root
    /// bundles, stale pages, reference overflow, or arena exhaustion without
    /// mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive state changes after collective preflight.
    pub fn publish_prefix_batch(
        &mut self,
        items: &[PrefixPublishItem],
    ) -> Result<Box<[PublishedPrefix]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        let mut seen_keys = BTreeSet::new();
        for item in items {
            if !seen_requests.insert(item.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            if !seen_keys.insert(item.key) || self.prefix_index.contains_key(&item.key) {
                return Err(KvManagerError::DuplicatePrefixKey);
            }
        }
        let planned = self.prefixes.plan_many(items.len())?;
        let mut increments = BTreeMap::<PageLease, u32>::new();
        let mut plans = Vec::with_capacity(items.len());
        for (item, slot) in items.iter().zip(planned.iter().copied()) {
            let state = self.request(item.request)?;
            if state.released || state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if state.head != item.expected_head {
                return Err(KvManagerError::StaleView);
            }
            if state.pending_step.is_some() || state.inflight_submission.is_some() {
                return Err(KvManagerError::RequestBusy);
            }
            let snapshot = self.request_snapshot(item.request)?;
            let entries = self.validate_prefix_bundle(snapshot, item.key)?;
            for entry in entries {
                let increment = increments.entry(entry.page).or_default();
                *increment = increment
                    .checked_add(1)
                    .ok_or(KvManagerError::ReferenceCountOverflow(entry.page.page_id))?;
            }
            let prefix = PrefixLease {
                engine_epoch: self.engine_epoch,
                slot: slot.0,
                generation: slot.1,
            };
            let resident_count = u32::try_from(snapshot.resident_count())
                .map_err(|_| KvManagerError::ArithmeticOverflow("prefix resident count"))?;
            plans.push((
                slot,
                prefix,
                item.key,
                Arc::clone(&snapshot.roots),
                resident_count,
            ));
        }
        for (lease, increment) in &increments {
            let page = self.page(lease.page_id)?;
            if page.generation != lease.generation
                || page.phase != PagePhase::Live
                || page.reader_pins != 0
                || page.writer.is_some()
                || page.prefix_refs.checked_add(*increment).is_none()
            {
                return Err(KvManagerError::StalePage);
            }
        }
        for (lease, increment) in increments {
            self.page_mut(lease.page_id)
                .expect("prefix publication preflight retained page")
                .prefix_refs += increment;
        }
        let mut outputs = Vec::with_capacity(plans.len());
        for (slot, prefix, key, roots, resident_count) in plans {
            self.prefixes.insert_planned(
                slot,
                PrefixState {
                    key,
                    roots,
                    evicted: false,
                },
            );
            let previous = self.prefix_index.insert(key, prefix);
            debug_assert!(previous.is_none());
            outputs.push(PublishedPrefix {
                prefix,
                key,
                resident_count,
            });
        }
        self.active_prefixes += outputs.len() as u64;
        Ok(outputs.into_boxed_slice())
    }

    /// Atomically revalidates and attaches an ordered hint batch. Every hit
    /// receives a distinct snapshot lease while the immutable root bundle is
    /// shared. A stale hint rejects the batch without mutation.
    ///
    /// # Errors
    ///
    /// Rejects stale/missing hints, non-empty or unavailable requests,
    /// reference overflow, or snapshot arena exhaustion without mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive state changes after collective preflight.
    #[allow(clippy::too_many_lines)]
    pub fn attach_prefix_batch(
        &mut self,
        items: &[PrefixAttachItem],
    ) -> Result<Box<[AttachedPrefix]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        for item in items {
            if !seen_requests.insert(item.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
        }
        let planned_snapshots = self.snapshots.plan_many(items.len())?;
        let mut page_increments = BTreeMap::<PageLease, u32>::new();
        let mut plans = Vec::with_capacity(items.len());
        for (item, planned) in items.iter().zip(planned_snapshots.iter().copied()) {
            let prefix = item.hint.candidate.ok_or(KvManagerError::PrefixMiss)?;
            self.check_prefix_epoch(prefix)?;
            let prefix_state = self.prefixes.get(prefix.slot, prefix.generation)?;
            if prefix_state.evicted
                || prefix_state.key != item.hint.key
                || self.prefix_index.get(&item.hint.key) != Some(&prefix)
            {
                return Err(KvManagerError::PrefixHintStale);
            }
            let request_state = self.request(item.request)?;
            if request_state.released || request_state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if request_state.head != item.expected_empty_head {
                return Err(KvManagerError::StaleView);
            }
            if request_state.pending_step.is_some() || request_state.inflight_submission.is_some() {
                return Err(KvManagerError::RequestBusy);
            }
            let old_snapshot = self.request_snapshot(item.request)?;
            if old_snapshot.boundary != 0 || !old_snapshot.is_empty() {
                return Err(KvManagerError::AttachRequiresEmptyRequest);
            }
            let version = ViewVersion(
                old_snapshot
                    .view_version
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ViewVersionExhausted)?,
            );
            let pages =
                self.materialize_snapshot_roots(prefix_state.key.boundary, &prefix_state.roots)?;
            let resident_count = u32::try_from(pages.len())
                .map_err(|_| KvManagerError::ArithmeticOverflow("resident count"))?;
            for entry in Self::root_entries(&prefix_state.roots) {
                *page_increments.entry(entry.page).or_default() = page_increments
                    .get(&entry.page)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or(KvManagerError::ReferenceCountOverflow(entry.page.page_id))?;
            }
            let snapshot = SnapshotLease {
                engine_epoch: self.engine_epoch,
                slot: planned.0,
                generation: planned.1,
            };
            plans.push((
                item.request,
                request_state.head,
                prefix,
                snapshot,
                planned,
                version,
                prefix_state.key.boundary,
                Arc::clone(&prefix_state.roots),
                resident_count,
                pages,
            ));
        }
        for (page_lease, increment) in &page_increments {
            let class = self.runtime_class(self.page(page_lease.page_id)?.class_id)?;
            self.validate_page_lease(class, *page_lease)?;
            let page = self.page(page_lease.page_id)?;
            if page.phase != PagePhase::Live
                || page.reader_pins != 0
                || page.writer.is_some()
                || page.generation != page_lease.generation
                || page.request_refs.checked_add(*increment).is_none()
            {
                return Err(KvManagerError::StalePage);
            }
        }
        for (page, increment) in page_increments {
            self.page_mut(page.page_id)
                .expect("prefix attach preflight retained page")
                .request_refs += increment;
        }
        for (_, _, _, _, planned, version, boundary, roots, _, _) in &plans {
            self.snapshots.insert_planned(
                *planned,
                RequestSnapshot {
                    boundary: *boundary,
                    view_version: *version,
                    roots: Arc::clone(roots),
                },
            );
        }
        let mut outputs = Vec::with_capacity(plans.len());
        for (request, old_head, prefix, snapshot, _, version, boundary, _, resident_count, pages) in
            plans
        {
            self.snapshots
                .remove(old_head.slot, old_head.generation)
                .expect("prefix attach preflight retained empty snapshot");
            self.request_mut(request)
                .expect("prefix attach preflight retained request")
                .head = snapshot;
            outputs.push(AttachedPrefix {
                prefix,
                target: MaterializedRequestView {
                    view: RequestView {
                        request,
                        snapshot,
                        view_version: version,
                        boundary,
                        resident_count,
                    },
                    pages,
                },
            });
            debug_assert_eq!(
                self.request_snapshot(request)
                    .expect("attached snapshot")
                    .resident_count(),
                resident_count as usize
            );
        }
        Ok(outputs.into_boxed_slice())
    }

    /// Publishes a prefix and releases the source request in one reference
    /// transaction. Every page performs an exact request->prefix transfer.
    ///
    /// # Errors
    ///
    /// Rejects empty/duplicate batches, invalid root bundles, unavailable
    /// requests, stale pages, overflow, or arena exhaustion without mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive state changes after collective preflight.
    #[allow(clippy::too_many_lines)]
    pub fn publish_prefix_and_release_batch(
        &mut self,
        items: &[PrefixPublishItem],
    ) -> Result<Box<[PrefixPublishRelease]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        let mut seen_keys = BTreeSet::new();
        for item in items {
            if !seen_requests.insert(item.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            if !seen_keys.insert(item.key) || self.prefix_index.contains_key(&item.key) {
                return Err(KvManagerError::DuplicatePrefixKey);
            }
        }
        let planned = self.prefixes.plan_many(items.len())?;
        let mut transfers = BTreeMap::<PageLease, u32>::new();
        let mut plans = Vec::with_capacity(items.len());
        for (item, slot) in items.iter().zip(planned.iter().copied()) {
            let state = self.request(item.request)?;
            if state.released || state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if state.head != item.expected_head {
                return Err(KvManagerError::StaleView);
            }
            if state.pending_step.is_some() || state.inflight_submission.is_some() {
                return Err(KvManagerError::RequestBusy);
            }
            let detached_snapshot = state.head;
            let snapshot = self.request_snapshot(item.request)?;
            let entries = self.validate_prefix_bundle(snapshot, item.key)?;
            for entry in &entries {
                let transfer = transfers.entry(entry.page).or_default();
                *transfer = transfer
                    .checked_add(1)
                    .ok_or(KvManagerError::ReferenceCountOverflow(entry.page.page_id))?;
            }
            let detached = entries
                .iter()
                .copied()
                .map(|entry| {
                    self.clear_detached_binding(
                        entry,
                        snapshot.boundary,
                        DetachedReason::PrefixTransfer,
                    )
                })
                .collect::<Result<Vec<_>, KvManagerError>>()?;
            let prefix = PrefixLease {
                engine_epoch: self.engine_epoch,
                slot: slot.0,
                generation: slot.1,
            };
            let resident_count = u32::try_from(snapshot.resident_count())
                .map_err(|_| KvManagerError::ArithmeticOverflow("prefix resident count"))?;
            plans.push((
                item.request,
                detached_snapshot,
                slot,
                prefix,
                item.key,
                Arc::clone(&snapshot.roots),
                resident_count,
                detached,
            ));
        }
        for (lease, transfer) in &transfers {
            let page = self.page(lease.page_id)?;
            if page.generation != lease.generation
                || page.phase != PagePhase::Live
                || page.reader_pins != 0
                || page.writer.is_some()
                || page.request_refs < *transfer
                || page.prefix_refs.checked_add(*transfer).is_none()
            {
                return Err(KvManagerError::StalePage);
            }
        }
        for (lease, transfer) in transfers {
            let mut page = self
                .page_mut(lease.page_id)
                .expect("prefix transfer preflight retained page");
            page.request_refs -= transfer;
            page.prefix_refs += transfer;
        }
        let mut outputs = Vec::with_capacity(plans.len());
        for (request, detached_snapshot, slot, prefix, key, roots, resident_count, detached) in
            plans
        {
            self.prefixes.insert_planned(
                slot,
                PrefixState {
                    key,
                    roots,
                    evicted: false,
                },
            );
            let previous = self.prefix_index.insert(key, prefix);
            debug_assert!(previous.is_none());
            self.snapshots
                .remove(detached_snapshot.slot, detached_snapshot.generation)
                .expect("prefix transfer preflight retained source snapshot");
            self.request_mut(request)
                .expect("prefix transfer preflight retained source request")
                .released = true;
            outputs.push(PrefixPublishRelease {
                publication: PublishedPrefix {
                    prefix,
                    key,
                    resident_count,
                },
                release: ReleaseCompletion {
                    request,
                    detached_snapshot,
                    detached: detached.into_boxed_slice(),
                },
            });
        }
        self.active_prefixes += outputs.len() as u64;
        Ok(outputs.into_boxed_slice())
    }

    /// Evicts exact prefix identities and emits one certificate for each page
    /// whose aggregate request/prefix reference count reaches zero.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate, stale, busy, or under-provisioned batches
    /// without changing prefix or page ownership.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive state changes after collective preflight.
    pub fn evict_prefix_batch(
        &mut self,
        prefixes: &[PrefixLease],
    ) -> Result<PrefixEvictionBatch, KvManagerError> {
        if prefixes.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        let mut decrements = BTreeMap::<PageLease, (u32, RootEntry)>::new();
        let mut evicted = Vec::with_capacity(prefixes.len());
        for &prefix in prefixes {
            if !seen.insert(prefix) {
                return Err(KvManagerError::DuplicatePrefix);
            }
            self.check_prefix_epoch(prefix)?;
            let state = self.prefixes.get(prefix.slot, prefix.generation)?;
            if state.evicted || self.prefix_index.get(&state.key) != Some(&prefix) {
                return Err(KvManagerError::PrefixHintStale);
            }
            for entry in Self::root_entries(&state.roots) {
                let delta = decrements.entry(entry.page).or_insert((0, entry));
                delta.0 = delta
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ReferenceCountOverflow(entry.page.page_id))?;
            }
            evicted.push(EvictedPrefix {
                prefix,
                key: state.key,
            });
        }
        let mut retiring = Vec::new();
        for (lease, (decrement, entry)) in &decrements {
            let page = self.page(lease.page_id)?;
            if page.generation != lease.generation
                || page.phase != PagePhase::Live
                || page.reader_pins != 0
                || page.writer.is_some()
                || page.prefix_refs < *decrement
            {
                return Err(KvManagerError::StalePage);
            }
            if page.request_refs == 0 && page.prefix_refs == *decrement {
                retiring.push((*entry, page.completion_domain, page.completion_value));
            }
        }
        let planned = self.reclamations.plan_many(retiring.len())?;
        let certificates = retiring
            .iter()
            .zip(planned.iter().copied())
            .map(|((entry, domain, value), slot)| {
                let page_end = entry
                    .logical_ordinal
                    .checked_add(1)
                    .and_then(|ordinal| ordinal.checked_mul(self.page_tokens))
                    .ok_or(KvManagerError::ArithmeticOverflow("prefix page end"))?;
                self.certificate_for_root(*entry, page_end, slot, *domain, *value)
            })
            .collect::<Result<Vec<_>, KvManagerError>>()?;
        for (lease, (decrement, _)) in decrements {
            self.page_mut(lease.page_id)
                .expect("prefix eviction preflight retained page")
                .prefix_refs -= decrement;
        }
        for (certificate, slot) in certificates.iter().zip(planned) {
            self.set_page_phase(
                certificate.page.page_id,
                PagePhase::Retiring {
                    reclamation: certificate.reclamation,
                },
            )
            .expect("prefix eviction preflight retained retiring page transition");
            self.reclamations.insert_planned(
                slot,
                ReclamationState {
                    certificate: certificate.clone(),
                },
            );
        }
        for item in &evicted {
            let removed = self.prefix_index.remove(&item.key);
            debug_assert_eq!(removed, Some(item.prefix));
            let state = self
                .prefixes
                .get_mut(item.prefix.slot, item.prefix.generation)
                .expect("prefix eviction preflight retained prefix");
            state.evicted = true;
            state.roots = Arc::from([]);
        }
        self.active_prefixes -= evicted.len() as u64;
        self.evicted_prefixes += evicted.len() as u64;
        Ok(PrefixEvictionBatch {
            evicted: evicted.into_boxed_slice(),
            retirements: certificates.into_boxed_slice(),
        })
    }

    /// Recycles already-evicted prefix identities. Outstanding page
    /// certificates are transaction-owned and do not retain prefix slots.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate, stale, or live prefix identities without
    /// advancing any prefix generation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive state changes after collective preflight.
    pub fn recycle_prefixes_batch(
        &mut self,
        prefixes: &[PrefixLease],
    ) -> Result<(), KvManagerError> {
        if prefixes.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        for &prefix in prefixes {
            if !seen.insert(prefix) {
                return Err(KvManagerError::DuplicatePrefix);
            }
            self.check_prefix_epoch(prefix)?;
            if !self.prefixes.get(prefix.slot, prefix.generation)?.evicted {
                return Err(KvManagerError::PrefixNotEvicted);
            }
        }
        for &prefix in prefixes {
            self.prefixes
                .remove(prefix.slot, prefix.generation)
                .expect("prefix recycle preflight retained prefix");
        }
        self.evicted_prefixes -= prefixes.len() as u64;
        Ok(())
    }
    fn validate_prefix_bundle(
        &self,
        snapshot: &RequestSnapshot,
        key: PrefixSemanticKey,
    ) -> Result<Vec<RootEntry>, KvManagerError> {
        if key.boundary == 0 || !key.boundary.is_multiple_of(self.page_tokens) {
            return Err(KvManagerError::PrefixBoundaryNotPageAligned);
        }
        if snapshot.boundary != key.boundary || snapshot.roots.len() != self.classes.len() {
            return Err(KvManagerError::PrefixBoundaryMismatch);
        }
        let end = key.boundary / self.page_tokens;
        let mut seen = BTreeSet::new();
        let mut entries = Vec::new();
        for (class, root) in self.classes.iter().copied().zip(snapshot.roots.iter()) {
            let first = class.retained_start(key.boundary) / self.page_tokens;
            let expected_len = usize::try_from(end.saturating_sub(first))
                .map_err(|_| KvManagerError::ArithmeticOverflow("prefix root length"))?;
            if root.entries.len() != expected_len {
                return Err(KvManagerError::PrefixRootMismatch);
            }
            for (offset, entry) in root.entries.iter().copied().enumerate() {
                let ordinal = first
                    .checked_add(offset as u64)
                    .ok_or(KvManagerError::ArithmeticOverflow("prefix ordinal"))?;
                if self.root_entry_for_page(class, ordinal, entry.page)? != entry
                    || !seen.insert(entry.page)
                {
                    return Err(KvManagerError::PrefixRootMismatch);
                }
                let page = self.page(entry.page.page_id)?;
                if page.generation != entry.page.generation
                    || page.phase != PagePhase::Live
                    || page.request_refs == 0
                    || page.reader_pins != 0
                    || page.writer.is_some()
                {
                    return Err(KvManagerError::StalePage);
                }
                entries.push(entry);
            }
        }
        Ok(entries)
    }
}
