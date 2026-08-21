#[cfg(test)]
use super::root_instrumentation;
use super::{
    Arc, BTreeMap, BTreeSet, BackendBindReceipt, BackendCopyReceipt, BackendUnobservedReceipt,
    BatchCompletionReceipt, CanonicalKvManager, ClassDelta, ClassLowering, ClassRoot,
    ClassTransition, CompletionBatch, CopyIntent, DetachedReason, KvManagerError, OperationState,
    PageLease, PagePhase, PersistentRootEntries, PrepareBatchItem, PreparedState, PreparedStep,
    PublishedReceipt, ReclamationState, RequestSnapshot, RetentionKind, RootEntry, SnapshotLease,
    StepCompletion, StepDelta, StepLease, SubmissionLease, SubmitBatchItem, SubmittedState,
    SubmittedStep, TailAction, TailActionKind, ViewVersion, WriteIntent,
};

impl CanonicalKvManager {
    /// Atomically reserves manager-selected pages for an ordered request batch.
    ///
    /// Every request, operation slot, target boundary, and physical page is
    /// preflighted for the entire batch. Any error leaves requests, operation
    /// generations, page generations, and free lists unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty batch, duplicate request, stale identity,
    /// invalid boundary, insufficient operation capacity, or insufficient
    /// physical capacity.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn prepare_batch(
        &mut self,
        items: &[PrepareBatchItem],
    ) -> Result<Box<[PreparedStep]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        for item in items {
            if !seen_requests.insert(item.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
        }
        let planned_operations = self.operations.plan_many(items.len())?;
        let planned_snapshots = self.snapshots.plan_many(items.len())?;
        let mut page_cursors = vec![0_usize; self.classes.len()];
        let mut plans = Vec::with_capacity(items.len());

        #[cfg(test)]
        let mut delta_entries_touched = 0_u64;

        for ((item, planned_operation), planned_snapshot) in items
            .iter()
            .zip(planned_operations.iter().copied())
            .zip(planned_snapshots.iter().copied())
        {
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
            if snapshot.roots.len() != self.classes.len() {
                return Err(KvManagerError::Invariant("snapshot class cardinality"));
            }
            if item.target_boundary <= snapshot.boundary {
                return Err(KvManagerError::NonMonotonicBoundary {
                    current: snapshot.boundary,
                    target: item.target_boundary,
                });
            }
            let step_tokens = item.target_boundary - snapshot.boundary;
            if step_tokens > self.maximum_step_tokens {
                return Err(KvManagerError::StepTooLarge {
                    requested: step_tokens,
                    maximum: self.maximum_step_tokens,
                });
            }
            let target_version = ViewVersion(
                snapshot
                    .view_version
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ViewVersionExhausted)?,
            );
            let step = StepLease {
                engine_epoch: self.engine_epoch,
                slot: planned_operation.0,
                generation: planned_operation.1,
            };
            let base_snapshot = state.head;
            let target_snapshot = SnapshotLease {
                engine_epoch: self.engine_epoch,
                slot: planned_snapshot.0,
                generation: planned_snapshot.1,
            };
            let previous_boundary = snapshot.boundary;
            let first_new_ordinal = previous_boundary.div_ceil(self.page_tokens);
            let new_end_ordinal = item.target_boundary.div_ceil(self.page_tokens);
            let previous_tail_ordinal = (previous_boundary % self.page_tokens != 0)
                .then_some(previous_boundary / self.page_tokens);
            let tails = self
                .classes
                .iter()
                .copied()
                .map(|class| {
                    let root = snapshot
                        .roots
                        .get(usize::from(class.class_id))
                        .ok_or(KvManagerError::Invariant("snapshot class cardinality"))?;
                    Ok(previous_tail_ordinal.and_then(|ordinal| {
                        root.entries.back().copied().filter(|entry| {
                            entry.class_id == class.class_id && entry.logical_ordinal == ordinal
                        })
                    }))
                })
                .collect::<Result<Vec<_>, KvManagerError>>()?;
            let joint_cow =
                tails
                    .iter()
                    .flatten()
                    .copied()
                    .try_fold(false, |needs_cow, tail| {
                        self.tail_is_exclusive(tail)
                            .map(|exclusive| needs_cow || !exclusive)
                    })?;
            let mut planned_pages = Vec::new();
            let mut class_lowerings = Vec::with_capacity(self.classes.len());
            let mut tail_actions = Vec::with_capacity(self.classes.len());
            let mut copy_intents = Vec::new();
            let mut write_intents = Vec::new();
            let mut class_deltas = Vec::with_capacity(self.classes.len());

            for (class, previous_tail) in self.classes.iter().copied().zip(tails) {
                let class_tail_offset = u32::try_from(tail_actions.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class tail offset"))?;
                let class_copy_offset = u32::try_from(copy_intents.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class copy offset"))?;
                let class_write_offset = u32::try_from(write_intents.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class write offset"))?;
                let (tail_action, tail_destination, copy_intent) =
                    if let Some(ordinal) = previous_tail_ordinal {
                        let valid = u32::try_from(previous_boundary % self.page_tokens)
                            .map_err(|_| KvManagerError::ArithmeticOverflow("tail valid tokens"))?;
                        match previous_tail {
                            Some(source) if joint_cow => {
                                let page = self.plan_free_page(class, &mut page_cursors)?;
                                planned_pages.push(page);
                                let destination = self.root_entry_for_page(class, ordinal, page)?;
                                let intent = CopyIntent {
                                    class_id: class.class_id,
                                    backend_domain: class.backend.backend_domain,
                                    token_count: valid,
                                    source_token_offset: 0,
                                    destination_token_offset: 0,
                                    reserved: 0,
                                    source: source.page,
                                    destination: destination.page,
                                    source_backend_index: source.backend_index,
                                    destination_backend_index: destination.backend_index,
                                };
                                copy_intents.push(intent);
                                (
                                    TailAction {
                                        class_id: class.class_id,
                                        kind: TailActionKind::CopyOnWrite,
                                        valid_token_count: valid,
                                        logical_ordinal: ordinal,
                                        source: source.page,
                                        destination: destination.page,
                                        reserved: 0,
                                    },
                                    Some(destination),
                                    Some(intent),
                                )
                            }
                            Some(source) => (
                                TailAction {
                                    class_id: class.class_id,
                                    kind: TailActionKind::InPlace,
                                    valid_token_count: valid,
                                    logical_ordinal: ordinal,
                                    source: source.page,
                                    destination: source.page,
                                    reserved: 0,
                                },
                                None,
                                None,
                            ),
                            None => {
                                let page = self.plan_free_page(class, &mut page_cursors)?;
                                planned_pages.push(page);
                                let destination = self.root_entry_for_page(class, ordinal, page)?;
                                (
                                    TailAction {
                                        class_id: class.class_id,
                                        kind: TailActionKind::Fresh,
                                        valid_token_count: 0,
                                        logical_ordinal: ordinal,
                                        source: PageLease::default(),
                                        destination: destination.page,
                                        reserved: 0,
                                    },
                                    Some(destination),
                                    None,
                                )
                            }
                        }
                    } else {
                        (
                            TailAction {
                                class_id: class.class_id,
                                kind: TailActionKind::None,
                                valid_token_count: 0,
                                logical_ordinal: 0,
                                source: PageLease::default(),
                                destination: PageLease::default(),
                                reserved: 0,
                            },
                            None,
                            None,
                        )
                    };
                tail_actions.push(tail_action);
                let mut writes = Vec::with_capacity(
                    usize::try_from(new_end_ordinal.saturating_sub(first_new_ordinal))
                        .map_err(|_| KvManagerError::ArithmeticOverflow("class write count"))?,
                );
                for ordinal in first_new_ordinal..new_end_ordinal {
                    let page = self.plan_free_page(class, &mut page_cursors)?;
                    planned_pages.push(page);
                    let entry = self.root_entry_for_page(class, ordinal, page)?;
                    write_intents.push(WriteIntent {
                        page_generation: page.generation,
                        page_id: page.page_id,
                        reserved: 0,
                    });
                    writes.push(entry);
                }
                let class_write_count = u32::try_from(write_intents.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class write count"))?
                    .checked_sub(class_write_offset)
                    .ok_or(KvManagerError::Invariant("class write range"))?;
                let class_copy_count = u32::try_from(copy_intents.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class copy count"))?
                    .checked_sub(class_copy_offset)
                    .ok_or(KvManagerError::Invariant("class copy range"))?;
                class_lowerings.push(ClassLowering {
                    class_id: class.class_id,
                    flags: 0,
                    tail_offset: class_tail_offset,
                    tail_count: 1,
                    copy_offset: class_copy_offset,
                    copy_count: class_copy_count,
                    write_offset: class_write_offset,
                    write_count: class_write_count,
                    reserved: 0,
                });
                #[cfg(test)]
                {
                    delta_entries_touched = delta_entries_touched
                        .checked_add(writes.len() as u64)
                        .expect("test instrumentation does not overflow");
                }
                class_deltas.push(ClassDelta {
                    class_id: class.class_id,
                    tail_action: tail_action.kind,
                    tail_source: previous_tail,
                    tail_destination,
                    copy_intent,
                    writes: writes.into(),
                });
            }
            let output = PreparedStep {
                step,
                request: item.request,
                base_snapshot,
                target_snapshot,
                base_view_version: snapshot.view_version,
                target_view_version: target_version,
                previous_boundary,
                target_boundary: item.target_boundary,
                class_lowerings: class_lowerings.into_boxed_slice(),
                tail_actions: tail_actions.into_boxed_slice(),
                copy_intents: copy_intents.into_boxed_slice(),
                write_intents: write_intents.into_boxed_slice(),
            };
            let delta = Arc::new(StepDelta {
                request: item.request,
                base_snapshot,
                target_snapshot,
                base_view_version: snapshot.view_version,
                target_view_version: target_version,
                previous_boundary,
                target_boundary: item.target_boundary,
                classes: class_deltas.into_boxed_slice(),
            });
            plans.push((
                planned_operation,
                planned_snapshot,
                planned_pages,
                PreparedState { delta },
                output,
            ));
        }

        for (planned_operation, planned_snapshot, planned_pages, prepared, _) in &plans {
            let step = StepLease {
                engine_epoch: self.engine_epoch,
                slot: planned_operation.0,
                generation: planned_operation.1,
            };
            self.apply_page_reservations(step, planned_pages);
            self.snapshots.insert_planned(
                *planned_snapshot,
                RequestSnapshot {
                    boundary: prepared.delta.target_boundary,
                    view_version: prepared.delta.target_view_version,
                    roots: (0..self.classes.len())
                        .map(|_| ClassRoot {
                            entries: PersistentRootEntries::default(),
                        })
                        .collect::<Vec<_>>()
                        .into(),
                },
            );
            self.operations.insert_planned(
                *planned_operation,
                OperationState::Prepared(prepared.clone()),
            );
            self.prepared_steps += 1;
            self.request_mut(prepared.delta.request)
                .expect("batch preflight retained the request")
                .pending_step = Some(step);
        }
        #[cfg(test)]
        {
            self.hot_path.delta_entries_touched += delta_entries_touched;
        }
        Ok(plans
            .into_iter()
            .map(|(_, _, _, _, output)| output)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically validates backend bindings and pins an ordered step batch.
    ///
    /// Receipt ranges must form one canonical, gap-free partition of
    /// `receipts` in item order. The authoritative request identity is derived
    /// from each step; callers cannot substitute it.
    ///
    /// # Errors
    ///
    /// Structural identity/range failures reject the whole batch without
    /// mutation. Once all steps are resolved, any semantic bind-receipt
    /// mismatch fail-stops every candidate in the batch: all reachable pages
    /// and requests are quarantined, so they cannot be aborted or reused.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn submit_batch(
        &mut self,
        items: &[SubmitBatchItem],
        receipts: &[BackendBindReceipt],
        copy_receipts: &[BackendCopyReceipt],
    ) -> Result<Box<[SubmittedStep]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_steps = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut expected_receipt_offset = 0_usize;
        let mut expected_copy_offset = 0_usize;
        let mut plans = Vec::with_capacity(items.len());

        for item in items {
            if !seen_steps.insert(item.step) {
                return Err(KvManagerError::DuplicateStep);
            }
            let receipt_offset = usize::try_from(item.receipt_offset)
                .map_err(|_| KvManagerError::InvalidBatchRange)?;
            let receipt_count = usize::try_from(item.receipt_count)
                .map_err(|_| KvManagerError::InvalidBatchRange)?;
            if receipt_offset != expected_receipt_offset {
                return Err(KvManagerError::InvalidBatchRange);
            }
            let receipt_end = receipt_offset
                .checked_add(receipt_count)
                .ok_or(KvManagerError::InvalidBatchRange)?;
            receipts
                .get(receipt_offset..receipt_end)
                .ok_or(KvManagerError::InvalidBatchRange)?;
            expected_receipt_offset = receipt_end;
            let copy_offset = usize::try_from(item.copy_receipt_offset)
                .map_err(|_| KvManagerError::InvalidBatchRange)?;
            let copy_count = usize::try_from(item.copy_receipt_count)
                .map_err(|_| KvManagerError::InvalidBatchRange)?;
            if copy_offset != expected_copy_offset {
                return Err(KvManagerError::InvalidBatchRange);
            }
            let copy_end = copy_offset
                .checked_add(copy_count)
                .ok_or(KvManagerError::InvalidBatchRange)?;
            copy_receipts
                .get(copy_offset..copy_end)
                .ok_or(KvManagerError::InvalidBatchRange)?;
            expected_copy_offset = copy_end;

            self.check_step_epoch(item.step)?;
            let prepared = match self.operations.get(item.step.slot, item.step.generation)? {
                OperationState::Prepared(prepared) => prepared.clone(),
                OperationState::Submitted(_) => return Err(KvManagerError::StepAlreadySubmitted),
            };
            if !seen_requests.insert(prepared.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            plans.push((
                item.step,
                prepared,
                receipt_offset,
                receipt_end,
                copy_offset,
                copy_end,
            ));
        }
        if expected_receipt_offset != receipts.len() || expected_copy_offset != copy_receipts.len()
        {
            return Err(KvManagerError::InvalidBatchRange);
        }

        let semantic_result = (|| {
            let mut seen_pages = BTreeSet::new();
            for (step, prepared, begin, end, copy_begin, copy_end) in &plans {
                Self::validate_bind_receipts(*step, prepared, &receipts[*begin..*end])?;
                Self::validate_copy_receipts(
                    *step,
                    prepared,
                    &copy_receipts[*copy_begin..*copy_end],
                )?;
                self.preflight_prepared_delta(prepared, *step)?;
                for class_delta in &prepared.delta.classes {
                    if class_delta
                        .tail_destination
                        .iter()
                        .chain(class_delta.writes.iter())
                        .any(|entry| !seen_pages.insert(entry.page.page_id))
                    {
                        return Err(KvManagerError::DuplicatePage);
                    }
                }
                let request_state = self.request(prepared.delta.request)?;
                let snapshot = self.request_snapshot(prepared.delta.request)?;
                if request_state.pending_step != Some(*step)
                    || snapshot.view_version != prepared.delta.base_view_version
                    || snapshot.boundary != prepared.delta.previous_boundary
                    || snapshot.roots.len() != prepared.delta.classes.len()
                {
                    return Err(KvManagerError::StaleView);
                }
            }
            Ok(())
        })();
        if let Err(error) = semantic_result {
            for (step, prepared, _, _, _, _) in &plans {
                let destinations = prepared
                    .delta
                    .classes
                    .iter()
                    .flat_map(|delta| {
                        delta
                            .tail_source
                            .filter(|_| delta.tail_action == TailActionKind::InPlace)
                            .into_iter()
                            .chain(delta.tail_destination.iter().copied())
                            .chain(delta.writes.iter().copied())
                            .map(|entry| entry.page.page_id)
                    })
                    .collect::<Vec<_>>();
                for page_id in destinations {
                    self.set_page_phase(page_id, PagePhase::Quarantined)
                        .expect("batch submit retained manager-selected destination");
                }
                self.operations
                    .remove(step.slot, step.generation)
                    .expect("batch submit preflight retained prepared operation");
                self.snapshots
                    .remove(
                        prepared.delta.target_snapshot.slot,
                        prepared.delta.target_snapshot.generation,
                    )
                    .expect("batch submit retained reserved target snapshot");
                self.prepared_steps -= 1;
                let request = self
                    .request_mut(prepared.delta.request)
                    .expect("batch submit preflight retained request");
                request.pending_step = None;
                request.quarantined = true;
            }
            return Err(KvManagerError::BatchQuarantined(Box::new(error)));
        }

        let plans = plans
            .into_iter()
            .map(|(step, prepared, _, _, _, _)| {
                let submission = SubmissionLease {
                    engine_epoch: step.engine_epoch,
                    slot: step.slot,
                    generation: step.generation,
                };
                let output = SubmittedStep {
                    submission,
                    request: prepared.delta.request,
                    target_snapshot: prepared.delta.target_snapshot,
                };
                (step, prepared, submission, output)
            })
            .collect::<Vec<_>>();

        #[cfg(test)]
        let mut delta_entries_touched = 0_u64;
        for (step, prepared, submission, _) in &plans {
            for class_delta in &prepared.delta.classes {
                if let Some(source) = class_delta.tail_source {
                    let page = self
                        .page_mut(source.page.page_id)
                        .expect("batch preflight validated tail source");
                    page.reader_pins += 1;
                    if class_delta.tail_action == TailActionKind::InPlace {
                        page.writer = Some(*submission);
                    }
                }
                if let Some(destination) = class_delta.tail_destination {
                    self.set_page_phase(destination.page.page_id, PagePhase::Live)
                        .expect("batch preflight retained tail destination");
                    let page = self
                        .page_mut(destination.page.page_id)
                        .expect("batch preflight validated tail destination");
                    page.reader_pins = 1;
                    page.writer = Some(*submission);
                }
                for entry in class_delta.writes.iter().copied() {
                    self.set_page_phase(entry.page.page_id, PagePhase::Live)
                        .expect("batch preflight retained reserved write");
                    let page = self
                        .page_mut(entry.page.page_id)
                        .expect("batch preflight validated reserved write");
                    page.reader_pins = 1;
                    page.writer = Some(*submission);
                }
                #[cfg(test)]
                {
                    delta_entries_touched += u64::from(class_delta.tail_source.is_some())
                        + u64::from(class_delta.tail_destination.is_some())
                        + class_delta.writes.len() as u64;
                }
            }
            *self
                .operations
                .get_mut(step.slot, step.generation)
                .expect("batch preflight retained operation") =
                OperationState::Submitted(SubmittedState {
                    delta: Arc::clone(&prepared.delta),
                });
            self.prepared_steps -= 1;
            self.submitted_steps += 1;
            let request = self
                .request_mut(prepared.delta.request)
                .expect("batch preflight retained request");
            request.pending_step = None;
            request.inflight_submission = Some(*submission);
        }
        #[cfg(test)]
        {
            self.hot_path.delta_entries_touched += delta_entries_touched;
        }
        Ok(plans
            .into_iter()
            .map(|(_, _, _, output)| output)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically publishes an ordered submission batch at one shared backend
    /// completion point.
    ///
    /// Submission identities are authoritative and derive their requests.
    /// Every root, page pin, retirement, and reclamation slot is preflighted
    /// before any publication occurs.
    ///
    /// # Errors
    ///
    /// Any invalid completion event or submission rejects the whole batch with
    /// no published view, page phase, operation, or reclamation mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn complete_batch(
        &mut self,
        receipt: BatchCompletionReceipt,
        submissions: &[SubmissionLease],
    ) -> Result<CompletionBatch, KvManagerError> {
        if submissions.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        if receipt.reserved != 0 {
            return Err(KvManagerError::ReservedFieldNonZero);
        }
        if receipt.confirmed != 1 {
            return Err(KvManagerError::CompletionNotConfirmed);
        }
        if receipt.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        #[cfg(test)]
        let root_instrumentation_before = root_instrumentation();

        let mut seen_submissions = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_destinations = BTreeSet::new();
        let mut prelim = Vec::with_capacity(submissions.len());
        for &submission in submissions {
            if !seen_submissions.insert(submission) {
                return Err(KvManagerError::DuplicateSubmission);
            }
            self.check_submission_epoch(submission)?;
            let submitted = match self
                .operations
                .get(submission.slot, submission.generation)?
            {
                OperationState::Submitted(submitted) => submitted.clone(),
                OperationState::Prepared(_) => return Err(KvManagerError::StepNotSubmitted),
            };
            let delta = &submitted.delta;
            if !seen_requests.insert(delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request_state = self.request(delta.request)?;
            let snapshot = self.request_snapshot(delta.request)?;
            if request_state.inflight_submission != Some(submission)
                || request_state.head != delta.base_snapshot
                || snapshot.view_version != delta.base_view_version
                || snapshot.boundary != delta.previous_boundary
                || snapshot.roots.len() != self.classes.len()
                || delta.classes.len() != self.classes.len()
            {
                return Err(KvManagerError::StaleView);
            }
            self.preflight_submitted_delta(submission, &submitted)?;
            for class_delta in &delta.classes {
                for entry in class_delta
                    .tail_destination
                    .iter()
                    .chain(class_delta.writes.iter())
                {
                    if !seen_destinations.insert(entry.page.page_id) {
                        return Err(KvManagerError::DuplicatePage);
                    }
                }
            }

            let mut transitions = Vec::with_capacity(self.classes.len());
            let mut retire_entries = Vec::new();
            let mut resident_count = 0_usize;
            for ((class, root), class_delta) in self
                .classes
                .iter()
                .copied()
                .zip(snapshot.roots.iter())
                .zip(delta.classes.iter())
            {
                if class_delta.class_id != class.class_id {
                    return Err(KvManagerError::Invariant("delta class ordering"));
                }
                if let (Some(front), Some(back)) = (root.entries.front(), root.entries.back()) {
                    let expected_len = back
                        .logical_ordinal
                        .checked_sub(front.logical_ordinal)
                        .and_then(|span| span.checked_add(1))
                        .and_then(|span| usize::try_from(span).ok())
                        .ok_or(KvManagerError::Invariant("snapshot root span"))?;
                    if front.class_id != class.class_id
                        || back.class_id != class.class_id
                        || expected_len != root.entries.len()
                    {
                        return Err(KvManagerError::Invariant("snapshot root continuity"));
                    }
                }
                if let (Some(front), Some(back)) =
                    (class_delta.writes.first(), class_delta.writes.last())
                {
                    let expected_len = back
                        .logical_ordinal
                        .checked_sub(front.logical_ordinal)
                        .and_then(|span| span.checked_add(1))
                        .and_then(|span| usize::try_from(span).ok())
                        .ok_or(KvManagerError::Invariant("delta write span"))?;
                    if front.class_id != class.class_id
                        || back.class_id != class.class_id
                        || expected_len != class_delta.writes.len()
                    {
                        return Err(KvManagerError::Invariant("delta write continuity"));
                    }
                }
                if let (Some(back), Some(front)) = (root.entries.back(), class_delta.writes.first())
                    && back.logical_ordinal.checked_add(1) != Some(front.logical_ordinal)
                {
                    return Err(KvManagerError::StaleView);
                }
                let candidate_first = root
                    .entries
                    .front()
                    .or(class_delta.tail_destination.as_ref())
                    .or_else(|| class_delta.writes.first())
                    .map(|entry| entry.logical_ordinal)
                    .ok_or(KvManagerError::Invariant("empty append candidate"))?;
                let expected_first =
                    class.candidate_start(delta.previous_boundary) / self.page_tokens;
                let candidate_end = delta.target_boundary.div_ceil(self.page_tokens);
                let candidate_len = root
                    .entries
                    .len()
                    .checked_add(usize::from(
                        class_delta.tail_action == TailActionKind::Fresh,
                    ))
                    .ok_or(KvManagerError::ArithmeticOverflow("candidate root length"))?
                    .checked_add(class_delta.writes.len())
                    .ok_or(KvManagerError::ArithmeticOverflow("candidate root length"))?;
                let expected_len = candidate_end
                    .checked_sub(candidate_first)
                    .and_then(|span| usize::try_from(span).ok())
                    .ok_or(KvManagerError::Invariant("candidate root span"))?;
                if candidate_first != expected_first || candidate_len != expected_len {
                    return Err(KvManagerError::StaleView);
                }
                let retain_first_ordinal = match class.retention {
                    RetentionKind::Full => candidate_first,
                    RetentionKind::Sliding => (class.retained_start(delta.target_boundary)
                        / self.page_tokens)
                        .min(candidate_end),
                    RetentionKind::Chunked => {
                        unreachable!("canonical profile rejects chunked retention")
                    }
                };
                let retire_count = usize::try_from(
                    retain_first_ordinal
                        .checked_sub(candidate_first)
                        .ok_or(KvManagerError::Invariant("retained root prefix"))?,
                )
                .map_err(|_| KvManagerError::ArithmeticOverflow("retirement count"))?;
                if retire_count > candidate_len {
                    return Err(KvManagerError::Invariant("retirement exceeds candidate"));
                }
                let retire_from_root = retire_count.min(root.entries.len());
                let retire_after_root = retire_count - retire_from_root;
                let retire_tail_destination = usize::from(
                    class_delta.tail_action == TailActionKind::Fresh && retire_after_root != 0,
                );
                let retire_from_writes = retire_after_root - retire_tail_destination;
                if retire_from_root != 0 {
                    for entry in root.entries.iter().take(retire_from_root) {
                        // A private partial tail can be both the submitted
                        // write target and outside the post-step SWA window.
                        // Its authoritative pin/writer witness was validated
                        // above; every other retiring root must be idle Live.
                        if class_delta.tail_source != Some(*entry) {
                            self.preflight_active_root_entry(delta.request, *entry)?;
                        }
                        retire_entries.push(*entry);
                    }
                }
                if retire_tail_destination != 0 {
                    retire_entries.push(
                        class_delta
                            .tail_destination
                            .ok_or(KvManagerError::Invariant("fresh tail destination"))?,
                    );
                }
                retire_entries.extend(class_delta.writes.iter().take(retire_from_writes).copied());
                let class_resident = candidate_len - retire_count;
                resident_count = resident_count.checked_add(class_resident).ok_or(
                    KvManagerError::ArithmeticOverflow("published resident count"),
                )?;
                transitions.push(ClassTransition {
                    retire_from_root,
                    retire_from_writes,
                    retain_first_ordinal,
                    resident_count: class_resident,
                });
            }
            let mut detached = retire_entries
                .iter()
                .copied()
                .map(|entry| {
                    let resident_boundary = if delta.classes.iter().any(|class| {
                        class.tail_action == TailActionKind::CopyOnWrite
                            && class.tail_source == Some(entry)
                    }) {
                        delta.previous_boundary
                    } else {
                        delta.target_boundary
                    };
                    self.clear_detached_binding(entry, resident_boundary, DetachedReason::Retention)
                })
                .collect::<Result<Vec<_>, KvManagerError>>()?;
            for (class_delta, transition) in delta.classes.iter().zip(transitions.iter()) {
                if class_delta.tail_action == TailActionKind::CopyOnWrite {
                    let source = class_delta
                        .tail_source
                        .ok_or(KvManagerError::Invariant("COW tail source"))?;
                    if source.logical_ordinal >= transition.retain_first_ordinal {
                        detached.push(
                            self.replace_detached_binding(
                                source,
                                class_delta
                                    .tail_destination
                                    .ok_or(KvManagerError::Invariant("COW tail destination"))?,
                                delta.previous_boundary,
                            )?,
                        );
                    }
                }
            }
            detached.sort_by_key(|binding| {
                (
                    binding.class_id,
                    binding.logical_ordinal,
                    binding.action as u16,
                    binding.old,
                )
            });
            prelim.push((
                submission,
                submitted,
                transitions,
                retire_entries,
                detached,
                resident_count,
            ));
        }
        let mut ref_deltas = BTreeMap::<PageLease, (i64, RootEntry, u64)>::new();
        let mut pin_decrements = BTreeMap::<PageLease, u32>::new();
        let mut candidates = BTreeMap::<PageLease, (RootEntry, u64)>::new();
        for (_, submitted, transitions, retire_entries, _, _) in &prelim {
            for entry in retire_entries {
                let resident_boundary = if submitted.delta.classes.iter().any(|class| {
                    class.tail_action == TailActionKind::CopyOnWrite
                        && class.tail_source == Some(*entry)
                }) {
                    submitted.delta.previous_boundary
                } else {
                    submitted.delta.target_boundary
                };
                self.insert_completion_candidate(&mut candidates, *entry, resident_boundary)?;
                if self.page(entry.page.page_id)?.request_refs != 0 {
                    ref_deltas
                        .entry(entry.page)
                        .or_insert((0, *entry, submitted.delta.target_boundary))
                        .0 -= 1;
                }
            }
            for (class_delta, transition) in submitted.delta.classes.iter().zip(transitions.iter())
            {
                if let Some(source) = class_delta.tail_source {
                    *pin_decrements.entry(source.page).or_default() += 1;
                    let resident_boundary =
                        if class_delta.tail_action == TailActionKind::CopyOnWrite {
                            submitted.delta.previous_boundary
                        } else {
                            submitted.delta.target_boundary
                        };
                    self.insert_completion_candidate(&mut candidates, source, resident_boundary)?;
                    if class_delta.tail_action == TailActionKind::CopyOnWrite
                        && source.logical_ordinal >= transition.retain_first_ordinal
                    {
                        ref_deltas
                            .entry(source.page)
                            .or_insert((0, source, submitted.delta.target_boundary))
                            .0 -= 1;
                    }
                }
                if let Some(destination) = class_delta.tail_destination {
                    *pin_decrements.entry(destination.page).or_default() += 1;
                    self.insert_completion_candidate(
                        &mut candidates,
                        destination,
                        submitted.delta.target_boundary,
                    )?;
                    if destination.logical_ordinal >= transition.retain_first_ordinal {
                        ref_deltas
                            .entry(destination.page)
                            .or_insert((0, destination, submitted.delta.target_boundary))
                            .0 += 1;
                    }
                }
                for entry in class_delta.writes.iter().copied() {
                    *pin_decrements.entry(entry.page).or_default() += 1;
                    self.insert_completion_candidate(
                        &mut candidates,
                        entry,
                        submitted.delta.target_boundary,
                    )?;
                    if entry.logical_ordinal >= transition.retain_first_ordinal {
                        ref_deltas
                            .entry(entry.page)
                            .or_insert((0, entry, submitted.delta.target_boundary))
                            .0 += 1;
                    }
                }
            }
        }
        let mut post_refs = BTreeMap::<PageLease, u32>::new();
        for (lease, (delta, _, _)) in &ref_deltas {
            let page = self.page(lease.page_id)?;
            let post = i64::from(page.request_refs)
                .checked_add(*delta)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(KvManagerError::Invariant("request ref delta"))?;
            post_refs.insert(*lease, post);
        }
        for (lease, decrement) in &pin_decrements {
            if self.page(lease.page_id)?.reader_pins < *decrement {
                return Err(KvManagerError::StalePage);
            }
        }
        let mut retiring = Vec::new();
        for (lease, (entry, boundary)) in &candidates {
            let page = self.page(lease.page_id)?;
            let request_refs = post_refs.get(lease).copied().unwrap_or(page.request_refs);
            let final_pins = page
                .reader_pins
                .checked_sub(pin_decrements.get(lease).copied().unwrap_or(0))
                .ok_or(KvManagerError::StalePage)?;
            if request_refs == 0 && page.prefix_refs == 0 && final_pins == 0 {
                retiring.push((*entry, *boundary));
            }
        }
        retiring.sort_by_key(|(entry, _)| (entry.class_id, entry.logical_ordinal, entry.page));
        let planned_reclamations = self.reclamations.plan_many(retiring.len())?;
        let certificates = retiring
            .iter()
            .zip(planned_reclamations.iter().copied())
            .map(|((entry, boundary), planned)| {
                self.certificate_for_root(
                    *entry,
                    *boundary,
                    planned,
                    receipt.completion_domain,
                    receipt.completion_value,
                )
            })
            .collect::<Result<Vec<_>, KvManagerError>>()?;
        let mut plans = Vec::with_capacity(prelim.len());
        for (submission, submitted, transitions, retire_entries, detached, resident_count) in prelim
        {
            let base_snapshot = self
                .snapshots
                .get(
                    submitted.delta.base_snapshot.slot,
                    submitted.delta.base_snapshot.generation,
                )
                .expect("batch completion preflight retained base snapshot");
            let mut candidate_roots = base_snapshot.roots.iter().cloned().collect::<Vec<_>>();
            for ((root, class_delta), transition) in candidate_roots
                .iter_mut()
                .zip(submitted.delta.classes.iter())
                .zip(transitions.iter())
            {
                for _ in 0..transition.retire_from_root {
                    root.entries
                        .pop_front()
                        .expect("completion preflight validated root retirement");
                }
                if let Some(destination) = class_delta.tail_destination
                    && destination.logical_ordinal >= transition.retain_first_ordinal
                {
                    if class_delta.tail_action == TailActionKind::CopyOnWrite {
                        let source = class_delta
                            .tail_source
                            .expect("COW completion retained its source");
                        let removed = root
                            .entries
                            .pop_back()
                            .expect("COW completion preflight retained its tail");
                        debug_assert_eq!(removed, source);
                    }
                    root.entries.push_back(destination);
                }
                root.entries.extend(
                    class_delta
                        .writes
                        .iter()
                        .skip(transition.retire_from_writes)
                        .copied(),
                );
                debug_assert_eq!(root.entries.len(), transition.resident_count);
            }
            let resident_count = u32::try_from(resident_count)
                .map_err(|_| KvManagerError::ArithmeticOverflow("published resident count"))?;
            let output = StepCompletion {
                submission,
                request: submitted.delta.request,
                detached_snapshot: submitted.delta.base_snapshot,
                publication: PublishedReceipt {
                    snapshot: submitted.delta.target_snapshot,
                    view_version: submitted.delta.target_view_version,
                    boundary: submitted.delta.target_boundary,
                    resident_count,
                },
                detached: detached.into_boxed_slice(),
            };
            plans.push((
                submission,
                submitted,
                transitions,
                retire_entries,
                Arc::<[ClassRoot]>::from(candidate_roots),
                output,
            ));
        }

        #[cfg(test)]
        let mut delta_entries_touched = 0_u64;
        #[cfg(test)]
        let mut retirement_entries_touched = 0_u64;
        #[cfg(test)]
        let mut hot_root_entries_visited = 0_u64;
        // First discharge every submitted reader/writer witness. Shared COW
        // sources may occur more than once in this batch, so their pin count
        // is decremented once per authoritative operation rather than once per
        // distinct page.
        for (_, submitted, _, _, _, _) in &plans {
            for class_delta in &submitted.delta.classes {
                if let Some(source) = class_delta.tail_source {
                    let page = self
                        .page_mut(source.page.page_id)
                        .expect("batch preflight validated submitted tail source");
                    page.reader_pins -= 1;
                    if class_delta.tail_action == TailActionKind::InPlace {
                        page.writer = None;
                    }
                    page.completion_domain = receipt.completion_domain;
                    page.completion_value = receipt.completion_value;
                }
                if let Some(destination) = class_delta.tail_destination {
                    let page = self
                        .page_mut(destination.page.page_id)
                        .expect("batch preflight validated submitted tail destination");
                    page.reader_pins -= 1;
                    page.writer = None;
                    page.completion_domain = receipt.completion_domain;
                    page.completion_value = receipt.completion_value;
                }
                for entry in class_delta.writes.iter() {
                    let page = self
                        .page_mut(entry.page.page_id)
                        .expect("batch preflight validated submitted write");
                    page.reader_pins -= 1;
                    page.writer = None;
                    page.completion_domain = receipt.completion_domain;
                    page.completion_value = receipt.completion_value;
                }
                #[cfg(test)]
                {
                    delta_entries_touched += u64::from(class_delta.tail_source.is_some())
                        + u64::from(class_delta.tail_destination.is_some())
                        + class_delta.writes.len() as u64;
                }
            }
        }

        // Apply the batch's aggregated request-reference transaction exactly
        // once per physical page. This is what makes duplicate PageLease values
        // across requests legal while keeping shared detach/certification exact.
        for (lease, request_refs) in &post_refs {
            self.page_mut(lease.page_id)
                .expect("batch preflight retained referenced page")
                .request_refs = *request_refs;
        }
        for ((entry, _), (planned, certificate)) in retiring.iter().zip(
            planned_reclamations
                .iter()
                .copied()
                .zip(certificates.iter()),
        ) {
            let page = self
                .page(entry.page.page_id)
                .expect("batch preflight validated retiring page");
            debug_assert_eq!(page.reader_pins, 0);
            debug_assert_eq!(page.writer, None);
            debug_assert_eq!(page.request_refs, 0);
            debug_assert_eq!(page.prefix_refs, 0);
            self.set_page_phase(
                entry.page.page_id,
                PagePhase::Retiring {
                    reclamation: certificate.reclamation,
                },
            )
            .expect("batch preflight retained retiring page");
            self.reclamations.insert_planned(
                planned,
                ReclamationState {
                    certificate: certificate.clone(),
                },
            );
        }

        for (submission, submitted, _, _, candidate_roots, _) in &plans {
            {
                let mut snapshot = self
                    .snapshots
                    .remove(
                        submitted.delta.base_snapshot.slot,
                        submitted.delta.base_snapshot.generation,
                    )
                    .expect("batch preflight retained base snapshot");
                snapshot.boundary = submitted.delta.target_boundary;
                snapshot.view_version = submitted.delta.target_view_version;
                snapshot.roots = Arc::clone(candidate_roots);
                *self
                    .snapshots
                    .get_mut(
                        submitted.delta.target_snapshot.slot,
                        submitted.delta.target_snapshot.generation,
                    )
                    .expect("batch preflight retained target snapshot") = snapshot;
            }
            let request = self
                .request_mut(submitted.delta.request)
                .expect("batch preflight retained request");
            request.head = submitted.delta.target_snapshot;
            request.inflight_submission = None;
            request.last_completion_domain = receipt.completion_domain;
            request.last_completion_value = receipt.completion_value;
            self.operations
                .remove(submission.slot, submission.generation)
                .expect("batch preflight retained submitted operation");
            self.submitted_steps -= 1;
        }
        #[cfg(test)]
        {
            for (_, _, transitions, retire_entries, _, _) in &plans {
                retirement_entries_touched += retire_entries.len() as u64;
                hot_root_entries_visited += transitions
                    .iter()
                    .map(|transition| transition.retire_from_root as u64)
                    .sum::<u64>();
            }
            let root_instrumentation_after = root_instrumentation();
            self.hot_path.delta_entries_touched += delta_entries_touched;
            self.hot_path.retirement_entries_touched += retirement_entries_touched;
            self.hot_path.hot_root_entries_visited += hot_root_entries_visited;
            self.hot_path.root_node_visits +=
                root_instrumentation_after.0 - root_instrumentation_before.0;
            self.hot_path.root_iterator_allocs +=
                root_instrumentation_after.1 - root_instrumentation_before.1;
            self.hot_path.path_nodes_cloned +=
                root_instrumentation_after.2 - root_instrumentation_before.2;
        }
        Ok(CompletionBatch {
            completions: plans
                .into_iter()
                .map(|(_, _, _, _, _, output)| output)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            retirements: certificates.into_boxed_slice(),
        })
    }

    /// Atomically aborts a non-empty prepared batch proven backend-unobserved.
    ///
    /// # Errors
    ///
    /// Any missing proof, duplicate, stale step, or stale page rejects the
    /// whole batch without mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn abort_steps_batch(
        &mut self,
        receipts: &[BackendUnobservedReceipt],
    ) -> Result<(), KvManagerError> {
        if receipts.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_steps = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(receipts.len());
        for &receipt in receipts {
            if receipt.reserved != 0 {
                return Err(KvManagerError::ReservedFieldNonZero);
            }
            if receipt.backend_unobserved != 1 {
                return Err(KvManagerError::BackendObservationUnknown);
            }
            if !seen_steps.insert(receipt.step) {
                return Err(KvManagerError::DuplicateStep);
            }
            self.check_step_epoch(receipt.step)?;
            let prepared = match self
                .operations
                .get(receipt.step.slot, receipt.step.generation)?
            {
                OperationState::Prepared(prepared) => prepared.clone(),
                OperationState::Submitted(_) => return Err(KvManagerError::StepAlreadySubmitted),
            };
            if !seen_requests.insert(prepared.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(prepared.delta.request)?;
            if request.pending_step != Some(receipt.step) {
                return Err(KvManagerError::StaleView);
            }
            let reserved = prepared
                .delta
                .classes
                .iter()
                .flat_map(|class| class.tail_destination.iter().chain(class.writes.iter()))
                .map(|entry| entry.page.page_id)
                .collect::<Vec<_>>();
            for &page_id in &reserved {
                if !seen_pages.insert(page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(page_id)?;
                if page.phase != (PagePhase::Reserved { step: receipt.step })
                    || page.request_refs != 0
                    || page.prefix_refs != 0
                    || page.reader_pins != 0
                    || page.writer.is_some()
                {
                    return Err(KvManagerError::StalePage);
                }
            }
            plans.push((
                receipt.step,
                prepared.delta.request,
                prepared.delta.target_snapshot,
                reserved,
            ));
        }
        let mut recycled_by_class = vec![Vec::<u32>::new(); self.classes.len()];
        for (step, request, target_snapshot, reserved) in plans {
            for page_id in reserved {
                let (class_id, generation) = {
                    let page = self
                        .page(page_id)
                        .expect("batch abort preflight retained reserved page");
                    (page.class_id, page.generation)
                };
                if generation == u64::MAX {
                    self.set_page_phase(page_id, PagePhase::Exhausted)
                        .expect("batch abort preflight retained reserved page");
                } else {
                    self.set_page_phase(page_id, PagePhase::Free)
                        .expect("batch abort preflight retained reserved page");
                    recycled_by_class[usize::from(class_id)].push(page_id);
                }
            }
            self.operations
                .remove(step.slot, step.generation)
                .expect("batch abort preflight retained operation");
            self.snapshots
                .remove(target_snapshot.slot, target_snapshot.generation)
                .expect("batch abort preflight retained target snapshot");
            self.prepared_steps -= 1;
            self.request_mut(request)
                .expect("batch abort preflight retained request")
                .pending_step = None;
        }
        for (free, mut recycled) in self.free_pages.iter_mut().zip(recycled_by_class) {
            recycled.sort_unstable_by(|left, right| right.cmp(left));
            free.extend(recycled);
        }
        Ok(())
    }

    /// Atomically fail-stops an ordered prepared batch after ambiguous backend
    /// lowering.
    ///
    /// # Errors
    ///
    /// Any duplicate, stale, or submitted identity rejects the whole call
    /// before quarantine begins.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn quarantine_steps_batch(&mut self, steps: &[StepLease]) -> Result<(), KvManagerError> {
        if steps.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_steps = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_pages = BTreeSet::new();
        let mut plans = Vec::with_capacity(steps.len());
        for &step in steps {
            if !seen_steps.insert(step) {
                return Err(KvManagerError::DuplicateStep);
            }
            self.check_step_epoch(step)?;
            let prepared = match self.operations.get(step.slot, step.generation)? {
                OperationState::Prepared(prepared) => prepared.clone(),
                OperationState::Submitted(_) => return Err(KvManagerError::StepAlreadySubmitted),
            };
            if !seen_requests.insert(prepared.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(prepared.delta.request)?;
            if request.pending_step != Some(step) {
                return Err(KvManagerError::StaleView);
            }
            let affected = prepared
                .delta
                .classes
                .iter()
                .flat_map(|class| {
                    class
                        .tail_source
                        .filter(|_| class.tail_action == TailActionKind::InPlace)
                        .map(|entry| (entry.page.page_id, false))
                        .into_iter()
                        .chain(
                            class
                                .tail_destination
                                .iter()
                                .chain(class.writes.iter())
                                .map(|entry| (entry.page.page_id, true)),
                        )
                })
                .collect::<Vec<_>>();
            for &(page_id, reserved) in &affected {
                if !seen_pages.insert(page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                let page = self.page(page_id)?;
                let valid = if reserved {
                    page.phase == (PagePhase::Reserved { step })
                        && page.request_refs == 0
                        && page.prefix_refs == 0
                        && page.reader_pins == 0
                        && page.writer.is_none()
                } else {
                    page.phase == PagePhase::Live
                        && page.request_refs == 1
                        && page.prefix_refs == 0
                        && page.reader_pins == 0
                        && page.writer.is_none()
                };
                if !valid {
                    return Err(KvManagerError::StalePage);
                }
            }
            plans.push((
                step,
                prepared.delta.request,
                prepared.delta.target_snapshot,
                affected,
            ));
        }
        for (step, request, target_snapshot, affected) in plans {
            for (page_id, _) in affected {
                self.set_page_phase(page_id, PagePhase::Quarantined)
                    .expect("batch quarantine preflight retained page");
            }
            self.operations
                .remove(step.slot, step.generation)
                .expect("batch quarantine preflight retained operation");
            self.snapshots
                .remove(target_snapshot.slot, target_snapshot.generation)
                .expect("batch quarantine preflight retained target snapshot");
            self.prepared_steps -= 1;
            let request = self
                .request_mut(request)
                .expect("batch quarantine preflight retained request");
            request.pending_step = None;
            request.quarantined = true;
        }
        Ok(())
    }

    /// Atomically fail-stops every page reachable by an ordered ambiguous
    /// submission batch.
    ///
    /// # Errors
    ///
    /// Any duplicate, stale, or unsubmitted identity rejects the whole call
    /// before quarantine begins.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    pub fn quarantine_submissions_batch(
        &mut self,
        submissions: &[SubmissionLease],
    ) -> Result<(), KvManagerError> {
        if submissions.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_submissions = BTreeSet::new();
        let mut seen_requests = BTreeSet::new();
        let mut seen_destinations = BTreeSet::new();
        let mut plans = Vec::with_capacity(submissions.len());
        for &submission in submissions {
            if !seen_submissions.insert(submission) {
                return Err(KvManagerError::DuplicateSubmission);
            }
            self.check_submission_epoch(submission)?;
            let submitted = match self
                .operations
                .get(submission.slot, submission.generation)?
            {
                OperationState::Submitted(submitted) => submitted.clone(),
                OperationState::Prepared(_) => return Err(KvManagerError::StepNotSubmitted),
            };
            if !seen_requests.insert(submitted.delta.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
            let request = self.request(submitted.delta.request)?;
            if request.inflight_submission != Some(submission) {
                return Err(KvManagerError::StaleView);
            }
            self.preflight_submitted_delta(submission, &submitted)?;
            let destinations = submitted
                .delta
                .classes
                .iter()
                .flat_map(|class| {
                    class
                        .tail_source
                        .filter(|_| class.tail_action == TailActionKind::InPlace)
                        .into_iter()
                        .chain(class.tail_destination.iter().copied())
                        .chain(class.writes.iter().copied())
                        .map(|entry| entry.page.page_id)
                })
                .collect::<Vec<_>>();
            for &page_id in &destinations {
                if !seen_destinations.insert(page_id) {
                    return Err(KvManagerError::DuplicatePage);
                }
                self.page(page_id)?;
            }
            plans.push((
                submission,
                submitted.delta.request,
                submitted.delta.target_snapshot,
                destinations,
            ));
        }
        for (submission, request, target_snapshot, destinations) in plans {
            // Only pages that may have been modified are fail-stopped. COW
            // sources remain Live (and conservatively pinned) so an ambiguous
            // copy can never quarantine or recycle a shared source page.
            for page_id in destinations {
                self.set_page_phase(page_id, PagePhase::Quarantined)
                    .expect("batch quarantine preflight retained page");
            }
            let request = self
                .request_mut(request)
                .expect("batch quarantine preflight retained request");
            request.inflight_submission = None;
            request.quarantined = true;
            self.operations
                .remove(submission.slot, submission.generation)
                .expect("batch quarantine preflight retained operation");
            self.snapshots
                .remove(target_snapshot.slot, target_snapshot.generation)
                .expect("batch quarantine preflight retained target snapshot");
            self.submitted_steps -= 1;
        }
        Ok(())
    }
}
