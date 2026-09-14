use super::{
    BTreeSet, BackendUnobservedReceipt, CanonicalKvManager, KvManagerError, OperationState,
    PagePhase, StepLease, TailActionKind,
};

impl CanonicalKvManager {
    /// Atomically aborts a non-empty prepared batch proven backend-unobserved.
    ///
    /// # Errors
    ///
    /// Any missing proof, duplicate, stale step, or stale page rejects the
    /// whole batch without mutation.
    ///
    #[allow(clippy::missing_panics_doc)]
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
    #[allow(clippy::missing_panics_doc)]
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
}
