use serde::Serialize;
use std::collections::BTreeSet;

use crate::kv_manager::{
    BatchCompletionReceipt, ClassTokenDispositionUpdate, KvManagerError, PageLease,
    PrepareRelocationItem, PreparedRelocation, ReclamationCertificate, RelocationCopyReceipt,
    RelocationPolicy, RelocationUnobservedReceipt, SubmittedRelocation, TokenDisposition,
    TokenLocation, TokenMove, TokenPlacement, TokenViewQuery, ViewVersion,
};

use super::{
    EngineCompletionEvidence, EngineRelocationId, EngineRequestId, EngineRequestView,
    EngineRetirement, EngineRetirementEvidence, RequestPhase, RuntimeSession, RuntimeSessionError,
    engine_retirements, validate_reclamation_evidence,
};

/// One request/class query against the session-owned current request head.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineTokenViewQuery {
    pub request_id: EngineRequestId,
    pub class_id: u16,
    pub expected_boundary: u64,
}

/// A logical token view that contains no request or snapshot capability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineTokenView {
    pub request_id: EngineRequestId,
    pub class_id: u16,
    pub version: ViewVersion,
    pub page_tokens: u32,
    pub placements: Box<[TokenPlacement]>,
}

/// One capability-free semantic disposition update.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineTokenDispositionUpdate {
    pub class_id: u16,
    pub token_id: u64,
    pub disposition: TokenDisposition,
}

/// Atomic disposition updates for one session-owned request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineTokenDispositionBatchItem {
    pub request_id: EngineRequestId,
    pub updates: Box<[EngineTokenDispositionUpdate]>,
}

/// One request in a session relocation prepare batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePrepareRelocationItem {
    pub request_id: EngineRequestId,
    pub class_id: u16,
    pub policy: RelocationPolicy,
}

/// Exact copy plan for one request, with manager operation capabilities removed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationPlan {
    pub request_id: EngineRequestId,
    pub class_id: u16,
    pub base_version: ViewVersion,
    pub target_version: ViewVersion,
    pub fragmentation_milli: u16,
    pub source_pages: Box<[PageLease]>,
    pub destination_pages: Box<[PageLease]>,
    pub moves: Box<[TokenMove]>,
    pub projected_reclaimed_pages: u32,
}

/// Prepared relocation batch identified only by a session-minted opaque id.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePreparedRelocation {
    pub relocation_id: EngineRelocationId,
    pub plans: Box<[EngineRelocationPlan]>,
}

/// Engine-observed copy evidence for one planned token movement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationCopyEvidence {
    pub token_id: u64,
    pub source: TokenLocation,
    pub destination: TokenLocation,
    pub observed: bool,
    pub copied: bool,
}

/// Ordered copy evidence for one request in a relocation batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationRequestEvidence {
    pub request_id: EngineRequestId,
    pub copies: Box<[EngineRelocationCopyEvidence]>,
}

/// Exact copy evidence for one opaque relocation batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationExecutionEvidence {
    pub relocation_id: EngineRelocationId,
    pub requests: Box<[EngineRelocationRequestEvidence]>,
}

/// Ordered proof that a prepared request relocation was not backend-observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationAbortEvidence {
    pub request_id: EngineRequestId,
    pub backend_unobserved: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationTicket {
    relocation_id: EngineRelocationId,
}

impl EngineRelocationTicket {
    #[must_use]
    pub const fn relocation_id(&self) -> EngineRelocationId {
        self.relocation_id
    }
}

/// Published request head after confirmed relocation copy completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationRequestPublication {
    pub request_id: EngineRequestId,
    pub view_version: ViewVersion,
    pub boundary: u64,
    pub resident_count: u32,
}

/// Relocation publication gated on mirror cleanup and retirement ACK.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationPublication {
    pub relocation_id: EngineRelocationId,
    pub requests: Box<[EngineRelocationRequestPublication]>,
    pub retirements: Box<[EngineRetirement]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EngineRelocationPublicationEvidence {
    pub relocation_id: EngineRelocationId,
    pub mirror_cleanup_confirmed: bool,
    pub reclamation_receipts: Box<[EngineRetirementEvidence]>,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedState {
    requests: Box<[EngineRequestId]>,
    relocations: Box<[PreparedRelocation]>,
}

#[derive(Clone, Debug)]
pub(super) struct SubmittedState {
    requests: Box<[EngineRequestId]>,
    relocations: Box<[SubmittedRelocation]>,
}

#[derive(Clone, Debug)]
pub(super) struct PublicationState {
    requests: Box<[EngineRequestId]>,
    retirements: Box<[ReclamationCertificate]>,
}

#[derive(Clone, Debug)]
pub(super) enum PendingRelocation {
    Prepared(PreparedState),
    Submitted(SubmittedState),
    PublicationPending(PublicationState),
}

impl PendingRelocation {
    fn requests(&self) -> &[EngineRequestId] {
        match self {
            Self::Prepared(state) => &state.requests,
            Self::Submitted(state) => &state.requests,
            Self::PublicationPending(state) => &state.requests,
        }
    }
}

impl RuntimeSession {
    /// Reads logical token views through current private session heads.
    ///
    /// All named requests must be Ready. The returned values expose neither
    /// request nor snapshot leases.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate, unknown, non-ready, stale, or invalid queries.
    pub fn token_views_batch(
        &mut self,
        queries: &[EngineTokenViewQuery],
    ) -> Result<Box<[EngineTokenView]>, RuntimeSessionError> {
        self.ensure_healthy()?;
        if queries.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        let mut records = Vec::with_capacity(queries.len());
        for query in queries {
            if !seen.insert((query.request_id, query.class_id)) {
                return Err(RuntimeSessionError::DuplicateRequest(query.request_id));
            }
            let record = self.ready_request(query.request_id)?.clone();
            if record.view.boundary != query.expected_boundary {
                return Err(RuntimeSessionError::TokenViewBoundary {
                    request_id: query.request_id,
                    expected: query.expected_boundary,
                    actual: record.view.boundary,
                });
            }
            records.push(record);
        }
        let manager_queries = queries
            .iter()
            .zip(&records)
            .map(|(query, record)| TokenViewQuery {
                request: record.view.request,
                expected_snapshot: record.view.snapshot,
                class_id: query.class_id,
            })
            .collect::<Vec<_>>();
        let views = self.manager.token_views_batch(&manager_queries)?;
        #[cfg(any(test, feature = "test-support"))]
        let views = if self.test_fault == Some(super::RuntimeSessionTestFault::TokenViewOrdering) {
            let mut views = views;
            self.test_fault = None;
            if let Some(view) = views.first_mut() {
                view.class_id = view.class_id.wrapping_add(1);
            }
            views
        } else {
            views
        };
        if views.len() != queries.len()
            || views
                .iter()
                .zip(queries.iter().zip(&records))
                .any(|(view, (query, record))| {
                    view.class_id != query.class_id || view.version != record.view.view_version
                })
        {
            return Err(self.poison("token view result ordering"));
        }
        Ok(queries
            .iter()
            .zip(views)
            .map(|(query, view)| EngineTokenView {
                request_id: query.request_id,
                class_id: view.class_id,
                version: view.version,
                page_tokens: view.page_tokens,
                placements: view.placements,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically marks token dispositions and advances session-owned heads.
    ///
    /// # Errors
    ///
    /// Rejects invalid batches, non-ready requests, stale heads, or malformed
    /// disposition evidence without partially changing the session view.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map changes after full preflight.
    pub fn mark_token_dispositions_batch(
        &mut self,
        items: &[EngineTokenDispositionBatchItem],
    ) -> Result<Box<[EngineRequestView]>, RuntimeSessionError> {
        self.ensure_healthy()?;
        let request_ids = items.iter().map(|item| item.request_id).collect::<Vec<_>>();
        let records = self.preflight_ready_requests(&request_ids)?;
        let manager_items = items
            .iter()
            .zip(&records)
            .map(
                |(item, record)| crate::kv_manager::TokenDispositionBatchItem {
                    request: record.view.request,
                    expected_snapshot: record.view.snapshot,
                    updates: item
                        .updates
                        .iter()
                        .map(|update| ClassTokenDispositionUpdate {
                            class_id: update.class_id,
                            token_id: update.token_id,
                            disposition: update.disposition,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                },
            )
            .collect::<Vec<_>>();
        let updated = self.manager.mark_token_dispositions_batch(&manager_items)?;
        if updated.len() != records.len()
            || updated.iter().zip(&records).any(|(view, record)| {
                view.request != record.view.request
                    || view.boundary != record.view.boundary
                    || view.resident_count != record.view.resident_count
            })
        {
            return Err(self.poison("token disposition result ordering"));
        }
        for (&request_id, view) in request_ids.iter().zip(updated.iter().copied()) {
            self.requests
                .get_mut(&request_id)
                .expect("disposition preflight retained request")
                .view = view;
        }
        Ok(request_ids
            .into_iter()
            .zip(updated)
            .map(|(request_id, view)| super::engine_view(request_id, view))
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Atomically prepares full-evacuation relocation plans from private heads.
    ///
    /// # Errors
    ///
    /// Rejects invalid batches, non-ready requests, unsupported/shared roots,
    /// insufficient headroom, or manager capacity exhaustion atomically.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map changes after full preflight.
    pub fn prepare_relocation_batch(
        &mut self,
        items: &[EnginePrepareRelocationItem],
    ) -> Result<EnginePreparedRelocation, RuntimeSessionError> {
        self.ensure_healthy()?;
        let request_ids = items.iter().map(|item| item.request_id).collect::<Vec<_>>();
        let records = self.preflight_ready_requests(&request_ids)?;
        let relocation_sequence = self.next_relocation_sequence;
        let next_relocation_sequence = relocation_sequence
            .checked_add(1)
            .ok_or(RuntimeSessionError::IdentityExhausted("relocation"))?;
        let relocation_id = EngineRelocationId::from_parts(self.session_epoch, relocation_sequence);
        let manager_items = items
            .iter()
            .zip(&records)
            .map(|(item, record)| PrepareRelocationItem {
                request: record.view.request,
                expected_snapshot: record.view.snapshot,
                class_id: item.class_id,
                policy: item.policy.clone(),
            })
            .collect::<Vec<_>>();
        let prepared = self.manager.prepare_relocation_batch(&manager_items)?;
        if prepared.len() != items.len()
            || prepared
                .iter()
                .zip(items.iter().zip(&records))
                .any(|(prepared, (item, record))| {
                    prepared.request != record.view.request
                        || prepared.base_snapshot != record.view.snapshot
                        || prepared.plan.class_id != item.class_id
                        || prepared.plan.base_version != record.view.view_version
                })
        {
            return Err(self.poison("relocation prepare result ordering"));
        }
        self.next_relocation_sequence = next_relocation_sequence;
        let plans = request_ids
            .iter()
            .copied()
            .zip(prepared.iter())
            .map(|(request_id, prepared)| engine_relocation_plan(request_id, prepared))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let requests = request_ids.into_boxed_slice();
        for request_id in &requests {
            self.requests
                .get_mut(request_id)
                .expect("relocation preflight retained request")
                .phase = RequestPhase::RelocationPrepared(relocation_id);
        }
        self.relocations.insert(
            relocation_id,
            PendingRelocation::Prepared(PreparedState {
                requests,
                relocations: prepared,
            }),
        );
        Ok(EnginePreparedRelocation {
            relocation_id,
            plans,
        })
    }

    /// Aborts a prepared relocation only with exact ordered unobserved proof.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, non-prepared, reordered, incomplete, or observed
    /// relocations without releasing reservations.
    pub fn abort_prepared_relocation(
        &mut self,
        relocation_id: EngineRelocationId,
        evidence: &[EngineRelocationAbortEvidence],
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        let state = self.prepared_relocation(relocation_id)?.clone();
        validate_request_evidence(&state.requests, evidence, |item| item.request_id)?;
        self.preflight_request_phases(
            &state.requests,
            RequestPhase::RelocationPrepared(relocation_id),
        )?;
        let receipts = state
            .relocations
            .iter()
            .zip(evidence)
            .map(|(prepared, evidence)| RelocationUnobservedReceipt {
                relocation: prepared.relocation,
                backend_unobserved: u32::from(evidence.backend_unobserved),
                reserved: 0,
            })
            .collect::<Vec<_>>();
        self.manager.abort_relocations_batch(&receipts)?;
        self.finish_relocation(relocation_id, &state.requests, RequestPhase::Ready);
        Ok(())
    }

    /// Permanently gates a relocation whose backend observation is ambiguous.
    ///
    /// Canonical relocation and page reservations remain intentionally
    /// unreusable until this session is dropped. The session is sticky
    /// fail-stopped, so no subsequent operation can observe uncertain state.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, or internally inconsistent relocation state.
    pub fn quarantine_relocation(
        &mut self,
        relocation_id: EngineRelocationId,
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_relocation_epoch(relocation_id)?;
        let pending = self
            .relocations
            .get(&relocation_id)
            .cloned()
            .ok_or_else(|| self.relocation_id_error(relocation_id))?;
        let expected = match pending {
            PendingRelocation::Prepared(_) => RequestPhase::RelocationPrepared(relocation_id),
            PendingRelocation::Submitted(_) => RequestPhase::RelocationSubmitted(relocation_id),
            PendingRelocation::PublicationPending(_) => {
                RequestPhase::RelocationPublicationPending(relocation_id)
            }
        };
        self.preflight_request_phases(pending.requests(), expected)?;
        self.finish_relocation(relocation_id, pending.requests(), RequestPhase::Quarantined);
        Err(self.poison("relocation outcome is ambiguous"))
    }

    /// Submits exact ordered copy evidence while keeping raw leases private.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, non-prepared, reordered, incomplete, or invalid
    /// evidence. A canonical semantic mismatch quarantines the whole batch.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map changes after full preflight.
    pub fn submit_relocation(
        &mut self,
        evidence: &EngineRelocationExecutionEvidence,
    ) -> Result<EngineRelocationTicket, RuntimeSessionError> {
        self.ensure_healthy()?;
        let state = self.prepared_relocation(evidence.relocation_id)?.clone();
        validate_request_evidence(&state.requests, &evidence.requests, |item| item.request_id)?;
        self.preflight_request_phases(
            &state.requests,
            RequestPhase::RelocationPrepared(evidence.relocation_id),
        )?;
        let mut receipts = Vec::new();
        for (prepared, request_evidence) in state.relocations.iter().zip(&evidence.requests) {
            if request_evidence.copies.len() != prepared.plan.moves.len() {
                return Err(RuntimeSessionError::EvidenceCardinality {
                    field: "relocation copies",
                    expected: prepared.plan.moves.len(),
                    actual: request_evidence.copies.len(),
                });
            }
            receipts.extend(
                request_evidence
                    .copies
                    .iter()
                    .map(|copy| RelocationCopyReceipt {
                        relocation: prepared.relocation,
                        token_id: copy.token_id,
                        source: copy.source,
                        destination: copy.destination,
                        observed: u8::from(copy.observed),
                        copied: u8::from(copy.copied),
                        reserved16: 0,
                        reserved32: 0,
                    }),
            );
        }
        let leases = state
            .relocations
            .iter()
            .map(|prepared| prepared.relocation)
            .collect::<Vec<_>>();
        let submitted = match self.manager.submit_relocation_batch(&leases, &receipts) {
            Ok(submitted) => submitted,
            Err(error @ KvManagerError::BatchQuarantined(_)) => {
                self.finish_relocation(
                    evidence.relocation_id,
                    &state.requests,
                    RequestPhase::Quarantined,
                );
                self.poisoned.get_or_insert("relocation batch quarantined");
                return Err(error.into());
            }
            Err(error) => return Err(error.into()),
        };
        if submitted.len() != state.relocations.len()
            || submitted
                .iter()
                .zip(&state.relocations)
                .any(|(submitted, prepared)| {
                    submitted.relocation != prepared.relocation
                        || submitted.request != prepared.request
                        || submitted.target_snapshot != prepared.target_snapshot
                })
        {
            return Err(self.poison("relocation submit result ordering"));
        }
        for request_id in &state.requests {
            self.requests
                .get_mut(request_id)
                .expect("relocation submit retained request")
                .phase = RequestPhase::RelocationSubmitted(evidence.relocation_id);
        }
        self.relocations.insert(
            evidence.relocation_id,
            PendingRelocation::Submitted(SubmittedState {
                requests: state.requests,
                relocations: submitted,
            }),
        );
        Ok(EngineRelocationTicket {
            relocation_id: evidence.relocation_id,
        })
    }

    /// Publishes a submitted relocation at one confirmed GPU frontier.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, non-submitted, unconfirmed, or non-advancing
    /// completion evidence and propagates canonical completion failures.
    ///
    /// # Panics
    ///
    /// Panics only if the private request map changes after full preflight.
    pub fn complete_relocation(
        &mut self,
        relocation_id: EngineRelocationId,
        evidence: EngineCompletionEvidence,
    ) -> Result<EngineRelocationPublication, RuntimeSessionError> {
        self.ensure_healthy()?;
        let state = self.submitted_relocation(relocation_id)?.clone();
        self.preflight_request_phases(
            &state.requests,
            RequestPhase::RelocationSubmitted(relocation_id),
        )?;
        let Some(engine_epoch) = state
            .relocations
            .first()
            .map(|submitted| submitted.relocation.engine_epoch)
        else {
            return Err(self.poison("empty submitted relocation"));
        };
        let leases = state
            .relocations
            .iter()
            .map(|submitted| submitted.relocation)
            .collect::<Vec<_>>();
        let completed = self.manager.complete_relocation_batch(
            BatchCompletionReceipt {
                engine_epoch,
                completion_domain: evidence.completion_domain,
                completion_value: evidence.completion_value,
                confirmed: u32::from(evidence.confirmed),
                reserved: 0,
            },
            &leases,
        )?;
        if completed.publications.len() != state.relocations.len()
            || completed.publications.iter().zip(&state.relocations).any(
                |(publication, submitted)| {
                    publication.request != submitted.request
                        || publication.snapshot != submitted.target_snapshot
                },
            )
        {
            return Err(self.poison("relocation completion result ordering"));
        }
        let publications = state
            .requests
            .iter()
            .copied()
            .zip(completed.publications.iter().copied())
            .map(
                |(request_id, publication)| EngineRelocationRequestPublication {
                    request_id,
                    view_version: publication.view_version,
                    boundary: publication.boundary,
                    resident_count: publication.resident_count,
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice();
        for (&request_id, publication) in state.requests.iter().zip(&completed.publications) {
            let record = self
                .requests
                .get_mut(&request_id)
                .expect("relocation completion retained request");
            record.view = *publication;
            record.phase = RequestPhase::RelocationPublicationPending(relocation_id);
        }
        let retirements = engine_retirements(&completed.retirements);
        self.relocations.insert(
            relocation_id,
            PendingRelocation::PublicationPending(PublicationState {
                requests: state.requests,
                retirements: completed.retirements,
            }),
        );
        Ok(EngineRelocationPublication {
            relocation_id,
            requests: publications,
            retirements,
        })
    }

    /// Confirms mirror cleanup and ACKs the exact relocation retirements.
    ///
    /// # Errors
    ///
    /// Rejects foreign, stale, non-pending, incomplete, reordered, or
    /// unacknowledged retirement evidence.
    pub fn confirm_relocation_publication(
        &mut self,
        evidence: &EngineRelocationPublicationEvidence,
    ) -> Result<(), RuntimeSessionError> {
        self.ensure_healthy()?;
        let state = self
            .publication_pending_relocation(evidence.relocation_id)?
            .clone();
        let receipts = validate_reclamation_evidence(
            evidence.mirror_cleanup_confirmed,
            &state.retirements,
            &evidence.reclamation_receipts,
        )?;
        self.preflight_request_phases(
            &state.requests,
            RequestPhase::RelocationPublicationPending(evidence.relocation_id),
        )?;
        if !state.retirements.is_empty() {
            self.manager.acknowledge_reclamations_batch(&receipts)?;
        }
        self.finish_relocation(evidence.relocation_id, &state.requests, RequestPhase::Ready);
        Ok(())
    }

    fn ensure_relocation_epoch(
        &self,
        relocation_id: EngineRelocationId,
    ) -> Result<(), RuntimeSessionError> {
        if relocation_id.session_epoch() != self.session_epoch {
            return Err(RuntimeSessionError::ForeignRelocation(relocation_id));
        }
        Ok(())
    }

    fn prepared_relocation(
        &self,
        relocation_id: EngineRelocationId,
    ) -> Result<&PreparedState, RuntimeSessionError> {
        self.ensure_relocation_epoch(relocation_id)?;
        match self.relocations.get(&relocation_id) {
            Some(PendingRelocation::Prepared(state)) => Ok(state),
            Some(_) => Err(RuntimeSessionError::RelocationNotPrepared(relocation_id)),
            None => Err(self.relocation_id_error(relocation_id)),
        }
    }

    fn submitted_relocation(
        &self,
        relocation_id: EngineRelocationId,
    ) -> Result<&SubmittedState, RuntimeSessionError> {
        self.ensure_relocation_epoch(relocation_id)?;
        match self.relocations.get(&relocation_id) {
            Some(PendingRelocation::Submitted(state)) => Ok(state),
            Some(_) => Err(RuntimeSessionError::RelocationNotSubmitted(relocation_id)),
            None => Err(self.relocation_id_error(relocation_id)),
        }
    }

    fn publication_pending_relocation(
        &self,
        relocation_id: EngineRelocationId,
    ) -> Result<&PublicationState, RuntimeSessionError> {
        self.ensure_relocation_epoch(relocation_id)?;
        match self.relocations.get(&relocation_id) {
            Some(PendingRelocation::PublicationPending(state)) => Ok(state),
            Some(_) => Err(RuntimeSessionError::RelocationPublicationNotPending(
                relocation_id,
            )),
            None => Err(self.relocation_id_error(relocation_id)),
        }
    }

    fn relocation_id_error(&self, relocation_id: EngineRelocationId) -> RuntimeSessionError {
        if relocation_id.session_epoch() != self.session_epoch {
            return RuntimeSessionError::ForeignRelocation(relocation_id);
        }
        if super::was_issued(relocation_id.sequence(), self.next_relocation_sequence) {
            RuntimeSessionError::StaleRelocation(relocation_id)
        } else {
            RuntimeSessionError::UnknownRelocation(relocation_id)
        }
    }

    fn finish_relocation(
        &mut self,
        relocation_id: EngineRelocationId,
        request_ids: &[EngineRequestId],
        phase: RequestPhase,
    ) {
        for request_id in request_ids {
            self.requests
                .get_mut(request_id)
                .expect("relocation preflight retained request")
                .phase = phase;
        }
        self.relocations
            .remove(&relocation_id)
            .expect("relocation preflight retained operation");
    }
}

fn validate_request_evidence<T>(
    requests: &[EngineRequestId],
    evidence: &[T],
    request_id: impl Fn(&T) -> EngineRequestId,
) -> Result<(), RuntimeSessionError> {
    if evidence.len() != requests.len() {
        return Err(RuntimeSessionError::EvidenceCardinality {
            field: "relocation requests",
            expected: requests.len(),
            actual: evidence.len(),
        });
    }
    for (index, (&expected, item)) in requests.iter().zip(evidence).enumerate() {
        let actual = request_id(item);
        if actual != expected {
            return Err(RuntimeSessionError::EvidenceRequest {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn engine_relocation_plan(
    request_id: EngineRequestId,
    prepared: &PreparedRelocation,
) -> EngineRelocationPlan {
    EngineRelocationPlan {
        request_id,
        class_id: prepared.plan.class_id,
        base_version: prepared.plan.base_version,
        target_version: prepared.plan.target_version,
        fragmentation_milli: prepared.plan.fragmentation_milli,
        source_pages: prepared.plan.source_pages.clone(),
        destination_pages: prepared.plan.destination_pages.clone(),
        moves: prepared.plan.moves.clone(),
        projected_reclaimed_pages: prepared.plan.projected_reclaimed_pages,
    }
}
