use std::collections::BTreeSet;

use serde::Serialize;

use super::control::{PrefixPhase, SessionPrefix};
use super::{
    EnginePrefixId, EngineReleaseId, EngineRequestId, PendingRelease, PendingReleasePhase,
    RequestPhase, RuntimeSession, RuntimeSessionError, SessionRequest,
};
#[cfg(any(test, feature = "test-support"))]
use crate::kv_manager::PrefixPublishRelease;
use crate::kv_manager::{
    DetachedAction, DetachedBinding, DetachedReason, MaterializedRequestView, PageLease,
    PrefixPublishItem, PrefixSemanticKey,
};

/// One request-to-prefix ownership transfer committed by the canonical manager.
///
/// Manager request, snapshot, prefix, and reclamation capabilities are
/// deliberately absent. The release remains pending until the engine confirms
/// that every detached mirror binding has been cleared.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePublishedPrefixRelease {
    pub request_id: EngineRequestId,
    pub prefix_id: EnginePrefixId,
    pub key: PrefixSemanticKey,
    pub resident_count: u32,
    pub detached: Box<[DetachedBinding]>,
}

/// An already-committed atomic prefix publication and request release.
///
/// The caller must finish the release through [`RuntimeSession::confirm_release`].
/// This operation transfers every request page reference to its new prefix, so
/// it never produces reclamation certificates.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnginePrefixPublishReleasePlan {
    pub release_id: EngineReleaseId,
    pub items: Box<[EnginePublishedPrefixRelease]>,
}

impl RuntimeSession {
    /// Atomically publishes ready request heads and releases their requests.
    ///
    /// The manager commit transfers page ownership directly from each request
    /// to its new resident prefix. The returned release is therefore pending
    /// mirror cleanup but has no reclamation evidence; callers finish it with
    /// [`RuntimeSession::confirm_release`].
    ///
    /// # Errors
    ///
    /// Rejects empty or duplicate input, non-ready requests, duplicate semantic
    /// keys, exhausted prefix or release identities, and canonical manager
    /// failures without mutation. A malformed manager result after commit
    /// poisons the session.
    ///
    /// # Panics
    ///
    /// Panics only if private session maps change after complete preflight,
    /// which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn publish_prefix_and_release_batch(
        &mut self,
        items: &[(EngineRequestId, PrefixSemanticKey)],
    ) -> Result<EnginePrefixPublishReleasePlan, RuntimeSessionError> {
        self.ensure_healthy()?;
        self.ensure_prefix_operations_supported()?;
        if items.is_empty() {
            return Err(RuntimeSessionError::EmptyBatch);
        }
        let mut request_ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut records = Vec::with_capacity(items.len());
        for &(request_id, key) in items {
            if !request_ids.insert(request_id) {
                return Err(RuntimeSessionError::DuplicateRequest(request_id));
            }
            if !keys.insert(key) || self.prefix_index.contains_key(&key) {
                return Err(crate::kv_manager::KvManagerError::DuplicatePrefixKey.into());
            }
            records.push(self.ready_request(request_id)?.clone());
        }

        // Validate both identity ranges before the manager commit, but do not
        // advance either high-water mark until that commit succeeds.
        let prefix_count = u64::try_from(items.len())
            .map_err(|_| RuntimeSessionError::IdentityExhausted("prefix"))?;
        let first_prefix_sequence = self.next_prefix_sequence;
        let next_prefix_sequence = first_prefix_sequence
            .checked_add(prefix_count)
            .ok_or(RuntimeSessionError::IdentityExhausted("prefix"))?;
        let release_sequence = self.next_release_sequence;
        let next_release_sequence = release_sequence
            .checked_add(1)
            .ok_or(RuntimeSessionError::IdentityExhausted("release"))?;
        let prefix_ids = (first_prefix_sequence..next_prefix_sequence)
            .map(|sequence| EnginePrefixId {
                session_epoch: self.session_epoch,
                sequence,
            })
            .collect::<Vec<_>>();
        let release_id = EngineReleaseId {
            session_epoch: self.session_epoch,
            sequence: release_sequence,
        };
        let manager_items = records
            .iter()
            .zip(items)
            .map(|(record, (_, key))| PrefixPublishItem {
                request: record.view.request,
                expected_head: record.view.snapshot,
                key: *key,
            })
            .collect::<Vec<_>>();
        let materialized_items = records
            .iter()
            .map(|record| (record.view.request, record.view.snapshot))
            .collect::<Vec<_>>();
        let materialized = self
            .manager
            .materialize_request_views_batch(&materialized_items)?;
        let expected_detached =
            expected_detached(&records, &materialized, self.manager.page_tokens())
                .ok_or_else(|| self.poison("prefix publish-release snapshot changed"))?;

        let transferred = self
            .manager
            .publish_prefix_and_release_batch(&manager_items)?;
        self.next_prefix_sequence = next_prefix_sequence;
        self.next_release_sequence = next_release_sequence;
        #[cfg(any(test, feature = "test-support"))]
        let transferred = self.apply_prefix_release_test_fault(transferred);
        let lookup_keys = items.iter().map(|(_, key)| *key).collect::<Vec<_>>();
        let Ok(published_hints) = self.manager.lookup_prefix_batch(&lookup_keys) else {
            return Err(self.poison("prefix publish-release lookup failed"));
        };

        let mut manager_prefixes = BTreeSet::new();
        if transferred.len() != items.len()
            || published_hints.len() != items.len()
            || transferred.iter().enumerate().any(|(index, item)| {
                let record = &records[index];
                let (_, key) = items[index];
                let hint = published_hints[index];
                item.publication.key != key
                    || item.publication.resident_count != record.view.resident_count
                    || hint.key != key
                    || hint.candidate != Some(item.publication.prefix)
                    || hint.resident_count != item.publication.resident_count
                    || self.prefix_leases.contains_key(&item.publication.prefix)
                    || !manager_prefixes.insert(item.publication.prefix)
                    || item.release.request != record.view.request
                    || item.release.detached_snapshot != record.view.snapshot
                    || item.release.detached.as_ref() != expected_detached[index].as_ref()
            })
        {
            return Err(self.poison("prefix publish-release result changed"));
        }

        let leases = records
            .iter()
            .map(|record| record.view.request)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let pending_requests = items
            .iter()
            .map(|(request_id, _)| *request_id)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut engine_items = Vec::with_capacity(transferred.len());
        for ((transferred, prefix_id), &(request_id, key)) in transferred
            .into_vec()
            .into_iter()
            .zip(prefix_ids)
            .zip(items)
        {
            let previous = self.prefixes.insert(
                prefix_id,
                SessionPrefix {
                    lease: transferred.publication.prefix,
                    key,
                    resident_count: transferred.publication.resident_count,
                    phase: PrefixPhase::Resident,
                },
            );
            debug_assert!(previous.is_none());
            let previous = self
                .prefix_leases
                .insert(transferred.publication.prefix, prefix_id);
            debug_assert!(previous.is_none());
            let previous = self.prefix_index.insert(key, prefix_id);
            debug_assert!(previous.is_none());
            self.requests
                .get_mut(&request_id)
                .expect("prefix release preflight retained request")
                .phase = RequestPhase::ReleasePending(release_id);
            engine_items.push(EnginePublishedPrefixRelease {
                request_id,
                prefix_id,
                key,
                resident_count: transferred.publication.resident_count,
                detached: transferred.release.detached,
            });
        }
        let previous = self.releases.insert(
            release_id,
            PendingRelease {
                requests: pending_requests,
                leases,
                phase: PendingReleasePhase::AwaitingAcknowledgement {
                    retirements: Box::new([]),
                },
            },
        );
        debug_assert!(previous.is_none());
        Ok(EnginePrefixPublishReleasePlan {
            release_id,
            items: engine_items.into_boxed_slice(),
        })
    }

    #[cfg(any(test, feature = "test-support"))]
    fn apply_prefix_release_test_fault(
        &mut self,
        mut output: Box<[PrefixPublishRelease]>,
    ) -> Box<[PrefixPublishRelease]> {
        if self.test_fault == Some(super::RuntimeSessionTestFault::PrefixPublishReleaseOutput) {
            self.test_fault = None;
            if let Some(binding) = output
                .first_mut()
                .and_then(|item| item.release.detached.first_mut())
            {
                binding.old_backend_index = binding.old_backend_index.saturating_add(1);
            }
        }
        output
    }
}

fn expected_detached(
    records: &[SessionRequest],
    materialized: &[MaterializedRequestView],
    page_tokens: u64,
) -> Option<Vec<Box<[DetachedBinding]>>> {
    if materialized.len() != records.len() {
        return None;
    }
    records
        .iter()
        .zip(materialized)
        .map(|(record, view)| expected_request_detached(record, view, page_tokens))
        .collect()
}

fn expected_request_detached(
    record: &SessionRequest,
    materialized: &MaterializedRequestView,
    page_tokens: u64,
) -> Option<Box<[DetachedBinding]>> {
    let mut bindings = Vec::new();
    let mut all_pages = BTreeSet::new();
    if materialized.view != record.view {
        return None;
    }
    for page in &materialized.pages {
        if !all_pages.insert(page.page) {
            return None;
        }
        let token_begin = page.logical_ordinal.checked_mul(page_tokens)?;
        let token_end_exclusive = token_begin.checked_add(u64::from(page.valid_token_count))?;
        bindings.push(DetachedBinding {
            old: page.page,
            replacement: PageLease::default(),
            logical_ordinal: page.logical_ordinal,
            old_backend_index: page.backend_index,
            replacement_backend_index: 0,
            token_begin,
            token_end_exclusive,
            class_id: page.class_id,
            backend_domain: page.backend_domain,
            action: DetachedAction::Clear,
            reason: DetachedReason::PrefixTransfer,
            reserved: 0,
        });
    }
    bindings.sort_by_key(|binding| {
        (
            binding.class_id,
            binding.logical_ordinal,
            binding.action as u16,
            binding.old,
        )
    });
    (bindings.len() == record.view.resident_count as usize).then(|| bindings.into_boxed_slice())
}
