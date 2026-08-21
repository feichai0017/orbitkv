use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::Serialize;

use super::arena::RuntimeClass;
use super::manager_state::{ClassDelta, ClassTransition};
use super::persistent_snapshot::{ClassRoot, RootEntry};
use super::{
    CanonicalKvManager, KvManagerError, PageLease, RequestLease, RequestSnapshot, RequestView,
    SnapshotLease, TailActionKind, ViewVersion,
};

const TOKEN_CHUNK_CAPACITY: usize = 16;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct PersistentTokenTable {
    tail: Option<Arc<TokenChunk>>,
    updates: Option<Arc<TokenUpdateChunk>>,
    len: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TokenChunk {
    previous: Option<Arc<TokenChunk>>,
    placements: Box<[TokenPlacement]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TokenUpdateChunk {
    previous: Option<Arc<TokenUpdateChunk>>,
    placements: Box<[TokenPlacement]>,
}

impl PersistentTokenTable {
    pub(super) fn materialize(&self) -> Result<Box<[TokenPlacement]>, KvManagerError> {
        let capacity = usize::try_from(self.len)
            .map_err(|_| KvManagerError::ArithmeticOverflow("token table length"))?;
        let mut chunks = Vec::new();
        let mut current = self.tail.as_deref();
        while let Some(chunk) = current {
            chunks.push(chunk);
            current = chunk.previous.as_deref();
        }
        let mut placements = Vec::with_capacity(capacity);
        for chunk in chunks.into_iter().rev() {
            placements.extend_from_slice(&chunk.placements);
        }
        if placements.len() != capacity {
            return Err(KvManagerError::Invariant("token table materialization"));
        }
        let mut updates = Vec::new();
        let mut current = self.updates.as_deref();
        while let Some(chunk) = current {
            updates.push(chunk);
            current = chunk.previous.as_deref();
        }
        for chunk in updates.into_iter().rev() {
            for update in &chunk.placements {
                let index = usize::try_from(update.token_id)
                    .map_err(|_| KvManagerError::InvalidTokenView)?;
                let placement = placements
                    .get_mut(index)
                    .ok_or(KvManagerError::InvalidTokenView)?;
                if placement.token_id != update.token_id {
                    return Err(KvManagerError::InvalidTokenView);
                }
                *placement = *update;
            }
        }
        Ok(placements.into_boxed_slice())
    }

    pub(super) fn append(&self, placements: &[TokenPlacement]) -> Result<Self, KvManagerError> {
        if placements.is_empty() {
            return Ok(self.clone());
        }
        let mut result = self.clone();
        let mut remaining = placements;
        if let Some(tail) = result.tail.as_ref()
            && tail.placements.len() < TOKEN_CHUNK_CAPACITY
        {
            let take = (TOKEN_CHUNK_CAPACITY - tail.placements.len()).min(remaining.len());
            let mut values = tail.placements.to_vec();
            values.extend_from_slice(&remaining[..take]);
            result.tail = Some(Arc::new(TokenChunk {
                previous: tail.previous.clone(),
                placements: values.into_boxed_slice(),
            }));
            remaining = &remaining[take..];
        }
        while !remaining.is_empty() {
            let take = remaining.len().min(TOKEN_CHUNK_CAPACITY);
            result.tail = Some(Arc::new(TokenChunk {
                previous: result.tail.clone(),
                placements: remaining[..take].to_vec().into_boxed_slice(),
            }));
            remaining = &remaining[take..];
        }
        result.len = result
            .len
            .checked_add(u64::try_from(placements.len()).map_err(|_| {
                KvManagerError::ArithmeticOverflow("appended token placement count")
            })?)
            .ok_or(KvManagerError::ArithmeticOverflow("token table length"))?;
        Ok(result)
    }

    pub(super) fn patch(&self, placements: &[TokenPlacement]) -> Result<Self, KvManagerError> {
        if placements.is_empty() {
            return Ok(self.clone());
        }
        let mut previous = None;
        for placement in placements {
            if placement.token_id >= self.len
                || previous.is_some_and(|token_id| token_id >= placement.token_id)
            {
                return Err(KvManagerError::InvalidTokenView);
            }
            previous = Some(placement.token_id);
        }
        let mut result = self.clone();
        result.updates = Some(Arc::new(TokenUpdateChunk {
            previous: result.updates.clone(),
            placements: placements.to_vec().into_boxed_slice(),
        }));
        Ok(result)
    }

    fn from_materialized(placements: &[TokenPlacement]) -> Result<Self, KvManagerError> {
        Self::default().append(placements)
    }

    pub(super) const fn len(&self) -> u64 {
        self.len
    }
}

pub(super) fn append_class_token_delta(
    page_tokens: u64,
    class: RuntimeClass,
    root: &ClassRoot,
    delta: &ClassDelta,
    previous_boundary: u64,
    target_boundary: u64,
) -> Result<PersistentTokenTable, KvManagerError> {
    if root.tokens.len() != previous_boundary {
        return Err(KvManagerError::Invariant("token table boundary"));
    }
    let retained_start = class.retained_start(target_boundary);
    let previous_retained_start = class.retained_start(previous_boundary);
    let mut patches = Vec::new();
    for token_id in previous_retained_start..retained_start.min(previous_boundary) {
        patches.push(TokenPlacement {
            token_id,
            disposition: TokenDisposition::semantically_dead(
                u64::from(class.class_id) + 1,
                target_boundary,
            ),
            location: None,
        });
    }
    if delta.tail_action == TailActionKind::CopyOnWrite {
        let destination = delta
            .tail_destination
            .ok_or(KvManagerError::Invariant("COW token destination"))?;
        let tail_begin = destination
            .logical_ordinal
            .checked_mul(page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow("COW token begin"))?;
        for token_id in tail_begin.max(retained_start)..previous_boundary {
            patches.push(TokenPlacement {
                token_id,
                disposition: TokenDisposition::RETAINED,
                location: Some(token_location(destination, token_id, page_tokens)?),
            });
        }
    }
    patches.sort_unstable_by_key(|placement| placement.token_id);
    if patches
        .windows(2)
        .any(|pair| pair[0].token_id == pair[1].token_id)
    {
        return Err(KvManagerError::Invariant("overlapping token delta"));
    }

    let mut appended = Vec::with_capacity(
        usize::try_from(target_boundary - previous_boundary)
            .map_err(|_| KvManagerError::ArithmeticOverflow("appended tokens"))?,
    );
    for token_id in previous_boundary..target_boundary {
        let retained = token_id >= retained_start;
        let location = if retained {
            let ordinal = token_id / page_tokens;
            let entry = delta
                .tail_destination
                .or(delta.tail_source)
                .filter(|entry| entry.logical_ordinal == ordinal)
                .or_else(|| {
                    delta
                        .writes
                        .iter()
                        .copied()
                        .find(|entry| entry.logical_ordinal == ordinal)
                })
                .ok_or(KvManagerError::Invariant("new token placement"))?;
            Some(token_location(entry, token_id, page_tokens)?)
        } else {
            None
        };
        appended.push(TokenPlacement {
            token_id,
            disposition: if retained {
                TokenDisposition::RETAINED
            } else {
                TokenDisposition::semantically_dead(u64::from(class.class_id) + 1, target_boundary)
            },
            location,
        });
    }
    root.tokens.patch(&patches)?.append(&appended)
}

pub(super) fn apply_dense_class_transition(
    page_tokens: u64,
    class: RuntimeClass,
    root: &mut ClassRoot,
    delta: &ClassDelta,
    transition: &ClassTransition,
    previous_boundary: u64,
    target_boundary: u64,
) -> Result<(), KvManagerError> {
    for _ in 0..transition.retire_from_root {
        root.entries
            .pop_front()
            .ok_or(KvManagerError::Invariant("root retirement"))?;
    }
    if let Some(destination) = delta.tail_destination
        && destination.logical_ordinal >= transition.retain_first_ordinal
    {
        if delta.tail_action == TailActionKind::CopyOnWrite {
            let source = delta
                .tail_source
                .ok_or(KvManagerError::Invariant("COW tail source"))?;
            let removed = root
                .entries
                .pop_back()
                .ok_or(KvManagerError::Invariant("COW tail retirement"))?;
            if removed != source {
                return Err(KvManagerError::Invariant("COW tail identity"));
            }
        }
        root.entries.push_back(destination);
    }
    root.entries.extend(
        delta
            .writes
            .iter()
            .skip(transition.retire_from_writes)
            .copied(),
    );
    if root.entries.len() != transition.resident_count {
        return Err(KvManagerError::Invariant("published root count"));
    }
    root.tokens = append_class_token_delta(
        page_tokens,
        class,
        root,
        delta,
        previous_boundary,
        target_boundary,
    )?;
    Ok(())
}

fn token_location(
    entry: RootEntry,
    token_id: u64,
    page_tokens: u64,
) -> Result<TokenLocation, KvManagerError> {
    Ok(TokenLocation {
        page: entry.page,
        backend_index: entry.backend_index,
        offset: u32::try_from(token_id % page_tokens)
            .map_err(|_| KvManagerError::ArithmeticOverflow("token offset"))?,
        reserved: 0,
    })
}

/// Why one logical token is absent from the next executable attention view.
///
/// Semantic death is lossless and requires a compiler proof identity. Policy
/// eviction is explicitly approximate and carries the named quality contract
/// that must be qualified independently from byte-exact relocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(u16)]
pub enum TokenDispositionKind {
    Retained = 0,
    SemanticallyDead = 1,
    PolicyEvicted = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct TokenDisposition {
    pub kind: TokenDispositionKind,
    pub policy_or_proof_id: u64,
    pub version: u64,
    pub quality_contract: u64,
}

impl TokenDisposition {
    pub const RETAINED: Self = Self {
        kind: TokenDispositionKind::Retained,
        policy_or_proof_id: 0,
        version: 0,
        quality_contract: 0,
    };

    #[must_use]
    pub const fn semantically_dead(proof_id: u64, proof_version: u64) -> Self {
        Self {
            kind: TokenDispositionKind::SemanticallyDead,
            policy_or_proof_id: proof_id,
            version: proof_version,
            quality_contract: 0,
        }
    }

    #[must_use]
    pub const fn policy_evicted(
        policy_id: u64,
        policy_version: u64,
        quality_contract: u64,
    ) -> Self {
        Self {
            kind: TokenDispositionKind::PolicyEvicted,
            policy_or_proof_id: policy_id,
            version: policy_version,
            quality_contract,
        }
    }

    #[must_use]
    pub const fn retained(self) -> bool {
        matches!(self.kind, TokenDispositionKind::Retained)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(C)]
pub struct TokenLocation {
    pub page: PageLease,
    pub backend_index: u64,
    pub offset: u32,
    pub reserved: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TokenPlacement {
    pub token_id: u64,
    pub disposition: TokenDisposition,
    /// Dead tokens retain their old location until a relocation transaction
    /// physically reclaims that page. Afterwards the location is `None`.
    pub location: Option<TokenLocation>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TokenView {
    pub class_id: u16,
    pub version: ViewVersion,
    pub page_tokens: u32,
    pub placements: Box<[TokenPlacement]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TokenViewQuery {
    pub request: RequestLease,
    pub expected_snapshot: SnapshotLease,
    pub class_id: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TokenDispositionUpdate {
    pub token_id: u64,
    pub disposition: TokenDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ClassTokenDispositionUpdate {
    pub class_id: u16,
    pub token_id: u64,
    pub disposition: TokenDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TokenDispositionBatchItem {
    pub request: RequestLease,
    pub expected_snapshot: SnapshotLease,
    pub updates: Box<[ClassTokenDispositionUpdate]>,
}

impl CanonicalKvManager {
    /// Atomically publishes logical token-disposition changes for a request batch.
    ///
    /// Physical ownership and page references remain unchanged. The returned
    /// snapshots represent the Naive-Evict state from which a later relocation
    /// transaction may reclaim physical pages.
    ///
    /// # Errors
    ///
    /// Returns an error before mutation for empty/duplicate requests, stale
    /// snapshots, busy requests, invalid class/token ordering, malformed policy
    /// evidence, or snapshot staging exhaustion.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after the complete batch
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn mark_token_dispositions_batch(
        &mut self,
        items: &[TokenDispositionBatchItem],
    ) -> Result<Box<[RequestView]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen_requests = BTreeSet::new();
        for item in items {
            if !seen_requests.insert(item.request) {
                return Err(KvManagerError::DuplicateRequest);
            }
        }
        let planned = self.snapshots.plan_many(items.len())?;
        let mut changes = Vec::with_capacity(items.len());
        for (item, slot) in items.iter().zip(planned.iter().copied()) {
            if item.updates.is_empty() {
                return Err(KvManagerError::EmptyBatch);
            }
            let state = self.request(item.request)?;
            if state.released || state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if state.head != item.expected_snapshot {
                return Err(KvManagerError::StaleTokenView);
            }
            if state.pending_step.is_some() || state.inflight_submission.is_some() {
                return Err(KvManagerError::RequestBusy);
            }
            let snapshot = self.request_snapshot(item.request)?;
            let target_version = ViewVersion(
                snapshot
                    .view_version
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ViewVersionExhausted)?,
            );
            let mut roots = snapshot.roots.iter().cloned().collect::<Vec<_>>();
            let mut previous_class = None;
            let mut offset = 0_usize;
            while offset < item.updates.len() {
                let class_id = item.updates[offset].class_id;
                if previous_class.is_some_and(|value| value >= class_id) {
                    return Err(KvManagerError::InvalidTokenView);
                }
                previous_class = Some(class_id);
                let end = item.updates[offset..]
                    .iter()
                    .position(|update| update.class_id != class_id)
                    .map_or(item.updates.len(), |relative| offset + relative);
                let root = roots
                    .get_mut(usize::from(class_id))
                    .ok_or(KvManagerError::InvalidClass(class_id))?;
                let view = TokenView {
                    class_id,
                    version: snapshot.view_version,
                    page_tokens: u32::try_from(self.page_tokens)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("page tokens"))?,
                    placements: root.tokens.materialize()?,
                };
                let updates = item.updates[offset..end]
                    .iter()
                    .map(|update| TokenDispositionUpdate {
                        token_id: update.token_id,
                        disposition: update.disposition,
                    })
                    .collect::<Vec<_>>();
                let target = mark_token_dispositions(&view, &updates)?;
                if target.version != target_version {
                    return Err(KvManagerError::Invariant("token view target version"));
                }
                root.tokens = PersistentTokenTable::from_materialized(&target.placements)?;
                offset = end;
            }
            let target_snapshot = SnapshotLease {
                engine_epoch: self.engine_epoch,
                slot: slot.0,
                generation: slot.1,
            };
            let resident_count = u32::try_from(snapshot.resident_count())
                .map_err(|_| KvManagerError::ArithmeticOverflow("resident count"))?;
            changes.push((
                item.request,
                state.head,
                slot,
                target_snapshot,
                RequestSnapshot {
                    boundary: snapshot.boundary,
                    view_version: target_version,
                    roots: roots.into(),
                },
                resident_count,
            ));
        }
        for (_, _, slot, _, snapshot, _) in &changes {
            self.snapshots.insert_planned(*slot, snapshot.clone());
        }
        let mut outputs = Vec::with_capacity(changes.len());
        for (request, old, _, target, snapshot, resident_count) in changes {
            let view_version = snapshot.view_version;
            let boundary = snapshot.boundary;
            self.snapshots
                .remove(old.slot, old.generation)
                .expect("disposition preflight retained old snapshot");
            self.request_mut(request)
                .expect("disposition preflight retained request")
                .head = target;
            outputs.push(RequestView {
                request,
                snapshot: target,
                view_version,
                boundary,
                resident_count,
            });
        }
        Ok(outputs.into_boxed_slice())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RelocationPageState {
    pub page: PageLease,
    pub backend_index: u64,
    pub request_refs: u32,
    pub prefix_refs: u32,
    pub reader_pins: u32,
    pub writer_present: bool,
}

impl RelocationPageState {
    const fn privately_relocatable(self) -> bool {
        self.request_refs == 1
            && self.prefix_refs == 0
            && self.reader_pins == 0
            && !self.writer_present
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RelocationDestination {
    pub page: PageLease,
    pub backend_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RelocationPolicy {
    /// Fragmentation threshold in thousandths. The paper's default 0.25 is 250.
    pub fragmentation_threshold_milli: u16,
    pub maximum_source_pages: u32,
    pub evacuation_headroom_pages: u32,
}

impl Default for RelocationPolicy {
    fn default() -> Self {
        Self {
            fragmentation_threshold_milli: 250,
            maximum_source_pages: 32,
            evacuation_headroom_pages: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[repr(C)]
pub struct TokenMove {
    pub token_id: u64,
    pub source: TokenLocation,
    pub destination: TokenLocation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RelocationPlan {
    pub class_id: u16,
    pub base_version: ViewVersion,
    pub target_version: ViewVersion,
    pub fragmentation_milli: u16,
    pub source_pages: Box<[PageLease]>,
    pub destination_pages: Box<[PageLease]>,
    pub moves: Box<[TokenMove]>,
    pub projected_reclaimed_pages: u32,
}

#[derive(Clone, Copy)]
struct PageOccupancy {
    state: RelocationPageState,
    present: u32,
    live: u32,
}

/// Builds one profitable, request-local relocation plan.
///
/// `Ok(None)` is an ordinary admission decision: fragmentation, private-page
/// eligibility, destination headroom, or projected physical gain was
/// insufficient. Malformed identities and token mappings fail closed.
///
/// # Errors
///
/// Returns an error for malformed token views, duplicate/stale page identity,
/// invalid relocation policy, checked arithmetic failure, or a placement that
/// disagrees with its page census.
#[allow(clippy::too_many_lines)]
pub fn plan_token_relocation(
    view: &TokenView,
    pages: &[RelocationPageState],
    destinations: &[RelocationDestination],
    policy: RelocationPolicy,
) -> Result<Option<RelocationPlan>, KvManagerError> {
    validate_token_view(view)?;
    validate_policy(policy)?;
    let page_tokens = view.page_tokens;
    let page_states = pages
        .iter()
        .copied()
        .map(|state| (state.page, state))
        .collect::<BTreeMap<_, _>>();
    if page_states.len() != pages.len() {
        return Err(KvManagerError::DuplicatePage);
    }

    let mut occupancy = page_states
        .iter()
        .map(|(&page, &state)| {
            (
                page,
                PageOccupancy {
                    state,
                    present: 0,
                    live: 0,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut source_arena = None;
    for placement in &view.placements {
        let Some(location) = placement.location else {
            continue;
        };
        let arena = (
            location.page.engine_epoch,
            location.page.pool_epoch,
            location.page.pool_id,
        );
        if source_arena
            .replace(arena)
            .is_some_and(|expected| expected != arena)
        {
            return Err(KvManagerError::WrongPageArena);
        }
        let item = occupancy
            .get_mut(&location.page)
            .ok_or(KvManagerError::TokenPlacementMismatch)?;
        if item.state.backend_index != location.backend_index {
            return Err(KvManagerError::TokenPlacementMismatch);
        }
        item.present = item
            .present
            .checked_add(1)
            .ok_or(KvManagerError::ArithmeticOverflow("page occupancy"))?;
        item.live += u32::from(placement.disposition.retained());
    }
    if occupancy
        .values()
        .any(|item| item.present > page_tokens || item.live > item.present)
    {
        return Err(KvManagerError::TokenPlacementMismatch);
    }

    let allocated = occupancy.values().filter(|item| item.present > 0).count();
    if allocated == 0 {
        return Ok(None);
    }
    let live_tokens = occupancy
        .values()
        .try_fold(0_u64, |total, item| total.checked_add(u64::from(item.live)))
        .ok_or(KvManagerError::ArithmeticOverflow("live token count"))?;
    let slots = u64::try_from(allocated)
        .map_err(|_| KvManagerError::ArithmeticOverflow("allocated pages"))?
        .checked_mul(u64::from(page_tokens))
        .ok_or(KvManagerError::ArithmeticOverflow("allocated token slots"))?;
    let waste = slots - live_tokens;
    let fragmentation = u16::try_from(
        waste
            .checked_mul(1000)
            .ok_or(KvManagerError::ArithmeticOverflow("fragmentation ratio"))?
            / slots,
    )
    .map_err(|_| KvManagerError::ArithmeticOverflow("fragmentation ratio"))?;
    if fragmentation < policy.fragmentation_threshold_milli {
        return Ok(None);
    }

    let mut candidates = occupancy
        .iter()
        .filter(|(_, item)| {
            item.present > 0 && item.live < page_tokens && item.state.privately_relocatable()
        })
        .map(|(&page, item)| (item.live, page))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|&(live, page)| (live, page));
    candidates.truncate(policy.maximum_source_pages as usize);

    let mut selected = Vec::new();
    let mut selected_live = 0_u32;
    let mut destination_count = 0_u32;
    for (live, page) in candidates {
        selected.push(page);
        selected_live = selected_live
            .checked_add(live)
            .ok_or(KvManagerError::ArithmeticOverflow("relocation live tokens"))?;
        destination_count = selected_live.div_ceil(page_tokens);
    }
    let source_count = u32::try_from(selected.len())
        .map_err(|_| KvManagerError::ArithmeticOverflow("relocation source pages"))?;
    if source_count == 0 || source_count <= destination_count {
        return Ok(None);
    }
    if destination_count > policy.evacuation_headroom_pages
        || usize::try_from(destination_count)
            .map_err(|_| KvManagerError::ArithmeticOverflow("destination count"))?
            > destinations.len()
    {
        return Ok(None);
    }

    let selected_set = selected.iter().copied().collect::<BTreeSet<_>>();
    let source_pool = selected
        .first()
        .copied()
        .ok_or(KvManagerError::Invariant("relocation source selection"))?;
    let mut seen_destinations = BTreeSet::new();
    let selected_destinations = destinations
        .iter()
        .copied()
        .take(destination_count as usize)
        .map(|destination| {
            if destination.page.engine_epoch != source_pool.engine_epoch
                || destination.page.pool_epoch != source_pool.pool_epoch
                || destination.page.pool_id != source_pool.pool_id
                || selected_set.contains(&destination.page)
                || page_states.contains_key(&destination.page)
                || !seen_destinations.insert(destination.page)
            {
                return Err(KvManagerError::WrongPageArena);
            }
            Ok(destination)
        })
        .collect::<Result<Vec<_>, KvManagerError>>()?;

    let mut retained = view
        .placements
        .iter()
        .filter(|placement| {
            placement.disposition.retained()
                && placement
                    .location
                    .is_some_and(|location| selected_set.contains(&location.page))
        })
        .copied()
        .collect::<Vec<_>>();
    retained.sort_unstable_by_key(|placement| placement.token_id);
    if retained.len() != selected_live as usize {
        return Err(KvManagerError::TokenPlacementMismatch);
    }
    let moves = retained
        .iter()
        .enumerate()
        .map(|(index, placement)| {
            let destination_page = selected_destinations[index / page_tokens as usize];
            Ok(TokenMove {
                token_id: placement.token_id,
                source: placement
                    .location
                    .ok_or(KvManagerError::TokenPlacementMismatch)?,
                destination: TokenLocation {
                    page: destination_page.page,
                    backend_index: destination_page.backend_index,
                    offset: u32::try_from(index % page_tokens as usize).map_err(|_| {
                        KvManagerError::ArithmeticOverflow("destination token offset")
                    })?,
                    reserved: 0,
                },
            })
        })
        .collect::<Result<Vec<_>, KvManagerError>>()?;
    let target_version = ViewVersion(
        view.version
            .0
            .checked_add(1)
            .ok_or(KvManagerError::ViewVersionExhausted)?,
    );
    Ok(Some(RelocationPlan {
        class_id: view.class_id,
        base_version: view.version,
        target_version,
        fragmentation_milli: fragmentation,
        source_pages: selected.into_boxed_slice(),
        destination_pages: selected_destinations
            .iter()
            .map(|item| item.page)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        moves: moves.into_boxed_slice(),
        projected_reclaimed_pages: source_count - destination_count,
    }))
}

/// Applies a completed relocation to the logical token table.
///
/// Physical page phases and reclamation certificates remain the canonical
/// manager transaction's responsibility; this function only constructs the
/// immutable post-copy logical view after exact completion is known.
///
/// # Errors
///
/// Returns an error when the plan is stale, non-profitable, aliases a source
/// and destination, omits a retained token, moves a dead token, or otherwise
/// violates token conservation and unique placement.
pub fn apply_token_relocation(
    view: &TokenView,
    plan: &RelocationPlan,
) -> Result<TokenView, KvManagerError> {
    validate_token_view(view)?;
    if plan.class_id != view.class_id || plan.base_version != view.version {
        return Err(KvManagerError::StaleTokenView);
    }
    if plan.target_version.0 != plan.base_version.0.checked_add(1).unwrap_or(0)
        || plan.projected_reclaimed_pages == 0
        || plan.source_pages.len() <= plan.destination_pages.len()
    {
        return Err(KvManagerError::InvalidRelocationPlan);
    }
    let sources = plan.source_pages.iter().copied().collect::<BTreeSet<_>>();
    let destinations = plan
        .destination_pages
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if sources.len() != plan.source_pages.len()
        || destinations.len() != plan.destination_pages.len()
        || !sources.is_disjoint(&destinations)
    {
        return Err(KvManagerError::InvalidRelocationPlan);
    }
    let moves = plan
        .moves
        .iter()
        .map(|item| (item.token_id, *item))
        .collect::<BTreeMap<_, _>>();
    if moves.len() != plan.moves.len() {
        return Err(KvManagerError::InvalidRelocationPlan);
    }
    let mut destination_slots = BTreeSet::new();
    for item in &plan.moves {
        if !sources.contains(&item.source.page)
            || !destinations.contains(&item.destination.page)
            || item.source.reserved != 0
            || item.destination.reserved != 0
            || item.source.offset >= view.page_tokens
            || item.destination.offset >= view.page_tokens
            || !destination_slots.insert((item.destination.page, item.destination.offset))
        {
            return Err(KvManagerError::InvalidRelocationPlan);
        }
    }

    let mut placements = Vec::with_capacity(view.placements.len());
    let mut applied_moves = 0_usize;
    for placement in &view.placements {
        let mut next = *placement;
        match (placement.location, placement.disposition.retained()) {
            (Some(location), true) if sources.contains(&location.page) => {
                let movement = moves
                    .get(&placement.token_id)
                    .ok_or(KvManagerError::InvalidRelocationPlan)?;
                if movement.source != location {
                    return Err(KvManagerError::InvalidRelocationPlan);
                }
                next.location = Some(movement.destination);
                applied_moves += 1;
            }
            (Some(location), false) if sources.contains(&location.page) => {
                if moves.contains_key(&placement.token_id) {
                    return Err(KvManagerError::InvalidRelocationPlan);
                }
                next.location = None;
            }
            _ => {
                if moves.contains_key(&placement.token_id) {
                    return Err(KvManagerError::InvalidRelocationPlan);
                }
            }
        }
        placements.push(next);
    }
    if applied_moves != moves.len()
        || plan.projected_reclaimed_pages
            != u32::try_from(plan.source_pages.len() - plan.destination_pages.len())
                .map_err(|_| KvManagerError::InvalidRelocationPlan)?
    {
        return Err(KvManagerError::InvalidRelocationPlan);
    }
    let result = TokenView {
        class_id: view.class_id,
        version: plan.target_version,
        page_tokens: view.page_tokens,
        placements: placements.into_boxed_slice(),
    };
    validate_token_view(&result)?;
    Ok(result)
}

/// Applies one ordered victim/proof set to an immutable logical token view.
///
/// This is the common policy seam for Naive-Evict and physical relocation.
/// It changes logical liveness only; physical locations remain attached until
/// a later relocation transaction reclaims their pages.
///
/// # Errors
///
/// Returns an error for a malformed base view, duplicate/unordered or unknown
/// token IDs, an attempt to mark a token Retained, or an attempt to change the
/// evidence attached to a token that is already dead.
pub fn mark_token_dispositions(
    view: &TokenView,
    updates: &[TokenDispositionUpdate],
) -> Result<TokenView, KvManagerError> {
    validate_token_view(view)?;
    if updates.is_empty() {
        return Err(KvManagerError::EmptyBatch);
    }
    let mut previous = None;
    let mut update_map = BTreeMap::new();
    for update in updates {
        if update.disposition.retained()
            || previous.is_some_and(|token_id| token_id >= update.token_id)
            || update_map
                .insert(update.token_id, update.disposition)
                .is_some()
        {
            return Err(KvManagerError::InvalidTokenView);
        }
        previous = Some(update.token_id);
    }
    let mut placements = Vec::with_capacity(view.placements.len());
    for placement in &view.placements {
        let Some(disposition) = update_map.remove(&placement.token_id) else {
            placements.push(*placement);
            continue;
        };
        if !placement.disposition.retained() {
            return Err(KvManagerError::InvalidTokenView);
        }
        placements.push(TokenPlacement {
            disposition,
            ..*placement
        });
    }
    if !update_map.is_empty() {
        return Err(KvManagerError::InvalidTokenView);
    }
    let result = TokenView {
        class_id: view.class_id,
        version: ViewVersion(
            view.version
                .0
                .checked_add(1)
                .ok_or(KvManagerError::ViewVersionExhausted)?,
        ),
        page_tokens: view.page_tokens,
        placements: placements.into_boxed_slice(),
    };
    validate_token_view(&result)?;
    Ok(result)
}

/// Validates logical-token ordering, disposition contracts, and unique slots.
///
/// # Errors
///
/// Returns an error for zero page geometry, unordered or duplicate token IDs,
/// retained tokens without locations, duplicate physical slots, invalid
/// offsets, or incomplete semantic/policy evidence.
pub fn validate_token_view(view: &TokenView) -> Result<(), KvManagerError> {
    if view.page_tokens == 0 {
        return Err(KvManagerError::InvalidTokenView);
    }
    let mut previous = None;
    let mut occupied = BTreeSet::new();
    for placement in &view.placements {
        if previous.is_some_and(|token_id| token_id >= placement.token_id) {
            return Err(KvManagerError::InvalidTokenView);
        }
        previous = Some(placement.token_id);
        if placement.disposition.retained() && placement.location.is_none() {
            return Err(KvManagerError::InvalidTokenView);
        }
        if placement.disposition.retained()
            && (placement.disposition.policy_or_proof_id != 0
                || placement.disposition.version != 0
                || placement.disposition.quality_contract != 0)
        {
            return Err(KvManagerError::InvalidTokenView);
        }
        if let Some(location) = placement.location
            && (location.reserved != 0
                || location.offset >= view.page_tokens
                || !occupied.insert((location.page, location.offset)))
        {
            return Err(KvManagerError::TokenPlacementMismatch);
        }
        if matches!(
            placement.disposition.kind,
            TokenDispositionKind::PolicyEvicted
        ) && (placement.disposition.policy_or_proof_id == 0
            || placement.disposition.quality_contract == 0)
        {
            return Err(KvManagerError::InvalidTokenView);
        }
        if matches!(
            placement.disposition.kind,
            TokenDispositionKind::SemanticallyDead
        ) && (placement.disposition.policy_or_proof_id == 0
            || placement.disposition.quality_contract != 0)
        {
            return Err(KvManagerError::InvalidTokenView);
        }
    }
    Ok(())
}

fn validate_policy(policy: RelocationPolicy) -> Result<(), KvManagerError> {
    if policy.fragmentation_threshold_milli > 1000
        || policy.maximum_source_pages == 0
        || policy.evacuation_headroom_pages == 0
    {
        return Err(KvManagerError::InvalidRelocationPolicy);
    }
    Ok(())
}
