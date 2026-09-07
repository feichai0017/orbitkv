use super::arena::{PageMut, RuntimeClass};
use super::{
    AddressProgram, Arc, Arena, ArenaStats, BTreeMap, BTreeSet, BackendArenaRegistration,
    BlockDomain, CANONICAL_PAGE_TOKENS, CanonicalKvManager, CensusWork, ClassLayoutProgram,
    ClassRoot, CompiledKvClass, CompiledKvPlan, FIRST_POOL_EPOCH, ForkedRequest, KvManagerError,
    ManagerConfig, ManagerStats, MaterializedRequestView, NEXT_ENGINE_EPOCH, Ordering, PageCounts,
    PageLease, PagePhase, PageState, PersistentRootEntries, PersistentTokenTable,
    PhysicalResidencePolicy, PrefixLease, PrefixLookupHint, PrefixSemanticKey, ReclamationLease,
    RequestForkItem, RequestLease, RequestSnapshot, RequestState, RequestView, RetentionKind,
    RetirementProgram, RootEntry, RootLayout, SnapshotLease, SnapshotPage, StepLease,
    SubmissionLease, TokenView, TokenViewQuery, ViewVersion,
};
#[cfg(test)]
use super::{DeviceKvEntry, HotPathInstrumentation};

impl CanonicalKvManager {
    pub(crate) fn operation_capacity(&self) -> usize {
        self.operations.slots.len()
    }

    /// Creates the canonical Full, sliding, hybrid Full+sliding, or bounded
    /// whole-domain single-class Chunked manager.
    ///
    /// # Errors
    ///
    /// Rejects non-page-16, region-partitioned, multi-class Chunked, or
    /// otherwise unsupported profiles. Every accepted class owns one
    /// explicitly registered physical arena, and reclamation capacity must
    /// cover every registered page; unsupported profiles never fall back.
    pub fn new(
        plan: &CompiledKvPlan,
        config: ManagerConfig,
        backends: &[BackendArenaRegistration],
    ) -> Result<Self, KvManagerError> {
        Self::new_with_residence(plan, config, backends, PhysicalResidencePolicy::Compiled)
    }

    /// Creates a manager with an explicit physical-residence policy.
    ///
    /// The policy never changes compiler-authored attention visibility. The
    /// `RequestLifetime` variant exists as a same-semantics baseline for
    /// attributing benefits to compiled physical reclamation.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::new`], and rejects a
    /// residence policy that is incompatible with the compiled address
    /// program.
    pub fn new_with_residence(
        plan: &CompiledKvPlan,
        config: ManagerConfig,
        backends: &[BackendArenaRegistration],
        physical_residence: PhysicalResidencePolicy,
    ) -> Result<Self, KvManagerError> {
        let classes = compile_manager_profile(plan, config, backends, physical_residence)?;
        let total_pages = classes.iter().try_fold(0_usize, |total, class| {
            total
                .checked_add(
                    usize::try_from(class.backend.page_count)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("page count"))?,
                )
                .ok_or(KvManagerError::ArithmeticOverflow("total page count"))
        })?;
        let mut pages = Vec::with_capacity(total_pages);
        let mut free_pages = Vec::with_capacity(classes.len());
        let mut page_counts = Vec::with_capacity(classes.len());
        for class in &classes {
            pages.extend((0..class.backend.page_count).map(|_| PageState::free(class.class_id)));
            let end = class
                .first_page_id
                .checked_add(class.backend.page_count)
                .ok_or(KvManagerError::ArithmeticOverflow("class page range"))?;
            free_pages.push((class.first_page_id..end).rev().collect());
            page_counts.push(PageCounts {
                free: u64::from(class.backend.page_count),
                ..PageCounts::default()
            });
        }
        let engine_epoch = NEXT_ENGINE_EPOCH
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_add(1)
            })
            .map_err(|_| KvManagerError::EngineEpochExhausted)?;
        // Live request heads, a full attach batch's replacement heads, and
        // prepared operation candidates must coexist until collective commit.
        // Keeping the old empty heads alive is what makes every pre-attach
        // SnapshotLease immutable and generation checked.
        let maximum_snapshots = config
            .maximum_requests
            .checked_mul(2)
            .and_then(|requests| requests.checked_add(config.maximum_operations))
            .ok_or(KvManagerError::ArithmeticOverflow("snapshot capacity"))?;
        Ok(Self {
            engine_epoch,
            pool_epoch: engine_epoch
                .checked_add(FIRST_POOL_EPOCH)
                .ok_or(KvManagerError::EngineEpochExhausted)?,
            plan_fingerprint: plan.fingerprint_digest(),
            page_tokens: plan.page_tokens,
            classes: classes.into_boxed_slice(),
            maximum_step_tokens: u64::from(config.maximum_step_tokens),
            requests: Arena::new("request", config.maximum_requests)?,
            snapshots: Arena::new("snapshot", maximum_snapshots)?,
            prefixes: Arena::new("prefix", config.maximum_prefixes)?,
            prefix_index: BTreeMap::new(),
            operations: Arena::new("operation", config.maximum_operations)?,
            relocations: Arena::new("relocation", config.maximum_operations)?,
            reclamations: Arena::new("reclamation", config.maximum_reclamations)?,
            pages,
            free_pages,
            page_counts,
            completion_high_water: BTreeMap::new(),
            prepared_steps: 0,
            submitted_steps: 0,
            active_prefixes: 0,
            evicted_prefixes: 0,
            #[cfg(test)]
            hot_path: HotPathInstrumentation::default(),
        })
    }

    /// Returns the generation-checked head currently owned by a request.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong-engine, stale request, stale snapshot, or
    /// an unrepresentable resident count.
    pub(super) fn request_view(
        &self,
        request: RequestLease,
    ) -> Result<RequestView, KvManagerError> {
        let state = self.request(request)?;
        if state.released || state.quarantined {
            return Err(KvManagerError::RequestUnavailable);
        }
        let snapshot = self.request_snapshot(request)?;
        Ok(RequestView {
            request,
            snapshot: state.head,
            view_version: snapshot.view_version,
            boundary: snapshot.boundary,
            resident_count: u32::try_from(snapshot.resident_count())
                .map_err(|_| KvManagerError::ArithmeticOverflow("resident count"))?,
        })
    }

    /// Returns an ordered batch of generation-checked request heads after all
    /// identities have been collectively preflighted.
    ///
    /// # Errors
    ///
    /// Rejects an empty batch, duplicate request, unavailable request, stale
    /// lease, or unrepresentable resident count without returning a partial
    /// observation.
    pub fn request_views_batch(
        &self,
        requests: &[RequestLease],
    ) -> Result<Box<[RequestView]>, KvManagerError> {
        if requests.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        requests
            .iter()
            .copied()
            .map(|request| {
                if !seen.insert(request) {
                    return Err(KvManagerError::DuplicateRequest);
                }
                self.request_view(request)
            })
            .collect::<Result<Vec<_>, KvManagerError>>()
            .map(Vec::into_boxed_slice)
    }

    /// Materializes generation-checked logical token views for cold planning.
    ///
    /// Normal append never calls this path. The complete ordered query batch is
    /// preflighted before any view is returned, and every retained placement is
    /// revalidated against the canonical page generation and backend arena.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or duplicate query, stale request/snapshot,
    /// unavailable request, invalid class, malformed token table, or placement
    /// that no longer names a live canonical page generation.
    pub fn token_views_batch(
        &self,
        queries: &[TokenViewQuery],
    ) -> Result<Box<[TokenView]>, KvManagerError> {
        if queries.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        queries
            .iter()
            .map(|query| {
                if !seen.insert((query.request, query.class_id)) {
                    return Err(KvManagerError::DuplicateRequest);
                }
                let state = self.request(query.request)?;
                if state.released || state.quarantined {
                    return Err(KvManagerError::RequestUnavailable);
                }
                if state.head != query.expected_snapshot {
                    return Err(KvManagerError::StaleTokenView);
                }
                let class = self.runtime_class(query.class_id)?;
                let snapshot = self.request_snapshot(query.request)?;
                let root = snapshot
                    .roots
                    .get(usize::from(query.class_id))
                    .ok_or(KvManagerError::InvalidClass(query.class_id))?;
                let placements = root.tokens.materialize()?;
                let view = TokenView {
                    class_id: query.class_id,
                    version: snapshot.view_version,
                    page_tokens: u32::try_from(self.page_tokens)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("page tokens"))?,
                    placements,
                };
                super::validate_token_view(&view)?;
                for placement in &view.placements {
                    let Some(location) = placement.location else {
                        continue;
                    };
                    self.validate_page_lease(class, location.page)?;
                    let page = self.page(location.page.page_id)?;
                    if page.class_id != query.class_id
                        || page.generation != location.page.generation
                        || page.phase != PagePhase::Live
                        || class.backend_index(location.page.page_id)? != location.backend_index
                    {
                        return Err(KvManagerError::TokenPlacementMismatch);
                    }
                }
                Ok(view)
            })
            .collect::<Result<Vec<_>, KvManagerError>>()
            .map(Vec::into_boxed_slice)
    }

    /// Produces non-owning lookup hints. Attach always revalidates the exact
    /// key and generation, so an eviction/recycle race degrades to a miss.
    ///
    /// # Errors
    ///
    /// Returns an error if a resident prefix generation is stale or its
    /// materialized resident count cannot be represented.
    pub fn lookup_prefix_batch(
        &self,
        keys: &[PrefixSemanticKey],
    ) -> Result<Box<[PrefixLookupHint]>, KvManagerError> {
        if keys.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        keys.iter()
            .copied()
            .map(|key| {
                let candidate = self.prefix_index.get(&key).copied();
                let resident_count = candidate.map_or(Ok(0), |prefix| {
                    let state = self.prefixes.get(prefix.slot, prefix.generation)?;
                    let resident_count = state.roots.iter().try_fold(0_usize, |count, root| {
                        count
                            .checked_add(root.entries.len())
                            .ok_or(KvManagerError::ArithmeticOverflow("prefix resident count"))
                    })?;
                    u32::try_from(resident_count)
                        .map_err(|_| KvManagerError::ArithmeticOverflow("prefix resident count"))
                })?;
                Ok(PrefixLookupHint {
                    key,
                    candidate,
                    resident_count,
                })
            })
            .collect::<Result<Vec<_>, KvManagerError>>()
            .map(Vec::into_boxed_slice)
    }

    /// Materializes one exact immutable request view for a preflighted cold
    /// output buffer.
    ///
    /// # Errors
    ///
    /// Returns an error for a stale request/snapshot or unrepresentable token
    /// geometry. No manager state is changed.
    #[cfg(test)]
    pub(super) fn materialize_request_view(
        &self,
        request: RequestLease,
        expected: SnapshotLease,
    ) -> Result<Box<[SnapshotPage]>, KvManagerError> {
        let state = self.request(request)?;
        if state.released || state.quarantined {
            return Err(KvManagerError::RequestUnavailable);
        }
        if state.head != expected {
            return Err(KvManagerError::StaleView);
        }
        let snapshot = self.request_snapshot(request)?;
        self.materialize_snapshot_roots(snapshot.boundary, &snapshot.roots)
    }

    pub(super) fn materialize_snapshot_roots(
        &self,
        boundary: u64,
        roots: &[ClassRoot],
    ) -> Result<Box<[SnapshotPage]>, KvManagerError> {
        self.materialize_roots(boundary, None, roots)
    }

    pub(super) fn materialize_attention_roots(
        &self,
        previous_boundary: u64,
        target_boundary: u64,
        roots: &[ClassRoot],
    ) -> Result<Box<[SnapshotPage]>, KvManagerError> {
        self.materialize_roots(target_boundary, Some(previous_boundary), roots)
    }

    fn materialize_roots(
        &self,
        boundary: u64,
        attention_previous_boundary: Option<u64>,
        roots: &[ClassRoot],
    ) -> Result<Box<[SnapshotPage]>, KvManagerError> {
        if roots.len() != self.classes.len() {
            return Err(KvManagerError::Invariant("snapshot class cardinality"));
        }
        let resident_count = roots.iter().try_fold(0_usize, |count, root| {
            count
                .checked_add(root.entries.len())
                .ok_or(KvManagerError::ArithmeticOverflow("resident count"))
        })?;
        let mut pages = Vec::with_capacity(resident_count);
        for (class, root) in self.classes.iter().copied().zip(roots.iter()) {
            let mirror_boundary = root.mirror_boundary(boundary);
            for entry in root.entries.iter().copied() {
                let (token_begin, token_end) = self.clamped_page_token_span(
                    entry.logical_ordinal,
                    mirror_boundary,
                    "materialized token begin",
                    "empty materialized token span",
                )?;
                let semantic_start = if root.is_dense() {
                    class.semantic_start(boundary)
                } else {
                    0
                };
                let required_start =
                    attention_previous_boundary.map_or(semantic_start, |previous| {
                        if root.is_dense() {
                            class.semantic_candidate_start(previous)
                        } else {
                            0
                        }
                    });
                if token_end <= required_start {
                    continue;
                }
                let visible_begin = semantic_start.max(token_begin).min(token_end);
                pages.push(SnapshotPage {
                    class_id: entry.class_id,
                    backend_domain: entry.backend_domain,
                    logical_ordinal: entry.logical_ordinal,
                    temporal_cell_index: entry.temporal_cell_index,
                    temporal_cycle: entry.temporal_cycle,
                    page: entry.page,
                    backend_index: entry.backend_index,
                    valid_token_count: u32::try_from(token_end - token_begin).map_err(|_| {
                        KvManagerError::ArithmeticOverflow("materialized valid tokens")
                    })?,
                    visible_token_offset: u32::try_from(visible_begin - token_begin).map_err(
                        |_| KvManagerError::ArithmeticOverflow("materialized visible offset"),
                    )?,
                    visible_token_count: u32::try_from(token_end - visible_begin).map_err(
                        |_| KvManagerError::ArithmeticOverflow("materialized visible tokens"),
                    )?,
                });
            }
        }
        Ok(pages.into_boxed_slice())
    }

    /// Cold-materializes an ordered batch only after every expected snapshot
    /// head has been collectively revalidated.
    ///
    /// # Errors
    ///
    /// Rejects empty/duplicate input, stale heads, unavailable requests, or
    /// unrepresentable geometry without returning partial materialization.
    #[cfg(test)]
    pub(super) fn materialize_request_views_batch(
        &self,
        items: &[(RequestLease, SnapshotLease)],
    ) -> Result<Box<[MaterializedRequestView]>, KvManagerError> {
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut seen = BTreeSet::new();
        let views = items
            .iter()
            .copied()
            .map(|(request, expected)| {
                if !seen.insert(request) {
                    return Err(KvManagerError::DuplicateRequest);
                }
                let view = self.request_view(request)?;
                if view.snapshot != expected {
                    return Err(KvManagerError::StaleView);
                }
                Ok(view)
            })
            .collect::<Result<Vec<_>, KvManagerError>>()?;
        items
            .iter()
            .copied()
            .zip(views)
            .map(|((request, expected), view)| {
                Ok(MaterializedRequestView {
                    view,
                    pages: self.materialize_request_view(request, expected)?,
                })
            })
            .collect::<Result<Vec<_>, KvManagerError>>()
            .map(Vec::into_boxed_slice)
    }

    /// Acquires a non-empty batch of request identities atomically.
    ///
    /// # Errors
    ///
    /// Returns an error with no state change when the batch is empty or the
    /// fixed request arena cannot satisfy the entire batch.
    pub(super) fn acquire_request_views(
        &mut self,
        request_count: usize,
    ) -> Result<Box<[RequestView]>, KvManagerError> {
        if request_count == 0 {
            return Err(KvManagerError::EmptyBatch);
        }
        let planned = self.requests.plan_many(request_count)?;
        let planned_snapshots = self.snapshots.plan_many(request_count)?;
        let mut views = Vec::with_capacity(request_count);
        for (&slot, &snapshot_slot) in planned.iter().zip(&planned_snapshots) {
            let request = RequestLease {
                engine_epoch: self.engine_epoch,
                slot: slot.0,
                generation: slot.1,
            };
            let head = SnapshotLease {
                engine_epoch: self.engine_epoch,
                slot: snapshot_slot.0,
                generation: snapshot_slot.1,
            };
            self.snapshots.insert_planned(
                snapshot_slot,
                RequestSnapshot {
                    boundary: 0,
                    view_version: ViewVersion(0),
                    roots: (0..self.classes.len())
                        .map(|_| ClassRoot {
                            entries: PersistentRootEntries::default(),
                            tokens: PersistentTokenTable::default(),
                            layout: RootLayout::Dense,
                            resident_tokens: 0,
                        })
                        .collect::<Vec<_>>()
                        .into(),
                },
            );
            self.requests.insert_planned(
                slot,
                RequestState {
                    head,
                    pending_step: None,
                    inflight_submission: None,
                    pending_relocation: None,
                    last_completion_domain: 0,
                    last_completion_value: 0,
                    released: false,
                    quarantined: false,
                },
            );
            debug_assert_eq!(request.engine_epoch, self.engine_epoch);
            views.push(RequestView {
                request,
                snapshot: head,
                view_version: ViewVersion(0),
                boundary: 0,
                resident_count: 0,
            });
        }
        Ok(views.into_boxed_slice())
    }

    /// Acquires request identities together with their independently owned
    /// empty snapshot heads.
    ///
    /// # Errors
    ///
    /// Returns an error with no state change when the batch is empty or the
    /// fixed request/snapshot arenas cannot satisfy the entire batch.
    pub fn acquire_requests_batch(
        &mut self,
        request_count: usize,
    ) -> Result<Box<[RequestView]>, KvManagerError> {
        self.acquire_request_views(request_count)
    }

    #[cfg(test)]
    pub(super) fn acquire_request_leases_for_test(
        &mut self,
        request_count: usize,
    ) -> Result<Box<[RequestLease]>, KvManagerError> {
        self.acquire_request_views(request_count).map(|views| {
            views
                .iter()
                .map(|view| view.request)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }

    /// Atomically forks immutable source heads into distinct empty targets.
    /// Persistent root bundles are shared, while every target receives an
    /// independent generation-checked [`SnapshotLease`] and contributes its
    /// own exact request reference to every resident page.
    ///
    /// # Errors
    ///
    /// Rejects empty input, duplicate/overlapping targets, stale expected
    /// heads, busy or unavailable requests, non-empty targets, reference
    /// overflow, or snapshot staging exhaustion with no mutation.
    ///
    /// # Panics
    ///
    /// Panics only if exclusive manager state changes after collective
    /// preflight, which indicates an internal invariant violation.
    #[allow(clippy::too_many_lines)]
    pub fn fork_requests_batch(
        &mut self,
        items: &[RequestForkItem],
    ) -> Result<Box<[ForkedRequest]>, KvManagerError> {
        if self
            .classes
            .iter()
            .copied()
            .any(|class| !class.uses_compiled_residence())
        {
            return Err(KvManagerError::UnsupportedProfile(
                "request fork requires compiled physical residence",
            ));
        }
        if items.is_empty() {
            return Err(KvManagerError::EmptyBatch);
        }
        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for item in items {
            sources.insert(item.source_request);
            if !targets.insert(item.target_empty_request) {
                return Err(KvManagerError::DuplicateRequest);
            }
        }
        if sources.iter().any(|source| targets.contains(source)) {
            return Err(KvManagerError::DuplicateRequest);
        }
        let planned_snapshots = self.snapshots.plan_many(items.len())?;
        let mut page_increments = BTreeMap::<PageLease, u32>::new();
        let mut plans = Vec::with_capacity(items.len());
        for (item, planned) in items.iter().zip(planned_snapshots.iter().copied()) {
            let source_state = self.request(item.source_request)?;
            if source_state.released || source_state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if source_state.head != item.expected_source_head {
                return Err(KvManagerError::StaleView);
            }
            if source_state.busy() {
                return Err(KvManagerError::RequestBusy);
            }
            let source_snapshot = self.request_snapshot(item.source_request)?;
            if source_snapshot.roots.len() != self.classes.len() {
                return Err(KvManagerError::Invariant("snapshot class cardinality"));
            }
            let target_state = self.request(item.target_empty_request)?;
            if target_state.released || target_state.quarantined {
                return Err(KvManagerError::RequestUnavailable);
            }
            if target_state.head != item.expected_target_head {
                return Err(KvManagerError::StaleView);
            }
            if target_state.busy() {
                return Err(KvManagerError::RequestBusy);
            }
            let target_snapshot = self.request_snapshot(item.target_empty_request)?;
            if target_snapshot.boundary != 0 || !target_snapshot.is_empty() {
                return Err(KvManagerError::AttachRequiresEmptyRequest);
            }
            let target_version = ViewVersion(
                target_snapshot
                    .view_version
                    .0
                    .checked_add(1)
                    .ok_or(KvManagerError::ViewVersionExhausted)?,
            );
            let resident_count = u32::try_from(source_snapshot.resident_count())
                .map_err(|_| KvManagerError::ArithmeticOverflow("resident count"))?;
            let pages =
                self.materialize_snapshot_roots(source_snapshot.boundary, &source_snapshot.roots)?;
            for entry in Self::root_entries(
                &source_snapshot.roots,
                source_snapshot.boundary,
                self.page_tokens,
            ) {
                let increment = page_increments.entry(entry.page).or_default();
                *increment = increment
                    .checked_add(1)
                    .ok_or(KvManagerError::ReferenceCountOverflow(entry.page.page_id))?;
            }
            let snapshot = SnapshotLease {
                engine_epoch: self.engine_epoch,
                slot: planned.0,
                generation: planned.1,
            };
            plans.push((
                item.source_request,
                item.target_empty_request,
                target_state.head,
                snapshot,
                planned,
                target_version,
                source_snapshot.boundary,
                resident_count,
                Arc::clone(&source_snapshot.roots),
                pages,
            ));
        }
        for (lease, increment) in &page_increments {
            let class = self.runtime_class(self.page(lease.page_id)?.class_id)?;
            self.validate_page_lease(class, *lease)?;
            let page = self.page(lease.page_id)?;
            if page.phase != PagePhase::Live
                || page.reader_pins != 0
                || page.writer.is_some()
                || page.generation != lease.generation
                || page.request_refs == 0
            {
                return Err(KvManagerError::StalePage);
            }
            page.request_refs
                .checked_add(*increment)
                .ok_or(KvManagerError::ReferenceCountOverflow(lease.page_id))?;
        }
        for (lease, increment) in page_increments {
            self.page_mut(lease.page_id)
                .expect("fork preflight retained source page")
                .request_refs += increment;
        }
        for (_, _, _, _, planned, version, boundary, _, roots, _) in &plans {
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
        for (
            source,
            target,
            old_head,
            snapshot,
            _,
            view_version,
            boundary,
            resident_count,
            _,
            pages,
        ) in plans
        {
            self.snapshots
                .remove(old_head.slot, old_head.generation)
                .expect("fork preflight retained empty target snapshot");
            self.request_mut(target)
                .expect("fork preflight retained target request")
                .head = snapshot;
            outputs.push(ForkedRequest {
                source,
                target: MaterializedRequestView {
                    view: RequestView {
                        request: target,
                        snapshot,
                        view_version,
                        boundary,
                        resident_count,
                    },
                    pages,
                },
            });
        }
        Ok(outputs.into_boxed_slice())
    }

    #[must_use]
    pub fn stats(&self) -> ManagerStats {
        self.stats_impl(None)
    }

    fn stats_impl(&self, mut work: Option<&mut CensusWork>) -> ManagerStats {
        let mut stats = ManagerStats {
            active_requests: self.requests.active_len() as u64,
            active_snapshots: self.snapshots.active_len() as u64,
            active_prefixes: self.active_prefixes,
            evicted_prefixes: self.evicted_prefixes,
            prepared_steps: self.prepared_steps,
            submitted_steps: self.submitted_steps,
            pending_reclamations: self.reclamations.active_len() as u64,
            ..ManagerStats::default()
        };
        for counts in &self.page_counts {
            if let Some(instrumentation) = work.as_deref_mut() {
                instrumentation.classes += 1;
            }
            stats.free_pages += counts.free;
            stats.reserved_pages += counts.reserved;
            stats.writing_pages += counts.writing;
            stats.active_pages += counts.active - counts.writing;
            stats.retiring_pages += counts.retiring;
            stats.quarantined_pages += counts.quarantined;
            stats.exhausted_pages += counts.exhausted;
            stats.total_request_page_refs += counts.request_refs;
            stats.total_prefix_page_refs += counts.prefix_refs;
            stats.total_reader_pins += counts.reader_pins;
        }
        stats
    }

    #[cfg(test)]
    pub(super) fn stats_instrumented(&self) -> (ManagerStats, CensusWork) {
        let mut work = CensusWork::default();
        let stats = self.stats_impl(Some(&mut work));
        (stats, work)
    }

    /// Returns an immutable per-class physical-page census in class-id order.
    ///
    /// # Panics
    ///
    /// Panics only if a constructor-validated class page range no longer fits
    /// the manager's in-memory page arena, which is an internal invariant.
    #[must_use]
    pub fn arena_stats(&self) -> Box<[ArenaStats]> {
        self.arena_stats_impl(None)
    }

    fn arena_stats_impl(&self, mut work: Option<&mut CensusWork>) -> Box<[ArenaStats]> {
        self.classes
            .iter()
            .zip(&self.page_counts)
            .map(|(class, counts)| {
                if let Some(instrumentation) = work.as_deref_mut() {
                    instrumentation.classes += 1;
                }
                let resident_pages =
                    u64::from(class.backend.page_count) - counts.free - counts.exhausted;
                ArenaStats {
                    engine_epoch: self.engine_epoch,
                    pool_epoch: self.pool_epoch,
                    class_id: class.class_id,
                    backend_domain: class.backend.backend_domain,
                    pool_id: class.backend.pool_id,
                    page_count: class.backend.page_count,
                    first_page_id: class.first_page_id,
                    page_payload_bytes: class.page_payload_bytes,
                    physical_residence: class.physical_residence,
                    resident_pages,
                    resident_bytes: resident_pages * class.page_payload_bytes,
                    free_pages: counts.free,
                    reserved_pages: counts.reserved,
                    writing_pages: counts.writing,
                    active_pages: counts.active - counts.writing,
                    retiring_pages: counts.retiring,
                    quarantined_pages: counts.quarantined,
                    exhausted_pages: counts.exhausted,
                    request_page_refs: counts.request_refs,
                    prefix_page_refs: counts.prefix_refs,
                    reader_pins: counts.reader_pins,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[cfg(test)]
    pub(super) fn arena_stats_instrumented(&self) -> (Box<[ArenaStats]>, CensusWork) {
        let mut work = CensusWork::default();
        let stats = self.arena_stats_impl(Some(&mut work));
        (stats, work)
    }

    pub(super) fn runtime_class(&self, class_id: u16) -> Result<RuntimeClass, KvManagerError> {
        self.classes
            .get(usize::from(class_id))
            .copied()
            .filter(|class| class.class_id == class_id)
            .ok_or(KvManagerError::InvalidClass(class_id))
    }

    pub(super) fn request(&self, request: RequestLease) -> Result<&RequestState, KvManagerError> {
        self.check_request_epoch(request)?;
        self.requests.get(request.slot, request.generation)
    }

    pub(super) fn request_mut(
        &mut self,
        request: RequestLease,
    ) -> Result<&mut RequestState, KvManagerError> {
        self.check_request_epoch(request)?;
        self.requests.get_mut(request.slot, request.generation)
    }

    pub(super) fn request_snapshot(
        &self,
        request: RequestLease,
    ) -> Result<&RequestSnapshot, KvManagerError> {
        let head = self.request(request)?.head;
        self.check_snapshot_epoch(head)?;
        self.snapshots.get(head.slot, head.generation)
    }

    #[cfg(test)]
    pub(super) fn request_snapshot_mut(
        &mut self,
        request: RequestLease,
    ) -> Result<&mut RequestSnapshot, KvManagerError> {
        let head = self.request(request)?.head;
        self.check_snapshot_epoch(head)?;
        self.snapshots.get_mut(head.slot, head.generation)
    }

    pub(super) fn root_entries(
        roots: &[ClassRoot],
        _boundary: u64,
        _page_tokens: u64,
    ) -> Vec<RootEntry> {
        roots
            .iter()
            .flat_map(|root| root.entries.iter().copied())
            .collect()
    }

    pub(super) fn check_request_epoch(&self, request: RequestLease) -> Result<(), KvManagerError> {
        if request.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    pub(super) fn check_snapshot_epoch(
        &self,
        snapshot: SnapshotLease,
    ) -> Result<(), KvManagerError> {
        if snapshot.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    pub(super) fn check_prefix_epoch(&self, prefix: PrefixLease) -> Result<(), KvManagerError> {
        if prefix.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    pub(super) fn check_step_epoch(&self, step: StepLease) -> Result<(), KvManagerError> {
        if step.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    pub(super) fn check_submission_epoch(
        &self,
        submission: SubmissionLease,
    ) -> Result<(), KvManagerError> {
        if submission.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    pub(super) fn check_reclamation_epoch(
        &self,
        reclamation: ReclamationLease,
    ) -> Result<(), KvManagerError> {
        if reclamation.engine_epoch != self.engine_epoch {
            return Err(KvManagerError::WrongEngine);
        }
        Ok(())
    }

    pub(super) fn page(&self, page_id: u32) -> Result<&PageState, KvManagerError> {
        let index = page_id
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(KvManagerError::InvalidPage(page_id))?;
        self.pages
            .get(index)
            .ok_or(KvManagerError::InvalidPage(page_id))
    }

    pub(super) fn page_mut(&mut self, page_id: u32) -> Result<PageMut<'_>, KvManagerError> {
        let index = page_id
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(KvManagerError::InvalidPage(page_id))?;
        let page = self
            .pages
            .get_mut(index)
            .ok_or(KvManagerError::InvalidPage(page_id))?;
        let counts = self
            .page_counts
            .get_mut(usize::from(page.class_id))
            .ok_or(KvManagerError::Invariant("page census class"))?;
        Ok(PageMut::new(page, counts))
    }

    pub(super) fn set_page_phase(
        &mut self,
        page_id: u32,
        target: PagePhase,
    ) -> Result<(), KvManagerError> {
        self.page_mut(page_id)?.phase = target;
        Ok(())
    }

    pub(super) fn apply_page_reservations(&mut self, step: StepLease, pages: &[PageLease]) {
        for lease in pages {
            let class_id = self
                .page(lease.page_id)
                .expect("planned page id remains valid")
                .class_id;
            let popped = self.free_pages[usize::from(class_id)].pop();
            assert_eq!(
                popped,
                Some(lease.page_id),
                "planned page remains class stack head"
            );
            {
                let mut page = self
                    .page_mut(lease.page_id)
                    .expect("planned page id remains valid");
                debug_assert_eq!(page.phase, PagePhase::Free);
                debug_assert_eq!(page.generation.checked_add(1), Some(lease.generation));
                page.generation = lease.generation;
            }
            self.set_page_phase(lease.page_id, PagePhase::Reserved { step })
                .expect("planned page remains valid");
        }
    }

    pub(super) fn plan_free_page(
        &self,
        class: RuntimeClass,
        page_cursors: &mut [usize],
    ) -> Result<PageLease, KvManagerError> {
        let class_index = usize::from(class.class_id);
        let cursor = *page_cursors
            .get(class_index)
            .ok_or(KvManagerError::Invariant("class page cursor"))?;
        let free = &self.free_pages[class_index];
        let stack_index = free
            .len()
            .checked_sub(cursor.saturating_add(1))
            .ok_or(KvManagerError::PageCapacityExhausted)?;
        let page_id = free[stack_index];
        let state = self.page(page_id)?;
        if state.class_id != class.class_id
            || state.phase != PagePhase::Free
            || state.request_refs != 0
            || state.prefix_refs != 0
            || state.reader_pins != 0
            || state.writer.is_some()
        {
            return Err(KvManagerError::Invariant("free-page stack state"));
        }
        let generation = state
            .generation
            .checked_add(1)
            .ok_or(KvManagerError::PageCapacityExhausted)?;
        page_cursors[class_index] = cursor
            .checked_add(1)
            .ok_or(KvManagerError::ArithmeticOverflow("batch page cursor"))?;
        Ok(PageLease {
            engine_epoch: self.engine_epoch,
            pool_epoch: self.pool_epoch,
            generation,
            page_id,
            pool_id: class.backend.pool_id,
        })
    }

    pub(super) fn tail_is_exclusive(&self, entry: RootEntry) -> Result<bool, KvManagerError> {
        let class = self.runtime_class(entry.class_id)?;
        self.validate_page_lease(class, entry.page)?;
        let page = self.page(entry.page.page_id)?;
        if page.generation != entry.page.generation
            || page.phase != PagePhase::Live
            || page.reader_pins != 0
            || page.writer.is_some()
            || page.request_refs == 0
        {
            return Err(KvManagerError::StalePage);
        }
        Ok(page.request_refs == 1 && page.prefix_refs == 0)
    }

    pub(super) fn root_entry_for_page(
        &self,
        class: RuntimeClass,
        ordinal: u64,
        page: PageLease,
    ) -> Result<RootEntry, KvManagerError> {
        self.validate_page_lease(class, page)?;
        let (cell, cycle) = class.temporal_address(ordinal);
        let backend_index = class.backend_index(page.page_id)?;
        Ok(RootEntry {
            class_id: class.class_id,
            backend_domain: class.backend.backend_domain,
            logical_ordinal: ordinal,
            temporal_cell_index: cell,
            temporal_cycle: cycle,
            page,
            backend_index,
        })
    }

    pub(super) fn validate_page_lease(
        &self,
        class: RuntimeClass,
        page: PageLease,
    ) -> Result<(), KvManagerError> {
        if page.engine_epoch != self.engine_epoch
            || page.pool_epoch != self.pool_epoch
            || page.pool_id != class.backend.pool_id
            || !class.contains_page(page.page_id)
            || self.page(page.page_id)?.class_id != class.class_id
        {
            return Err(KvManagerError::WrongPageArena);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn device_entry(
        &self,
        root: RootEntry,
        access_flags: u32,
        valid: u64,
        visible_offset: u64,
        visible: u64,
    ) -> Result<DeviceKvEntry, KvManagerError> {
        let token_begin = root
            .logical_ordinal
            .checked_mul(self.page_tokens)
            .ok_or(KvManagerError::ArithmeticOverflow("entry token begin"))?;
        Ok(DeviceKvEntry {
            class_id: root.class_id,
            backend_domain: root.backend_domain,
            access_flags,
            logical_ordinal: root.logical_ordinal,
            token_begin,
            valid_token_count: u32::try_from(valid)
                .map_err(|_| KvManagerError::ArithmeticOverflow("valid token count"))?,
            visible_token_offset: u32::try_from(visible_offset)
                .map_err(|_| KvManagerError::ArithmeticOverflow("visible token offset"))?,
            visible_token_count: u32::try_from(visible)
                .map_err(|_| KvManagerError::ArithmeticOverflow("visible token count"))?,
            pool_id: root.page.pool_id,
            temporal_cell_index: root.temporal_cell_index,
            temporal_cycle: root.temporal_cycle,
            pool_epoch: root.page.pool_epoch,
            page_generation: root.page.generation,
            backend_index: root.backend_index,
            page_id: root.page.page_id,
            reserved: 0,
        })
    }
}

fn compile_manager_profile(
    plan: &CompiledKvPlan,
    config: ManagerConfig,
    backends: &[BackendArenaRegistration],
    physical_residence: PhysicalResidencePolicy,
) -> Result<Vec<RuntimeClass>, KvManagerError> {
    if plan.page_tokens != CANONICAL_PAGE_TOKENS {
        return Err(KvManagerError::UnsupportedProfile(
            "page_tokens must equal 16",
        ));
    }
    if plan.classes.is_empty() || plan.classes.len() != backends.len() {
        return Err(KvManagerError::InvalidConfiguration);
    }
    if plan
        .classes
        .iter()
        .any(|class| class.spec.retention == RetentionKind::Chunked)
        && plan.classes.len() != 1
    {
        return Err(KvManagerError::UnsupportedProfile(
            "chunked retention requires one whole-domain class",
        ));
    }
    if config.maximum_requests == 0
        || config.maximum_operations == 0
        || config.maximum_prefixes == 0
        || config.maximum_reclamations == 0
        || config.maximum_step_tokens == 0
    {
        return Err(KvManagerError::InvalidConfiguration);
    }
    let layout = plan
        .layout_program()
        .map_err(|_| KvManagerError::UnsupportedProfile("invalid compiled layout"))?;
    if layout.classes.len() != plan.classes.len() {
        return Err(KvManagerError::UnsupportedProfile(
            "compiled layout class count mismatch",
        ));
    }
    validate_backend_registrations(backends)?;

    if physical_residence == PhysicalResidencePolicy::RequestLifetime
        && plan
            .classes
            .iter()
            .any(|class| class.spec.retention == RetentionKind::Chunked)
    {
        return Err(KvManagerError::UnsupportedProfile(
            "request-lifetime residence does not support resettable classes",
        ));
    }

    let mut next_page_id = 1_u64;
    let mut runtime = Vec::with_capacity(plan.classes.len());
    for (index, (class, class_layout)) in plan.classes.iter().zip(&layout.classes).enumerate() {
        let class_id = u16::try_from(index).map_err(|_| {
            KvManagerError::UnsupportedProfile("retention class count exceeds u16 class ids")
        })?;
        let backend = *backends
            .iter()
            .find(|backend| backend.class_id == class_id)
            .ok_or(KvManagerError::InvalidConfiguration)?;
        let validated = validate_class_program(class, class_layout)?;
        let page_payload_bytes = class
            .spec
            .bytes_per_token_per_layer
            .checked_mul(
                u64::try_from(class.spec.layers.len())
                    .map_err(|_| KvManagerError::ArithmeticOverflow("class layer count"))?,
            )
            .and_then(|bytes| bytes.checked_mul(plan.page_tokens))
            .ok_or(KvManagerError::ArithmeticOverflow(
                "class page payload bytes",
            ))?;
        page_payload_bytes
            .checked_mul(u64::from(backend.page_count))
            .ok_or(KvManagerError::ArithmeticOverflow("class arena bytes"))?;
        if u64::from(backend.page_count) < validated.minimum_pages {
            return Err(KvManagerError::InvalidConfiguration);
        }
        let first_page_id = u32::try_from(next_page_id)
            .map_err(|_| KvManagerError::ArithmeticOverflow("global page id"))?;
        next_page_id = next_page_id
            .checked_add(u64::from(backend.page_count))
            .ok_or(KvManagerError::ArithmeticOverflow("global page id"))?;
        if next_page_id > u64::from(u32::MAX) + 1 {
            return Err(KvManagerError::ArithmeticOverflow("global page id"));
        }
        runtime.push(RuntimeClass {
            class_id,
            page_payload_bytes,
            retention: class.spec.retention,
            physical_residence,
            window_tokens: validated.window_tokens,
            period_blocks: validated.period_blocks,
            chunk_tokens: validated.chunk_tokens,
            blocks_per_epoch: validated.blocks_per_epoch,
            backend,
            first_page_id,
        });
    }
    let total_pages = next_page_id - 1;
    if u64::from(config.maximum_reclamations) < total_pages {
        return Err(KvManagerError::InvalidConfiguration);
    }
    Ok(runtime)
}

fn validate_backend_registrations(
    backends: &[BackendArenaRegistration],
) -> Result<(), KvManagerError> {
    let mut pool_ids = BTreeSet::new();
    let mut backend_classes = BTreeSet::new();
    let mut backend_ranges = Vec::<(u16, u64, u64)>::with_capacity(backends.len());
    for backend in backends {
        let Some(last_backend_index) = backend
            .backend_base_index
            .checked_add(u64::from(backend.page_count.saturating_sub(1)))
        else {
            return Err(KvManagerError::InvalidConfiguration);
        };
        if backend.pool_id == 0
            || backend.page_count == 0
            || backend.reserved != 0
            || !pool_ids.insert(backend.pool_id)
            || !backend_classes.insert(backend.class_id)
            || backend_ranges.iter().any(|&(domain, first, last)| {
                domain == backend.backend_domain
                    && backend.backend_base_index <= last
                    && first <= last_backend_index
            })
        {
            return Err(KvManagerError::InvalidConfiguration);
        }
        backend_ranges.push((
            backend.backend_domain,
            backend.backend_base_index,
            last_backend_index,
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ValidatedClassProgram {
    window_tokens: Option<u64>,
    period_blocks: Option<u64>,
    chunk_tokens: Option<u64>,
    blocks_per_epoch: Option<u64>,
    minimum_pages: u64,
}

pub(super) fn validate_class_program(
    class: &CompiledKvClass,
    class_layout: &ClassLayoutProgram,
) -> Result<ValidatedClassProgram, KvManagerError> {
    if class.block_domain != BlockDomain::all()
        || class_layout.block_domain != BlockDomain::all()
        || class.kv_head_range.is_some()
        || class_layout.kv_head_range.is_some()
        || class.source_state.is_some()
    {
        return Err(KvManagerError::UnsupportedProfile(
            "canonical manager requires whole-domain layer classes",
        ));
    }
    match class.spec.retention {
        RetentionKind::Full => validate_full_class_program(class, class_layout),
        RetentionKind::Sliding => validate_sliding_class_program(class, class_layout),
        RetentionKind::Chunked => validate_chunked_class_program(class, class_layout),
    }
}

fn validate_full_class_program(
    class: &CompiledKvClass,
    class_layout: &ClassLayoutProgram,
) -> Result<ValidatedClassProgram, KvManagerError> {
    if class.spec.window_tokens.is_some()
        || class.slot_count.is_some()
        || !matches!(class_layout.address, AddressProgram::AppendOnly)
        || !retirement_program_matches(RetentionKind::Full, None, &class_layout.retirement)
    {
        return Err(KvManagerError::UnsupportedProfile(
            "full retention requires append-only addressing",
        ));
    }
    Ok(ValidatedClassProgram {
        window_tokens: None,
        period_blocks: None,
        chunk_tokens: None,
        blocks_per_epoch: None,
        minimum_pages: 1,
    })
}

fn validate_sliding_class_program(
    class: &CompiledKvClass,
    class_layout: &ClassLayoutProgram,
) -> Result<ValidatedClassProgram, KvManagerError> {
    let window = class
        .spec
        .window_tokens
        .filter(|window| *window > 0)
        .ok_or(KvManagerError::UnsupportedProfile(
            "sliding retention requires a positive window",
        ))?;
    let history = window - 1;
    let expected_period = 1_u64
        .checked_add(history / CANONICAL_PAGE_TOKENS)
        .and_then(|blocks| blocks.checked_add(u64::from(history % CANONICAL_PAGE_TOKENS != 0)))
        .ok_or(KvManagerError::ArithmeticOverflow("periodic block count"))?;
    if class.slot_count != Some(expected_period)
        || !matches!(
            class_layout.address,
            AddressProgram::Periodic { period_blocks } if period_blocks == expected_period
        )
        || !retirement_program_matches(
            RetentionKind::Sliding,
            Some(history),
            &class_layout.retirement,
        )
    {
        return Err(KvManagerError::UnsupportedProfile(
            "sliding capacity does not match periodic semantics",
        ));
    }
    Ok(ValidatedClassProgram {
        window_tokens: Some(window),
        period_blocks: Some(expected_period),
        chunk_tokens: None,
        blocks_per_epoch: None,
        minimum_pages: expected_period,
    })
}

fn validate_chunked_class_program(
    class: &CompiledKvClass,
    class_layout: &ClassLayoutProgram,
) -> Result<ValidatedClassProgram, KvManagerError> {
    let chunk_tokens =
        class
            .chunk_tokens
            .filter(|value| *value > 0)
            .ok_or(KvManagerError::UnsupportedProfile(
                "chunked retention requires a positive chunk size",
            ))?;
    if !chunk_tokens.is_multiple_of(CANONICAL_PAGE_TOKENS) {
        return Err(KvManagerError::UnsupportedProfile(
            "chunk size must be page aligned",
        ));
    }
    let blocks_per_epoch = chunk_tokens / CANONICAL_PAGE_TOKENS;
    if class.spec.window_tokens.is_some()
        || class.slot_count != Some(blocks_per_epoch)
        || class_layout.minimum_slots_per_request != Some(blocks_per_epoch)
        || !matches!(
            class_layout.address,
            AddressProgram::ResettableArena { blocks_per_epoch: actual }
                if actual == blocks_per_epoch
        )
        || !matches!(
            class_layout.retirement,
            RetirementProgram::EpochEnd { blocks_per_epoch: actual }
                if actual == blocks_per_epoch
        )
    {
        return Err(KvManagerError::UnsupportedProfile(
            "chunked capacity does not match resettable epoch semantics",
        ));
    }
    Ok(ValidatedClassProgram {
        window_tokens: None,
        period_blocks: None,
        chunk_tokens: Some(chunk_tokens),
        blocks_per_epoch: Some(blocks_per_epoch),
        minimum_pages: blocks_per_epoch,
    })
}

fn retirement_program_matches(
    retention: RetentionKind,
    history_tokens: Option<u64>,
    program: &RetirementProgram,
) -> bool {
    match (retention, history_tokens, program) {
        (RetentionKind::Full, None, RetirementProgram::Never) => true,
        (
            RetentionKind::Sliding,
            Some(expected),
            RetirementProgram::BlockEndPlus { offset_tokens },
        ) => *offset_tokens == expected,
        _ => false,
    }
}
