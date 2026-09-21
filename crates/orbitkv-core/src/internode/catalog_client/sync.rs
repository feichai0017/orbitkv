use super::*;

#[derive(Clone)]
enum Phase {
    Restart,
    Snapshot { cursor: Option<StateKey>, page: u64 },
    Replay { through: u64 },
    Live,
}

pub(super) struct InventorySync {
    shard: usize,
    membership: Arc<MembershipView>,
    target: Option<CatalogConnection>,
    node: String,
    node_id: String,
    epoch: String,
    progress: InventoryStatus,
    generation: u64,
    phase: Phase,
    verified_flush: u64,
    heartbeat_at: Instant,
    retry_at: Instant,
    retry_delay: Duration,
}

impl InventorySync {
    pub(super) fn new(shard: usize, membership: Arc<MembershipView>) -> Self {
        Self {
            shard,
            node: membership.owner().endpoint.clone(),
            node_id: membership.owner().incarnation.to_string(),
            membership,
            target: None,
            epoch: String::new(),
            progress: InventoryStatus::default(),
            generation: 0,
            phase: Phase::Restart,
            verified_flush: 0,
            heartbeat_at: Instant::now(),
            retry_at: Instant::now(),
            retry_delay: MIN_RETRY,
        }
    }

    fn destination(&mut self) -> Result<(CatalogRoute, GrpcClient<Channel>), Status> {
        let owner = self
            .membership
            .catalog_owner(self.shard)
            .ok_or_else(|| Status::unavailable("catalog shard has no available member"))?;
        if self
            .target
            .as_ref()
            .is_none_or(|(cached, _)| cached != &owner)
        {
            self.target = Some((owner.clone(), connect(&owner).map_err(Status::unavailable)?));
            self.phase = Phase::Restart;
            self.progress = InventoryStatus::default();
            self.epoch.clear();
            self.heartbeat_at = Instant::now();
        }
        Ok((
            route(&self.membership, self.shard, &owner),
            self.target.as_ref().expect("connected target").1.clone(),
        ))
    }

    pub(super) async fn run(
        mut self,
        cache: Weak<ReadCache>,
        changed: Arc<Notify>,
        control: Arc<Control>,
        mut shutdown: watch::Receiver<bool>,
        acknowledgements: watch::Sender<Acknowledgement>,
    ) {
        loop {
            if *shutdown.borrow() || cache.strong_count() == 0 {
                break;
            }
            if Instant::now() < self.retry_at {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    _ = tokio::time::sleep_until(self.retry_at) => {},
                }
            }
            if let Err(error) = self.destination() {
                self.progress.ready = false;
                self.publish(&acknowledgements);
                self.failed(&error);
                continue;
            }
            let ticket = control.flush_requests.load(Ordering::Acquire);
            if Instant::now() >= self.heartbeat_at || ticket > self.verified_flush {
                match self.heartbeat(ticket).await {
                    Ok(()) => self.publish(&acknowledgements),
                    Err(err) => {
                        core_metrics().catalog_heartbeat_failures.add(1, &[]);
                        self.failed(&err);
                        continue;
                    }
                }
            }
            let Some(source) = cache.upgrade() else {
                break;
            };
            let step = self.next_operation(&source);
            drop(source);
            match step {
                Ok(Some((operation, next))) => {
                    if let Err(err) = self.send(operation, next, &cache).await {
                        core_metrics().inventory_sync_failures.add(1, &[]);
                        // A lost reply during a scan makes its cursor ambiguous.
                        // Live deltas can resume from the next heartbeat's ACK.
                        if !matches!(self.phase, Phase::Live) {
                            self.phase = Phase::Restart;
                        }
                        self.failed(&err);
                    }
                    self.publish(&acknowledgements);
                    tokio::task::yield_now().await;
                }
                Ok(None) => {
                    tokio::select! {
                        _ = shutdown.changed() => break,
                        _ = changed.notified() => {},
                        _ = control.wake.notified() => {},
                        _ = tokio::time::sleep_until(self.heartbeat_at) => {},
                    }
                }
                Err(err) => {
                    if err.code() == tonic::Code::OutOfRange {
                        core_metrics().inventory_history_gaps.add(1, &[]);
                    }
                    self.phase = Phase::Restart;
                    self.failed(&err);
                }
            }
        }
        if let Ok((route, mut client)) = self.destination()
            && let Err(err) = client
                .unregister_node(timed(UnregisterNodeRequest {
                    route: Some(route),
                    node: self.node.clone(),
                    node_id: self.node_id.clone(),
                }))
                .await
        {
            warn!("Inventory unregister failed: {err}");
            core_metrics().catalog_unregister_failures.add(1, &[]);
        }
        acknowledgements.send_replace(Acknowledgement {
            stopped: true,
            ..Acknowledgement::default()
        });
    }

    fn publish(&self, tx: &watch::Sender<Acknowledgement>) {
        tx.send_replace(Acknowledgement {
            inventory: self.progress,
            verified_flush: self.verified_flush,
            stopped: false,
            catalog: self.target.as_ref().map(|(owner, _)| owner.clone()),
        });
    }

    fn failed(&mut self, error: &Status) {
        warn!("Inventory synchronization will retry: {error}");
        let jitter = Duration::from_millis(rand::random_range(
            0..=self.retry_delay.as_millis() as u64 / 4,
        ));
        self.retry_at = Instant::now() + self.retry_delay + jitter;
        self.heartbeat_at = self.retry_at;
        self.retry_delay = (self.retry_delay * 2).min(MAX_RETRY);
    }

    async fn heartbeat(&mut self, ticket: u64) -> Result<(), Status> {
        let (route, mut client) = self.destination()?;
        let response = client
            .heartbeat_node(timed(HeartbeatNodeRequest {
                route: Some(route),
                node: self.node.clone(),
                node_id: self.node_id.clone(),
            }))
            .await?
            .into_inner();
        let remote: InventoryStatus = response
            .progress
            .ok_or_else(|| Status::data_loss("missing inventory progress"))?
            .into();
        let same_epoch = self.epoch == response.catalog_epoch;
        let same_generation = remote.generation == self.progress.generation;
        let resumable = matches!(self.phase, Phase::Live) && remote.ready && same_generation;
        if !same_epoch || (!resumable && remote != self.progress) || remote.generation == 0 {
            self.phase = Phase::Restart;
        }
        self.generation = self.generation.max(remote.generation);
        self.epoch = response.catalog_epoch;
        self.progress = remote;
        self.verified_flush = ticket;
        self.heartbeat_at = Instant::now()
            + Duration::from_millis((response.stale_after_secs.saturating_mul(1000) / 3).max(100));
        Ok(())
    }

    fn next_operation(
        &mut self,
        cache: &ReadCache,
    ) -> Result<Option<(InventoryOperation, Phase)>, Status> {
        loop {
            match &self.phase {
                Phase::Restart => {
                    self.generation = self
                        .generation
                        .checked_add(1)
                        .ok_or_else(|| Status::out_of_range("inventory generation exhausted"))?;
                    core_metrics().inventory_snapshots_started.add(1, &[]);
                    return Ok(Some((
                        InventoryOperation::Begin {
                            sequence: cache.inventory_sequence(self.shard),
                        },
                        Phase::Snapshot {
                            cursor: None,
                            page: 0,
                        },
                    )));
                }
                Phase::Snapshot { cursor, page } => {
                    if !cache.inventory_covers(self.shard, self.progress.sequence) {
                        return Err(Status::out_of_range(
                            "inventory journal expired during snapshot",
                        ));
                    }
                    let records = cache
                        .inventory_page(self.shard, cursor.as_ref())
                        .map_err(inventory_error)?;
                    if let Some(last) = records.last() {
                        let next = Phase::Snapshot {
                            cursor: Some(last.key.clone()),
                            page: page + 1,
                        };
                        return Ok(Some((
                            InventoryOperation::Snapshot {
                                page: *page,
                                records,
                            },
                            next,
                        )));
                    }
                    self.phase = Phase::Replay {
                        through: cache.inventory_sequence(self.shard),
                    };
                }
                Phase::Replay { through } => {
                    let records = cache
                        .inventory_changes(self.shard, self.progress.sequence, *through)
                        .map_err(inventory_error)?;
                    if records.is_empty() {
                        return Ok(Some((
                            InventoryOperation::Commit { sequence: *through },
                            Phase::Live,
                        )));
                    }
                    return Ok(Some((
                        InventoryOperation::Delta {
                            after: self.progress.sequence,
                            records,
                        },
                        self.phase.clone(),
                    )));
                }
                Phase::Live => {
                    let records = cache
                        .inventory_changes(
                            self.shard,
                            self.progress.sequence,
                            cache.inventory_sequence(self.shard),
                        )
                        .map_err(inventory_error)?;
                    return Ok((!records.is_empty()).then_some((
                        InventoryOperation::Delta {
                            after: self.progress.sequence,
                            records,
                        },
                        Phase::Live,
                    )));
                }
            }
        }
    }

    async fn send(
        &mut self,
        operation: InventoryOperation,
        next: Phase,
        cache: &Weak<ReadCache>,
    ) -> Result<(), Status> {
        let mut expected = self.progress;
        match &operation {
            InventoryOperation::Begin { sequence } => {
                expected = InventoryStatus {
                    generation: self.generation,
                    sequence: *sequence,
                    next_page: 0,
                    ready: false,
                }
            }
            InventoryOperation::Snapshot { page, .. } => expected.next_page = page + 1,
            InventoryOperation::Delta { records, .. } => {
                expected.sequence = records
                    .last()
                    .ok_or_else(|| Status::internal("empty delta"))?
                    .sequence;
            }
            InventoryOperation::Commit { .. } => expected.ready = true,
        }
        let count = match &operation {
            InventoryOperation::Snapshot { records, .. }
            | InventoryOperation::Delta { records, .. } => records.len(),
            _ => 0,
        };
        let commit = matches!(operation, InventoryOperation::Commit { .. });
        let (route, mut client) = self.destination()?;
        let response = client
            .sync_inventory(timed(SyncInventoryRequest {
                route: Some(route),
                node: self.node.clone(),
                node_id: self.node_id.clone(),
                catalog_epoch: self.epoch.clone(),
                generation: self.generation,
                operation: Some(operation.into()),
            }))
            .await?
            .into_inner();
        let actual: InventoryStatus = response
            .progress
            .ok_or_else(|| Status::data_loss("missing inventory acknowledgement"))?
            .into();
        if actual != expected {
            return Err(Status::data_loss("unexpected inventory acknowledgement"));
        }
        self.progress = actual;
        self.phase = next;
        core_metrics().inventory_records_sent.add(count as u64, &[]);
        if commit {
            core_metrics().inventory_snapshots_completed.add(1, &[]);
        }
        if self.progress.ready {
            self.retry_delay = MIN_RETRY;
        }
        if !response.reclaimable.is_empty()
            && let Some(cache) = cache.upgrade()
        {
            cache.mark_reclaimable_records(
                &response
                    .reclaimable
                    .into_iter()
                    .map(Into::into)
                    .collect::<Vec<_>>(),
            );
        }
        Ok(())
    }
}

fn inventory_error(error: InventoryReadError) -> Status {
    match error {
        InventoryReadError::HistoryGap => {
            Status::out_of_range("inventory journal no longer covers directory progress")
        }
        InventoryReadError::RecordTooLarge => {
            Status::resource_exhausted("inventory record exceeds batch byte limit")
        }
    }
}
