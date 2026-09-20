//! Transport-independent registration and instance lifecycle.
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use log::{info, warn};
use orbitkv_core::{EngineError, OrbitKVEngine, TransferMode};
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::cache::session::SessionRegistry;
use crate::registry::RegistryHandle;

pub(crate) struct SessionSpec {
    pub(crate) instance_id: String,
    pub(crate) namespace: String,
    pub(crate) tp_size: u32,
    pub(crate) world_size: u32,
}

pub(crate) struct Registration {
    pub(crate) instance_id: String,
    pub(crate) namespace: String,
    pub(crate) tp_rank: u32,
    pub(crate) pp_rank: u32,
    pub(crate) tp_size: u32,
    pub(crate) world_size: u32,
    pub(crate) device_id: i32,
    pub(crate) layer_names: Vec<String>,
    pub(crate) wrapper_bytes: Vec<Vec<u8>>,
    pub(crate) num_blocks: Vec<u64>,
    pub(crate) bytes_per_block: Vec<u64>,
    pub(crate) kv_stride_bytes: Vec<u64>,
    pub(crate) segments: Vec<u32>,
    pub(crate) client_version: String,
    pub(crate) transfer_mode: TransferMode,
    pub(crate) page_first: bool,
    pub(crate) layer_group_ids: Vec<u32>,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(crate) struct ControlError {
    pub(crate) code: u16,
    pub(crate) message: String,
}
impl ControlError {
    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self {
            code: 1,
            message: message.into(),
        }
    }
    fn failed_precondition(message: impl Into<String>) -> Self {
        Self {
            code: 2,
            message: message.into(),
        }
    }
    fn internal(message: impl Into<String>) -> Self {
        Self {
            code: 3,
            message: message.into(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct LifecycleService {
    engine: Arc<OrbitKVEngine>,
    registry: RegistryHandle,
    sessions: Arc<SessionRegistry>,
    stopping: Arc<tokio::sync::RwLock<bool>>,
    locks: Arc<parking_lot::Mutex<HashMap<String, Weak<Mutex<()>>>>>,
}
impl LifecycleService {
    pub(crate) fn new(engine: Arc<OrbitKVEngine>, registry: RegistryHandle) -> Self {
        Self {
            engine,
            registry,
            sessions: SessionRegistry::new(),
            stopping: Arc::default(),
            locks: Arc::default(),
        }
    }

    async fn lock_instance(&self, instance_id: &str) -> OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.locks.lock();
            locks.retain(|_, lock| lock.strong_count() > 0);
            let entry = locks.entry(instance_id.to_string()).or_default();
            match entry.upgrade() {
                Some(lock) => lock,
                None => {
                    let lock = Arc::new(Mutex::new(()));
                    *entry = Arc::downgrade(&lock);
                    lock
                }
            }
        };
        lock.lock_owned().await
    }

    pub(crate) async fn open_session(&self, request: SessionSpec) -> Result<u64, ControlError> {
        let stopping = self.stopping.read().await;
        if *stopping {
            return Err(ControlError::failed_precondition(
                "Cache Manager is shutting down",
            ));
        }
        if request.instance_id.is_empty() || request.tp_size == 0 || request.world_size == 0 {
            return Err(ControlError::invalid_argument(
                "session requires instance_id and nonzero topology",
            ));
        }
        let _guard = self.lock_instance(&request.instance_id).await;
        Ok(self.sessions.install(
            request.instance_id,
            request.namespace,
            request.tp_size,
            request.world_size,
        ))
    }

    pub(crate) async fn close_session(&self, instance_id: &str, token: u64) {
        let _guard = self.lock_instance(instance_id).await;
        if self.sessions.take(instance_id, token)
            && let Err(error) = self.cleanup_locked(instance_id).await
        {
            warn!("Session cleanup failed for {instance_id}: {error}");
        }
    }

    pub(crate) async fn shutdown(&self) -> Result<(), ControlError> {
        // Wait for registrations already importing tensors before taking the snapshot.
        let mut stopping = self.stopping.write().await;
        *stopping = true;
        for instance in self.engine.list_instance_ids() {
            self.cleanup(&instance).await?;
        }
        Ok(())
    }

    pub(crate) async fn cleanup(&self, instance_id: &str) -> Result<usize, ControlError> {
        let _guard = self.lock_instance(instance_id).await;
        self.cleanup_locked(instance_id).await
    }

    async fn cleanup_locked(&self, instance_id: &str) -> Result<usize, ControlError> {
        match self.engine.unregister_instance_and_wait(instance_id).await {
            Ok(()) | Err(EngineError::InstanceMissing(_)) => {}
            Err(error) => return Err(Self::map_engine_error(error)),
        }
        let removed = self.registry.drop_instance(instance_id.to_string()).await;
        if removed > 0 {
            info!("Lifecycle cleanup: dropped {removed} CUDA tensors for instance {instance_id}");
        }
        Ok(removed)
    }
    pub(crate) async fn register(&self, req: Registration) -> Result<(), ControlError> {
        let stopping = self.stopping.read().await;
        if *stopping {
            return Err(ControlError::failed_precondition(
                "Cache Manager is shutting down",
            ));
        }
        Self::validate_register_context_request(&req)?;
        let _guard = self.lock_instance(&req.instance_id).await;

        let transfer_mode = req.transfer_mode;

        // Validate array lengths are consistent with each other.
        let batch_len = req.layer_names.len();
        if batch_len == 0
            || req.wrapper_bytes.len() != batch_len
            || req.num_blocks.len() != batch_len
            || req.bytes_per_block.len() != batch_len
            || req.kv_stride_bytes.len() != batch_len
            || req.segments.len() != batch_len
        {
            return Err(ControlError::invalid_argument(format!(
                "all layer arrays must have the same non-zero length (got layer_names={batch_len})"
            )));
        }

        let num_blocks_list: Vec<usize> = req
            .num_blocks
            .into_iter()
            .map(|v| Self::usize_from_u64(v, "num_blocks"))
            .collect::<Result<_, _>>()?;
        let bytes_per_block_list: Vec<usize> = req
            .bytes_per_block
            .into_iter()
            .map(|v| Self::usize_from_u64(v, "bytes_per_block"))
            .collect::<Result<_, _>>()?;
        let kv_stride_bytes_list: Vec<usize> = req
            .kv_stride_bytes
            .into_iter()
            .map(|v| Self::usize_from_u64(v, "kv_stride_bytes"))
            .collect::<Result<_, _>>()?;
        let segments_list: Vec<usize> = req
            .segments
            .into_iter()
            .map(|v| Self::usize_from_u32(v, "segments"))
            .collect::<Result<_, _>>()?;

        let tp_rank = Self::usize_from_u32(req.tp_rank, "tp_rank")?;
        let pp_rank = Self::usize_from_u32(req.pp_rank, "pp_rank")?;
        let tp_size = Self::usize_from_u32(req.tp_size, "tp_size")?;
        let world_size = Self::usize_from_u32(req.world_size, "world_size")?;

        // Materialize tensors and collect data_ptr/size_bytes
        let context_key =
            Self::context_key(&req.instance_id, req.tp_rank, req.pp_rank, req.device_id);
        // Materialize on the dedicated registry thread (GIL + CUDA IPC) and
        // await the result, so this RPC never blocks an async worker. Move
        // the (large) wrapper bytes over; clone the layer names since the
        // engine call below still needs them.
        let layers: Vec<(String, Vec<u8>)> = req
            .layer_names
            .iter()
            .cloned()
            .zip(req.wrapper_bytes)
            .collect();
        let metadatas = self
            .registry
            .register_layers(context_key.clone(), req.device_id, layers)
            .await
            .map_err(|message| {
                ControlError::internal(format!("register tensor failed: {message}"))
            })?;
        let mut data_ptrs = Vec::with_capacity(batch_len);
        let mut size_bytes_list = Vec::with_capacity(batch_len);
        for metadata in &metadatas {
            data_ptrs.push(metadata.data_ptr);
            size_bytes_list.push(metadata.size_bytes);
        }

        // Call engine batch registration
        let layer_group_ids: Option<&[u32]> = if req.layer_group_ids.is_empty() {
            None
        } else {
            Some(&req.layer_group_ids)
        };
        if let Err(err) = self.engine.register_context_layer_batch_strided(
            &req.instance_id,
            &req.namespace,
            req.device_id,
            tp_rank,
            pp_rank,
            tp_size,
            world_size,
            &req.layer_names,
            &data_ptrs,
            &size_bytes_list,
            &num_blocks_list,
            &bytes_per_block_list,
            &kv_stride_bytes_list,
            &segments_list,
            None,
            layer_group_ids,
            transfer_mode,
            req.page_first,
        ) {
            let status = Self::map_engine_error(err);
            let removed = self.registry.drop_context(context_key.clone()).await;
            if removed > 0 {
                warn!(
                    "Rolled back {} CUDA tensor(s) for failed register_context_batch context {}",
                    removed, context_key
                );
            }
            return Err(status);
        }

        Ok(())
    }
    fn context_key(instance_id: &str, tp_rank: u32, pp_rank: u32, device_id: i32) -> String {
        format!("{instance_id}:tp{tp_rank}:pp{pp_rank}:dev{device_id}")
    }

    fn map_engine_error(err: EngineError) -> ControlError {
        match err {
            EngineError::InvalidArgument(_) => ControlError::invalid_argument(err.to_string()),
            EngineError::InstanceMissing(_) | EngineError::WorkerMissing(_, _) => {
                ControlError::failed_precondition(err.to_string())
            }
            EngineError::TopologyMismatch(_) => ControlError::failed_precondition(err.to_string()),
            EngineError::CudaInit(_) | EngineError::Storage(_) | EngineError::Poisoned(_) => {
                ControlError::internal(err.to_string())
            }
        }
    }

    fn usize_from_u64(value: u64, field: &str) -> Result<usize, ControlError> {
        usize::try_from(value).map_err(|_| {
            ControlError::invalid_argument(format!("{field}={value} does not fit into usize"))
        })
    }

    fn usize_from_u32(value: u32, field: &str) -> Result<usize, ControlError> {
        usize::try_from(value).map_err(|_| {
            ControlError::invalid_argument(format!("{field}={value} does not fit into usize"))
        })
    }

    fn validate_device_id(device_id: i32) -> Result<(), ControlError> {
        if device_id < 0 {
            return Err(ControlError::invalid_argument(format!(
                "device_id {device_id} must be >= 0"
            )));
        }
        Ok(())
    }

    pub(crate) fn validate_register_context_request(
        req: &Registration,
    ) -> Result<(), ControlError> {
        if req.instance_id.is_empty() || req.namespace.is_empty() {
            return Err(ControlError::invalid_argument(
                "instance_id and namespace must not be empty",
            ));
        }
        let server_version = env!("CARGO_PKG_VERSION");
        if req.client_version != server_version {
            return Err(ControlError::failed_precondition(format!(
                "OrbitKV version mismatch: client={} server={server_version}",
                if req.client_version.is_empty() {
                    "<missing>"
                } else {
                    &req.client_version
                }
            )));
        }
        Self::validate_device_id(req.device_id)?;
        if req.tp_size == 0 {
            return Err(ControlError::invalid_argument("tp_size must be > 0"));
        }
        if req.world_size == 0 {
            return Err(ControlError::invalid_argument("world_size must be > 0"));
        }
        if req.tp_rank >= req.tp_size {
            return Err(ControlError::invalid_argument(format!(
                "tp_rank {} out of range (tp_size {})",
                req.tp_rank, req.tp_size
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint::ProcessEndpoint;
    use crate::proto::engine::{RegisterContextRequest, SessionRequest};
    use crate::registry::CudaTensorRegistry;
    use orbitkv_common::hll::MultiWindowHllTracker;
    use orbitkv_core::StorageConfig;
    use orbitkv_local::lifecycle::LifecycleCommand;
    use orbitkv_local::{CallOptions, LocalQueryClient, LocalQueryError, QueryBundleRequest};
    use prost::Message;
    use std::time::Duration;
    use tokio::sync::Notify;

    pub(crate) fn test_engine() -> Arc<OrbitKVEngine> {
        Arc::new(OrbitKVEngine::new_with_config(1 << 20, false, StorageConfig::default()).unwrap())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn local_lifecycle_and_cache_control_need_no_grpc() {
        let engine = test_engine();
        let lifecycle = LifecycleService::new(
            Arc::clone(&engine),
            RegistryHandle::spawn(CudaTensorRegistry::empty()),
        );
        let shutdown = Arc::new(Notify::new());
        let id = uuid::Uuid::new_v4();
        let socket = std::env::temp_dir().join(format!("orbitkv-lifecycle-{id}.sock"));
        let mut endpoint = ProcessEndpoint::start(
            format!("orbitkv/test/lifecycle/{id}"),
            42,
            socket.clone(),
            1 << 20,
            1 << 16,
            engine,
            tokio::runtime::Handle::current(),
            Arc::new(std::sync::Mutex::new(MultiWindowHllTracker::new(
                vec![("test".into(), Duration::from_secs(60))],
                4,
            ))),
            Arc::clone(&shutdown),
            lifecycle.clone(),
        )
        .unwrap();
        let (first, second) = tokio::task::spawn_blocking(move || {
            let first = LocalQueryClient::connect(&socket, CallOptions::default()).unwrap();
            first.lifecycle(LifecycleCommand::Health, &[]).unwrap();
            assert!(matches!(
                first.lifecycle(LifecycleCommand::Register, &[0xff]),
                Err(LocalQueryError::Lifecycle { code: 1, .. })
            ));
            let bad_version = RegisterContextRequest {
                instance_id: "inst".into(),
                namespace: "ns".into(),
                client_version: "old".into(),
                ..Default::default()
            };
            assert!(matches!(
                first.lifecycle(LifecycleCommand::Register, &bad_version.encode_to_vec()),
                Err(LocalQueryError::Lifecycle { code: 2, .. })
            ));
            first.lifecycle(LifecycleCommand::Health, &[]).unwrap();
            assert!(
                first
                    .query_bundle(
                        1,
                        &QueryBundleRequest {
                            instance_id: "missing".into(),
                            request_id: "query".into(),
                            block_hashes: vec![],
                            wait_for_full_prefix: true,
                            group_id: 0,
                        }
                    )
                    .is_err()
            );
            let session = SessionRequest {
                instance_id: "inst".into(),
                namespace: "ns".into(),
                tp_size: 1,
                world_size: 1,
            }
            .encode_to_vec();
            first
                .lifecycle(LifecycleCommand::Session, &session)
                .unwrap();
            let second = LocalQueryClient::connect(&socket, CallOptions::default()).unwrap();
            second
                .lifecycle(LifecycleCommand::Session, &session)
                .unwrap();
            (first, second)
        })
        .await
        .unwrap();
        // Old connection teardown must leave the replacement session intact.
        first.close();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(lifecycle.sessions.topology("inst").is_some());
        second.close();
        tokio::time::timeout(Duration::from_secs(2), async {
            while lifecycle.sessions.topology("inst").is_some() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("disconnect must clean up the current session");
        assert!(matches!(
            second.lifecycle(LifecycleCommand::Health, &[]),
            Err(LocalQueryError::SessionRequiresReconnect)
        ));
        endpoint.stop();
        shutdown.notify_waiters();
    }
}
