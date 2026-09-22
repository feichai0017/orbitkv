use std::time::Duration;

use orbitkv_channel::lifecycle::LifecycleCommand;
use orbitkv_channel::{
    BlockHashes, CacheClient, CallOptions, ChannelError, PublishLayer, PublishRequest, QueryIntent,
    RestoreHandle, RestoreLease, RestoreRequest, RestoreState,
};
use orbitkv_proto::proto::engine::{
    RegisterContextRequest, SessionRequest, TransferMode, UnregisterRequest,
};
use prost::Message;
use pyo3::{
    exceptions::{PyTimeoutError, PyValueError},
    prelude::*,
    types::PySlice,
};

use crate::{OrbitKVError, OrbitKVInternal, query_response};

type PyLeaseLoad = (Vec<u8>, Vec<Vec<Option<u32>>>);

#[derive(PartialEq, Eq)]
#[pyclass(name = "BlockHashes", frozen, eq)]
pub(crate) struct PyBlockHashes(BlockHashes);

#[pymethods]
impl PyBlockHashes {
    #[new]
    fn new(hashes: Vec<Vec<u8>>) -> Self {
        Self(BlockHashes::new(hashes))
    }

    fn __len__(&self) -> usize {
        self.0.as_slice().len()
    }

    fn __getitem__(&self, slice: &Bound<'_, PySlice>) -> PyResult<Self> {
        let indices = slice.indices(self.__len__() as isize)?;
        if indices.step != 1 {
            return Err(PyValueError::new_err(
                "hash views require a unit slice step",
            ));
        }
        self.0
            .slice(indices.start as usize..indices.stop.max(indices.start) as usize)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err("hash view outside its batch"))
    }
}

#[pyclass(name = "RestoreHandle", frozen)]
pub(crate) struct PyRestoreHandle(RestoreHandle);

#[pymethods]
impl PyRestoreHandle {
    #[getter]
    fn operation_id(&self) -> u64 {
        self.0.operation_id
    }
    #[getter]
    fn session_epoch(&self) -> u64 {
        self.0.session_epoch
    }
    #[getter]
    fn key(&self) -> String {
        format!("manager:{}:{}", self.0.session_epoch, self.0.operation_id)
    }
}

#[pyclass(frozen)]
pub(crate) struct RestoreStatus {
    #[pyo3(get)]
    done: bool,
    #[pyo3(get)]
    success: bool,
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl RestoreStatus {
    #[new]
    #[pyo3(signature = (done, success, message=String::new()))]
    fn new(done: bool, success: bool, message: String) -> Self {
        Self {
            done,
            success,
            message,
        }
    }
}

impl From<orbitkv_channel::RestoreResponse> for RestoreStatus {
    fn from(response: orbitkv_channel::RestoreResponse) -> Self {
        Self {
            done: response.state != RestoreState::Pending,
            success: response.state == RestoreState::Succeeded,
            message: response.message,
        }
    }
}

fn client_error(error: ChannelError) -> PyErr {
    match error {
        ChannelError::Lifecycle { code: 1, message } => PyValueError::new_err(message),
        ChannelError::Lifecycle { code: 3, message } => OrbitKVInternal::new_err(message),
        ChannelError::RestoreTimeout { .. } => {
            PyTimeoutError::new_err("OrbitKV GPU restore timed out")
        }
        other => OrbitKVError::new_err(other.to_string()),
    }
}

fn seconds(value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value)
        .map_err(|_| PyValueError::new_err("timeout must be finite and non-negative"))
}

#[pyclass(name = "CacheManagerClient", frozen)]
pub(crate) struct PyCacheManagerClient {
    inner: CacheClient,
}

impl PyCacheManagerClient {
    fn lifecycle_call(
        &self,
        py: Python<'_>,
        command: LifecycleCommand,
        payload: Vec<u8>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.channel().lifecycle(command, &payload))
            .map_err(client_error)
    }
}

#[pymethods]
impl PyCacheManagerClient {
    #[new]
    #[pyo3(signature = (bootstrap_socket, *, timeout_ms=5000, spin_iterations=64))]
    fn new(
        py: Python<'_>,
        bootstrap_socket: String,
        timeout_ms: u64,
        spin_iterations: u32,
    ) -> PyResult<Self> {
        if timeout_ms == 0 {
            return Err(PyValueError::new_err("timeout_ms must be non-zero"));
        }
        let inner = py
            .detach(|| {
                CacheClient::connect(
                    bootstrap_socket,
                    CallOptions {
                        timeout: Duration::from_millis(timeout_ms),
                        spin_iterations,
                    },
                )
            })
            .map_err(client_error)?;
        Ok(Self { inner })
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| self.inner.close());
    }

    #[getter]
    fn transport(&self) -> &'static str {
        "iceoryx2"
    }
    #[getter]
    fn bootstrap_socket(&self) -> String {
        self.inner.socket().to_string_lossy().into_owned()
    }
    #[getter]
    fn service_name(&self) -> String {
        self.inner.channel().service_name().to_owned()
    }
    #[getter]
    fn session_epoch(&self) -> u64 {
        self.inner.channel().session_epoch()
    }
    #[getter]
    fn notification_fd(&self) -> i32 {
        self.inner.channel().notification_fd()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "matches framework registration metadata"
    )]
    #[pyo3(signature = (instance_id, namespace, tp_rank, pp_rank, tp_size, world_size, device_id, layer_names, wrapper_bytes_list, num_blocks_list, bytes_per_block_list, kv_stride_bytes_list, segments_list, transfer_backend, page_first, layer_group_ids=None))]
    fn register_context_batch(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_rank: u32,
        pp_rank: u32,
        tp_size: u32,
        world_size: u32,
        device_id: i32,
        layer_names: Vec<String>,
        wrapper_bytes_list: Vec<Vec<u8>>,
        num_blocks_list: Vec<u64>,
        bytes_per_block_list: Vec<u64>,
        kv_stride_bytes_list: Vec<u64>,
        segments_list: Vec<u32>,
        transfer_backend: &str,
        page_first: bool,
        layer_group_ids: Option<Vec<u32>>,
    ) -> PyResult<(bool, String)> {
        let transfer_mode = match transfer_backend {
            "direct" => TransferMode::Direct,
            "kernel" => TransferMode::Kernel,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown transfer_backend '{other}' (expected 'direct' or 'kernel')"
                )));
            }
        };
        let request = RegisterContextRequest {
            instance_id,
            namespace,
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            tp_rank,
            tp_size,
            world_size,
            device_id,
            layer_names,
            wrapper_bytes: wrapper_bytes_list,
            num_blocks: num_blocks_list,
            bytes_per_block: bytes_per_block_list,
            kv_stride_bytes: kv_stride_bytes_list,
            segments: segments_list,
            pp_rank,
            transfer_mode: transfer_mode as i32,
            page_first,
            layer_group_ids: layer_group_ids.unwrap_or_default(),
        };
        self.lifecycle_call(py, LifecycleCommand::Register, request.encode_to_vec())?;
        Ok((true, String::new()))
    }

    fn health(&self, py: Python<'_>) -> PyResult<(bool, String)> {
        self.lifecycle_call(py, LifecycleCommand::Health, Vec::new())?;
        Ok((true, String::new()))
    }

    fn unregister_context(&self, py: Python<'_>, instance_id: String) -> PyResult<(bool, String)> {
        self.lifecycle_call(
            py,
            LifecycleCommand::Unregister,
            UnregisterRequest { instance_id }.encode_to_vec(),
        )?;
        Ok((true, String::new()))
    }

    fn start_session_watcher(
        &self,
        py: Python<'_>,
        instance_id: String,
        namespace: String,
        tp_size: u32,
        world_size: u32,
    ) -> PyResult<()> {
        self.lifecycle_call(
            py,
            LifecycleCommand::Session,
            SessionRequest {
                instance_id,
                namespace,
                tp_size,
                world_size,
            }
            .encode_to_vec(),
        )
    }

    #[pyo3(signature = (instance_id, block_hashes, req_id, wait_for_full_prefix=false, group_id=0))]
    fn query_prefetch(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        wait_for_full_prefix: bool,
        group_id: u32,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner.query(
                    instance_id,
                    &block_hashes.0,
                    req_id,
                    group_id,
                    QueryIntent::Lookup {
                        wait_for_full_prefix,
                    },
                )
            })
            .map_err(client_error)?;
        query_response(py, response)
    }

    #[pyo3(signature = (instance_id, block_hashes, req_id, group_id=0))]
    fn query_candidates(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        group_id: u32,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner.query(
                    instance_id,
                    &block_hashes.0,
                    req_id,
                    group_id,
                    QueryIntent::Candidates,
                )
            })
            .map_err(client_error)?;
        query_response(py, response)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "native boundary for engine recovery context"
    )]
    fn read_recovery(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
        contract: &crate::recovery::PyRecoveryContract,
        namespace: &str,
        start: u64,
        end: u64,
        group_id: u32,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner.read_recovery(
                    instance_id,
                    &block_hashes.0,
                    req_id,
                    orbitkv_channel::RecoveryRead {
                        contract: &contract.contract,
                        namespace,
                        span: orbitkv_state::TokenRange { start, end },
                        group: group_id,
                    },
                )
            })
            .map_err(client_error)?;
        query_response(py, response)
    }

    fn warm_prefix(
        &self,
        py: Python<'_>,
        instance_id: &str,
        block_hashes: &PyBlockHashes,
        req_id: &str,
    ) -> PyResult<bool> {
        py.detach(|| self.inner.warm_prefix(instance_id, &block_hashes.0, req_id))
            .map_err(client_error)
    }

    #[pyo3(signature = (instance_id, req_id, group_id=0))]
    fn cancel_query(
        &self,
        py: Python<'_>,
        instance_id: &str,
        req_id: &str,
        group_id: u32,
    ) -> PyResult<()> {
        py.detach(|| self.inner.cancel_query(instance_id, req_id, group_id))
            .map_err(client_error)
    }

    fn release(&self, py: Python<'_>, lease: Vec<u8>) -> PyResult<()> {
        py.detach(|| self.inner.release(lease))
            .map_err(client_error)
    }

    fn save(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        pp_rank: u32,
        device_id: i32,
        saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
    ) -> PyResult<(bool, String)> {
        let layers = saves
            .into_iter()
            .map(|(layer_name, block_ids, block_hashes)| PublishLayer {
                layer_name,
                block_ids,
                block_hashes,
            })
            .collect();
        py.detach(|| {
            self.inner.publish(&PublishRequest {
                instance_id,
                tp_rank,
                pp_rank,
                device_id,
                layers,
            })
        })
        .map_err(client_error)?;
        Ok((true, String::new()))
    }

    fn start_restore(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        layer_groups: Vec<Vec<String>>,
        loads: Vec<PyLeaseLoad>,
    ) -> PyResult<PyRestoreHandle> {
        let loads = loads
            .into_iter()
            .map(|(lease, block_ids_by_group)| RestoreLease {
                lease,
                block_ids_by_group,
            })
            .collect();
        py.detach(|| {
            self.inner.start_restore(&RestoreRequest {
                instance_id,
                tp_rank,
                device_id,
                layer_groups,
                loads,
            })
        })
        .map(PyRestoreHandle)
        .map_err(client_error)
    }

    fn poll_restore(&self, py: Python<'_>, handle: &PyRestoreHandle) -> PyResult<RestoreStatus> {
        py.detach(|| self.inner.poll_restore(handle.0))
            .map(Into::into)
            .map_err(client_error)
    }

    #[pyo3(signature = (handle, *, timeout))]
    fn wait_restore(
        &self,
        py: Python<'_>,
        handle: &PyRestoreHandle,
        timeout: f64,
    ) -> PyResult<RestoreStatus> {
        let timeout = seconds(timeout)?;
        py.detach(|| self.inner.wait_restore(handle.0, timeout))
            .map(Into::into)
            .map_err(client_error)
    }

    #[pyo3(signature = (*, timeout=0.0))]
    fn restore_completions_ready(&self, py: Python<'_>, timeout: f64) -> PyResult<bool> {
        let timeout = seconds(timeout)?;
        py.detach(|| self.inner.restore_completions_ready(timeout))
            .map_err(client_error)
    }
}
