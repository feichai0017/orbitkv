use orbitkv_local::lifecycle::LifecycleCommand;
use orbitkv_local::{
    CallOptions, Command as LocalCommand, CommandCode, LocalClient, LocalQueryClient, PublishLayer,
    PublishRequest, QueryBundleRequest, QueryOutcomeCode, RestoreLease, RestoreRequest, StatusCode,
};
use orbitkv_proto::proto::engine::{
    CachePageRequest, RegisterContextRequest, SessionRequest, TransferMode, UnregisterRequest,
};
use prost::Message;
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
};
use std::time::Duration;

#[cfg(feature = "mooncake")]
mod mooncake;

// Custom Python exceptions for error classification
create_exception!(orbitkv, OrbitKVError, PyException);
create_exception!(orbitkv, OrbitKVInternal, OrbitKVError);

type PyLeaseLoad = (Vec<u8>, Vec<Vec<Option<u32>>>);

fn u64_to_usize(value: u64, field: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyRuntimeError::new_err(format!("{field}={value} exceeds usize range")))
}

#[pyclass(frozen)]
struct QueryLoading {}

#[pymethods]
impl QueryLoading {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn __repr__(&self) -> String {
        "QueryLoading()".to_string()
    }
}

#[derive(Clone)]
struct PyQueryLease(Vec<u8>);

impl<'a, 'py> FromPyObject<'a, 'py> for PyQueryLease {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let bytes: Vec<u8> = obj.extract()?;
        Ok(Self(bytes))
    }
}

impl<'py> IntoPyObject<'py> for PyQueryLease {
    type Target = pyo3::types::PyBytes;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(pyo3::types::PyBytes::new(py, &self.0))
    }
}

#[pyclass(frozen)]
struct QueryReady {
    #[pyo3(get)]
    num_hit_blocks: usize,
    lease: PyQueryLease,
    /// Membership queries (group_id > 0) only: indices into the queried
    /// block_hashes whose block is cached; lease block i corresponds to
    /// query position hit_positions[i]. Empty for prefix queries.
    #[pyo3(get)]
    hit_positions: Vec<u32>,
}

#[pymethods]
impl QueryReady {
    #[new]
    #[pyo3(signature = (num_hit_blocks, lease, hit_positions=Vec::new()))]
    fn new(num_hit_blocks: usize, lease: PyQueryLease, hit_positions: Vec<u32>) -> Self {
        Self {
            num_hit_blocks,
            lease,
            hit_positions,
        }
    }

    #[getter]
    fn lease<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        self.lease.clone().into_pyobject(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryReady(num_hit_blocks={}, hits={:?}, has_lease={})",
            self.num_hit_blocks,
            self.hit_positions,
            !self.lease.0.is_empty()
        )
    }
}

#[pyclass]
struct LocalControlClient {
    service_name: String,
    session_epoch: u64,
    options: CallOptions,
    client: LocalClient,
}

#[pyclass(name = "LocalQueryClient")]
struct PyLocalQueryClient {
    inner: LocalQueryClient,
}

impl PyLocalQueryClient {
    fn lifecycle_call(
        &self,
        py: Python<'_>,
        command: LifecycleCommand,
        payload: Vec<u8>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.lifecycle(command, &payload))
            .map_err(|error| match error {
                orbitkv_local::LocalQueryError::Lifecycle { code: 1, message } => {
                    PyValueError::new_err(message)
                }
                orbitkv_local::LocalQueryError::Lifecycle { code: 3, message } => {
                    OrbitKVInternal::new_err(message)
                }
                other => OrbitKVError::new_err(other.to_string()),
            })
    }
}

#[pymethods]
impl PyLocalQueryClient {
    #[new]
    #[pyo3(signature = (bootstrap_socket, timeout_ms=5000, spin_iterations=64))]
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
                LocalQueryClient::connect(
                    bootstrap_socket,
                    CallOptions {
                        timeout: Duration::from_millis(timeout_ms),
                        spin_iterations,
                    },
                )
            })
            .map_err(|error| {
                OrbitKVError::new_err(format!("local query connect failed: {error}"))
            })?;
        Ok(Self { inner })
    }

    fn close(&self) {
        self.inner.close();
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

    fn put_host_page(
        &self,
        py: Python<'_>,
        namespace: String,
        key: Vec<u8>,
        data: Vec<u8>,
    ) -> PyResult<()> {
        let request = CachePageRequest {
            namespace,
            key,
            data,
        };
        self.lifecycle_call(py, LifecycleCommand::PagePut, request.encode_to_vec())
    }

    fn get_host_page<'py>(
        &self,
        py: Python<'py>,
        namespace: String,
        key: Vec<u8>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyBytes>>> {
        let request = CachePageRequest {
            namespace,
            key,
            data: Vec::new(),
        };
        let body = py
            .detach(|| {
                self.inner
                    .lifecycle_bytes(LifecycleCommand::PageGet, &request.encode_to_vec())
            })
            .map_err(|error| OrbitKVError::new_err(format!("host page get failed: {error}")))?;
        match body.split_first() {
            Some((&0, [])) => Ok(None),
            Some((&1, data)) => Ok(Some(pyo3::types::PyBytes::new(py, data))),
            _ => Err(OrbitKVInternal::new_err("invalid host page get response")),
        }
    }

    fn has_host_page(&self, py: Python<'_>, namespace: String, key: Vec<u8>) -> PyResult<bool> {
        let request = CachePageRequest {
            namespace,
            key,
            data: Vec::new(),
        };
        let body = py
            .detach(|| {
                self.inner
                    .lifecycle_bytes(LifecycleCommand::PageExists, &request.encode_to_vec())
            })
            .map_err(|error| OrbitKVError::new_err(format!("host page exists failed: {error}")))?;
        match body.as_slice() {
            [0] => Ok(false),
            [1] => Ok(true),
            _ => Err(OrbitKVInternal::new_err(
                "invalid host page exists response",
            )),
        }
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

    #[getter]
    fn service_name(&self) -> &str {
        self.inner.service_name()
    }

    #[getter]
    fn session_epoch(&self) -> u64 {
        self.inner.session_epoch()
    }

    #[getter]
    fn notification_fd(&self) -> i32 {
        self.inner.notification_fd()
    }

    #[pyo3(signature = (instance_id, block_hashes, req_id, wait_for_full_prefix=false, group_id=0, request_id=1))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python API mirrors the framework-neutral local query contract"
    )]
    fn query_bundle(
        &self,
        py: Python<'_>,
        instance_id: String,
        block_hashes: Vec<Vec<u8>>,
        req_id: String,
        wait_for_full_prefix: bool,
        group_id: u32,
        request_id: u64,
    ) -> PyResult<Py<PyAny>> {
        let response = py
            .detach(|| {
                self.inner.query_bundle(
                    request_id,
                    &QueryBundleRequest {
                        instance_id,
                        request_id: req_id,
                        block_hashes,
                        group_id,
                        wait_for_full_prefix,
                    },
                )
            })
            .map_err(|error| OrbitKVError::new_err(format!("local query failed: {error}")))?;
        match response.outcome {
            QueryOutcomeCode::Loading => Py::new(py, QueryLoading {}).map(|value| value.into_any()),
            QueryOutcomeCode::Ready => Py::new(
                py,
                QueryReady {
                    num_hit_blocks: u64_to_usize(response.num_hit_blocks, "num_hit_blocks")?,
                    lease: PyQueryLease(response.lease),
                    hit_positions: response.hit_positions,
                },
            )
            .map(|value| value.into_any()),
        }
    }

    #[pyo3(signature = (lease, request_id=1))]
    fn release(&self, py: Python<'_>, lease: Vec<u8>, request_id: u64) -> PyResult<()> {
        py.detach(|| self.inner.release(request_id, lease))
            .map_err(|error| OrbitKVError::new_err(format!("local release failed: {error}")))
    }

    #[pyo3(signature = (instance_id, tp_rank, pp_rank, device_id, saves, request_id=1))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python API mirrors the framework-neutral local publish contract"
    )]
    fn publish(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        pp_rank: u32,
        device_id: i32,
        saves: Vec<(String, Vec<u32>, Vec<Vec<u8>>)>,
        request_id: u64,
    ) -> PyResult<()> {
        let layers = saves
            .into_iter()
            .map(|(layer_name, block_ids, block_hashes)| PublishLayer {
                layer_name,
                block_ids,
                block_hashes,
            })
            .collect();
        py.detach(|| {
            self.inner.publish(
                request_id,
                &PublishRequest {
                    instance_id,
                    tp_rank,
                    pp_rank,
                    device_id,
                    layers,
                },
            )
        })
        .map_err(|error| OrbitKVError::new_err(format!("local publish failed: {error}")))
    }

    #[pyo3(signature = (instance_id, tp_rank, device_id, layer_groups, loads, timeout_ms=5000, request_id=1))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python API mirrors the framework-neutral local restore contract"
    )]
    fn restore(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        layer_groups: Vec<Vec<String>>,
        loads: Vec<PyLeaseLoad>,
        timeout_ms: u64,
        request_id: u64,
    ) -> PyResult<()> {
        if timeout_ms == 0 {
            return Err(PyValueError::new_err("timeout_ms must be non-zero"));
        }
        let loads = loads
            .into_iter()
            .map(|(lease, block_ids_by_group)| RestoreLease {
                lease,
                block_ids_by_group,
            })
            .collect();
        py.detach(|| {
            let operation_id = self.inner.restore_submit(
                request_id,
                &RestoreRequest {
                    instance_id,
                    tp_rank,
                    device_id,
                    layer_groups,
                    loads,
                },
            )?;
            self.inner.restore_wait(
                request_id
                    .checked_add(1)
                    .ok_or(orbitkv_local::LocalQueryError::SessionRequiresReconnect)?,
                operation_id,
                Duration::from_millis(timeout_ms),
            )
        })
        .map_err(|error| OrbitKVError::new_err(format!("local restore failed: {error}")))
    }

    #[pyo3(signature = (instance_id, tp_rank, device_id, layer_groups, loads, request_id=1))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python API mirrors the framework-neutral local restore contract"
    )]
    fn restore_submit(
        &self,
        py: Python<'_>,
        instance_id: String,
        tp_rank: u32,
        device_id: i32,
        layer_groups: Vec<Vec<String>>,
        loads: Vec<PyLeaseLoad>,
        request_id: u64,
    ) -> PyResult<u64> {
        let loads = loads
            .into_iter()
            .map(|(lease, block_ids_by_group)| RestoreLease {
                lease,
                block_ids_by_group,
            })
            .collect();
        py.detach(|| {
            self.inner.restore_submit(
                request_id,
                &RestoreRequest {
                    instance_id,
                    tp_rank,
                    device_id,
                    layer_groups,
                    loads,
                },
            )
        })
        .map_err(|error| OrbitKVError::new_err(format!("local restore submit failed: {error}")))
    }

    #[pyo3(signature = (operation_id, request_id=1))]
    fn restore_poll(
        &self,
        py: Python<'_>,
        operation_id: u64,
        request_id: u64,
    ) -> PyResult<(String, String)> {
        let response = py
            .detach(|| self.inner.restore_poll(request_id, operation_id))
            .map_err(|error| {
                OrbitKVError::new_err(format!("local restore poll failed: {error}"))
            })?;
        let state = match response.state {
            orbitkv_local::RestoreState::Pending => "pending",
            orbitkv_local::RestoreState::Succeeded => "succeeded",
            orbitkv_local::RestoreState::Failed => "failed",
        };
        Ok((state.to_string(), response.message))
    }
}

impl LocalControlClient {
    fn call(&self, py: Python<'_>, command: LocalCommand) -> PyResult<orbitkv_local::Response> {
        let response = py
            .detach(|| self.client.call(command, self.options))
            .map_err(|error| OrbitKVError::new_err(format!("local control failed: {error}")))?;
        if response.status != StatusCode::Ok {
            return Err(OrbitKVError::new_err(format!(
                "local control returned {:?}: client_epoch={} manager_epoch={}",
                response.status, self.session_epoch, response.session_epoch
            )));
        }
        Ok(response)
    }
}

#[pymethods]
impl LocalControlClient {
    #[new]
    #[pyo3(signature = (service_name, session_epoch, timeout_ms=5000, spin_iterations=64))]
    fn new(
        service_name: String,
        session_epoch: u64,
        timeout_ms: u64,
        spin_iterations: u32,
    ) -> PyResult<Self> {
        if session_epoch == 0 {
            return Err(PyValueError::new_err("session_epoch must be non-zero"));
        }
        if timeout_ms == 0 {
            return Err(PyValueError::new_err("timeout_ms must be non-zero"));
        }
        let client = LocalClient::connect(&service_name).map_err(|error| {
            OrbitKVError::new_err(format!("local control connect failed: {error}"))
        })?;
        Ok(Self {
            service_name,
            session_epoch,
            options: CallOptions {
                timeout: Duration::from_millis(timeout_ms),
                spin_iterations,
            },
            client,
        })
    }

    #[getter]
    fn service_name(&self) -> &str {
        &self.service_name
    }

    #[getter]
    fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    #[pyo3(signature = (value=0, request_id=1))]
    fn ping(&self, py: Python<'_>, value: u64, request_id: u64) -> PyResult<u64> {
        let mut command = LocalCommand::ping(request_id, self.session_epoch);
        command.arg0 = value;
        Ok(self.call(py, command)?.value0)
    }

    #[pyo3(signature = (request_id=1))]
    fn shutdown(&self, py: Python<'_>, request_id: u64) -> PyResult<()> {
        self.call(
            py,
            LocalCommand {
                code: CommandCode::Shutdown,
                request_id,
                ..LocalCommand::ping(request_id, self.session_epoch)
            },
        )?;
        Ok(())
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn orbitkv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    orbitkv_common::logging::init_stderr("info,orbitkv_core=info");
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<LocalControlClient>()?;
    m.add_class::<PyLocalQueryClient>()?;
    #[cfg(feature = "mooncake")]
    mooncake::add_classes(m)?;
    // Register custom exceptions for error classification
    m.add("OrbitKVError", m.py().get_type::<OrbitKVError>())?;
    m.add("OrbitKVInternal", m.py().get_type::<OrbitKVInternal>())?;
    m.add_class::<QueryLoading>()?;
    m.add_class::<QueryReady>()?;

    Ok(())
}
