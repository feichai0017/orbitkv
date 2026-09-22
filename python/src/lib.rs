use orbitkv_channel::{
    CallOptions, Command as ChannelCommand, CommandCode, QueryOutcomeCode, StatusCode,
    TransportClient,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
};
use std::time::Duration;

#[cfg(feature = "mooncake")]
mod mooncake;

mod client;
mod recovery;

// Custom Python exceptions for error classification
create_exception!(orbitkv, OrbitKVError, PyException);
create_exception!(orbitkv, OrbitKVInternal, OrbitKVError);

fn u64_to_usize(value: u64, field: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyRuntimeError::new_err(format!("{field}={value} exceeds usize range")))
}

#[pyclass(frozen)]
struct QueryLoading {
    #[pyo3(get)]
    admitted: bool,
}

#[pymethods]
impl QueryLoading {
    #[new]
    #[pyo3(signature = (admitted=true))]
    fn new(admitted: bool) -> Self {
        Self { admitted }
    }

    fn __repr__(&self) -> String {
        "QueryLoading()".to_string()
    }
}

#[pyclass(frozen)]
struct QueryCandidates {
    #[pyo3(get)]
    num_hit_blocks: usize,
    #[pyo3(get)]
    hit_positions: Vec<u32>,
}

#[pymethods]
impl QueryCandidates {
    #[new]
    fn new(hit_positions: Vec<u32>) -> Self {
        Self {
            num_hit_blocks: hit_positions.len(),
            hit_positions,
        }
    }
}

fn query_response(
    py: Python<'_>,
    response: orbitkv_channel::QueryBundleResponse,
) -> PyResult<Py<PyAny>> {
    match response.outcome {
        QueryOutcomeCode::Loading | QueryOutcomeCode::Busy => Py::new(
            py,
            QueryLoading {
                admitted: response.outcome == QueryOutcomeCode::Loading,
            },
        )
        .map(|value| value.into_any()),
        QueryOutcomeCode::Candidates => Py::new(
            py,
            QueryCandidates {
                num_hit_blocks: u64_to_usize(response.num_hit_blocks, "num_hit_blocks")?,
                hit_positions: response.hit_positions,
            },
        )
        .map(|value| value.into_any()),
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
struct ChannelProbeClient {
    service_name: String,
    session_epoch: u64,
    options: CallOptions,
    client: TransportClient,
}

impl ChannelProbeClient {
    fn call(&self, py: Python<'_>, command: ChannelCommand) -> PyResult<orbitkv_channel::Response> {
        let response = py
            .detach(|| self.client.call(command, self.options))
            .map_err(|error| OrbitKVError::new_err(format!("channel probe failed: {error}")))?;
        if response.status != StatusCode::Ok {
            return Err(OrbitKVError::new_err(format!(
                "channel probe returned {:?}: client_epoch={} manager_epoch={}",
                response.status, self.session_epoch, response.session_epoch
            )));
        }
        Ok(response)
    }
}

#[pymethods]
impl ChannelProbeClient {
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
        let client = TransportClient::connect(&service_name).map_err(|error| {
            OrbitKVError::new_err(format!("channel probe connect failed: {error}"))
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
        let mut command = ChannelCommand::ping(request_id, self.session_epoch);
        command.arg0 = value;
        Ok(self.call(py, command)?.value0)
    }

    #[pyo3(signature = (request_id=1))]
    fn shutdown(&self, py: Python<'_>, request_id: u64) -> PyResult<()> {
        self.call(
            py,
            ChannelCommand {
                code: CommandCode::Shutdown,
                request_id,
                ..ChannelCommand::ping(request_id, self.session_epoch)
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
    m.add_class::<ChannelProbeClient>()?;
    m.add_class::<client::PyCacheManagerClient>()?;
    m.add_class::<client::PyBlockHashes>()?;
    m.add_class::<client::PyRestoreHandle>()?;
    m.add_class::<client::RestoreStatus>()?;
    #[cfg(feature = "mooncake")]
    mooncake::add_classes(m)?;
    // Register custom exceptions for error classification
    m.add("OrbitKVError", m.py().get_type::<OrbitKVError>())?;
    m.add("OrbitKVInternal", m.py().get_type::<OrbitKVInternal>())?;
    m.add_class::<QueryLoading>()?;
    m.add_class::<QueryReady>()?;
    m.add_class::<QueryCandidates>()?;
    m.add_class::<recovery::PyRecoveryContract>()?;

    Ok(())
}
