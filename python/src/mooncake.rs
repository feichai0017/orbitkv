use crate::{OrbitKVError, u64_to_usize};

use orbitkv_transfer::{
    AUTO_MEMORY_LOCATION, Notification, P2P_METADATA, TransferEngine, TransferOp, TransferSlice,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use std::{ptr::NonNull, sync::Arc, time::Duration};

fn transfer_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    OrbitKVError::new_err(format!("{context}: {error}"))
}

fn py_get<'py, T>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a, 'py, Error = PyErr>,
{
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

fn pointer(value: u64, field: &str) -> PyResult<NonNull<u8>> {
    NonNull::new(value as *mut u8)
        .ok_or_else(|| PyValueError::new_err(format!("{field} must be non-zero")))
}

#[pyclass(name = "MooncakeTransferEngine")]
struct PyMooncakeTransferEngine {
    engine: Arc<TransferEngine>,
    endpoint: String,
}

#[pymethods]
impl PyMooncakeTransferEngine {
    #[new]
    #[pyo3(signature = (*, bind_host, nics=Vec::new()))]
    fn new(bind_host: String, nics: Vec<String>) -> PyResult<Self> {
        if bind_host.is_empty() {
            return Err(PyValueError::new_err("bind_host must not be empty"));
        }
        let local_server_name = format!("{bind_host}:0");
        let engine = TransferEngine::new(P2P_METADATA, &local_server_name, &bind_host, 0, &nics)
            .map_err(|error| transfer_error("Mooncake Transfer Engine init failed", error))?;
        let endpoint = engine
            .local_segment_name()
            .map_err(|error| transfer_error("read Mooncake endpoint failed", error))?;
        Ok(Self {
            engine: Arc::new(engine),
            endpoint,
        })
    }

    #[getter]
    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn register_memory(&self, py: Python<'_>, regions: Vec<Py<PyDict>>) -> PyResult<()> {
        let mut parsed = Vec::with_capacity(regions.len());
        for region in regions {
            let region = region.bind(py);
            let address = py_get(region, "addr")?;
            let length = u64_to_usize(py_get(region, "len")?, "memory region len")?;
            if length == 0 {
                return Err(PyValueError::new_err("memory region len must be positive"));
            }
            let location = region
                .get_item("location")?
                .map(|value| value.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| AUTO_MEMORY_LOCATION.to_string());
            pointer(address, "addr")?;
            parsed.push((address, length, location));
        }
        let engine = Arc::clone(&self.engine);
        py.detach(move || {
            for (address, length, location) in parsed {
                unsafe {
                    engine
                        .register_memory(
                            NonNull::new_unchecked(address as *mut u8),
                            length,
                            &location,
                        )
                        .map_err(|error| transfer_error("register memory failed", error))?;
                }
            }
            Ok(())
        })
    }

    fn unregister_memory(&self, py: Python<'_>, addresses: Vec<u64>) -> PyResult<()> {
        for &address in &addresses {
            pointer(address, "addr")?;
        }
        let engine = Arc::clone(&self.engine);
        py.detach(move || {
            for address in addresses {
                unsafe {
                    engine
                        .unregister_memory(NonNull::new_unchecked(address as *mut u8))
                        .map_err(|error| transfer_error("unregister memory failed", error))?;
                }
            }
            Ok(())
        })
    }

    #[pyo3(signature = (remote_endpoint, slices, timeout_s=30.0, notify_name=None, notify_message=None))]
    fn write(
        &self,
        py: Python<'_>,
        remote_endpoint: String,
        slices: Vec<(u64, u64, u64)>,
        timeout_s: f64,
        notify_name: Option<String>,
        notify_message: Option<String>,
    ) -> PyResult<usize> {
        self.transfer(
            py,
            TransferOp::Write,
            remote_endpoint,
            slices,
            timeout_s,
            notification(notify_name, notify_message)?,
        )
    }

    #[pyo3(signature = (remote_endpoint, slices, timeout_s=30.0))]
    fn read(
        &self,
        py: Python<'_>,
        remote_endpoint: String,
        slices: Vec<(u64, u64, u64)>,
        timeout_s: f64,
    ) -> PyResult<usize> {
        self.transfer(
            py,
            TransferOp::Read,
            remote_endpoint,
            slices,
            timeout_s,
            None,
        )
    }

    fn send_notification(
        &self,
        py: Python<'_>,
        remote_endpoint: String,
        name: String,
        message: String,
    ) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        py.detach(move || {
            engine
                .send_notification(&remote_endpoint, &Notification { name, message })
                .map_err(|error| transfer_error("send notification failed", error))
        })
    }

    fn take_notifications(&self) -> PyResult<Vec<(String, String)>> {
        self.engine
            .take_notifications()
            .map(|notifications| {
                notifications
                    .into_iter()
                    .map(|notification| (notification.name, notification.message))
                    .collect()
            })
            .map_err(|error| transfer_error("take notifications failed", error))
    }

    fn invalidate_segment(&self, remote_endpoint: String) {
        self.engine.invalidate_segment(&remote_endpoint);
    }
}

impl PyMooncakeTransferEngine {
    fn transfer(
        &self,
        py: Python<'_>,
        operation: TransferOp,
        remote_endpoint: String,
        slices: Vec<(u64, u64, u64)>,
        timeout_s: f64,
        notification: Option<Notification>,
    ) -> PyResult<usize> {
        if !timeout_s.is_finite() || timeout_s <= 0.0 {
            return Err(PyValueError::new_err("timeout_s must be positive"));
        }
        let slices = slices
            .into_iter()
            .map(|(local, remote, length)| {
                let length = u64_to_usize(length, "slice length")?;
                if length == 0 {
                    return Err(PyValueError::new_err("slice length must be positive"));
                }
                Ok(TransferSlice {
                    local: pointer(local, "local address")?,
                    remote_address: remote,
                    length,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let engine = Arc::clone(&self.engine);
        py.detach(move || {
            let timeout = Duration::from_secs_f64(timeout_s);
            match notification.as_ref() {
                Some(notification) => engine.submit_and_notify(
                    operation,
                    &remote_endpoint,
                    &slices,
                    timeout,
                    notification,
                ),
                None => engine.submit_and_wait(operation, &remote_endpoint, &slices, timeout),
            }
            .map_err(|error| transfer_error("Mooncake transfer failed", error))
        })
    }
}

fn notification(name: Option<String>, message: Option<String>) -> PyResult<Option<Notification>> {
    match (name, message) {
        (None, None) => Ok(None),
        (Some(name), Some(message)) => Ok(Some(Notification { name, message })),
        _ => Err(PyValueError::new_err(
            "notify_name and notify_message must be provided together",
        )),
    }
}

pub(crate) fn add_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMooncakeTransferEngine>()
}
