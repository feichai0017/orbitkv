use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use log::{error, info};
use orbitkv_local::{CommandCode, LocalServer, Response, StatusCode, TransportError};
use tokio::sync::Notify;

const IDLE_POLL_INTERVAL: Duration = Duration::from_micros(50);

/// Dedicated iceoryx2 endpoint owned by one sidecar process.
///
/// Only lifecycle probes are dispatched today. Data-path commands return
/// `Invalid` until their descriptor arenas and engine handlers are wired.
pub(crate) struct LocalControlEndpoint {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl LocalControlEndpoint {
    pub(crate) fn start(
        service_name: String,
        session_epoch: u64,
        shutdown: Arc<Notify>,
    ) -> Result<Self, TransportError> {
        let server = LocalServer::bind(&service_name)?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread_service = service_name.clone();
        let thread = thread::Builder::new()
            .name("orbitkv-local-control".to_string())
            .spawn(move || {
                info!(
                    "Local control endpoint ready: service={} session_epoch={}",
                    thread_service, session_epoch
                );
                while !thread_stop.load(Ordering::Acquire) {
                    let mut request_shutdown = false;
                    match server.try_serve_for_epoch(session_epoch, |command| {
                        let mut response = Response::ok(command);
                        match command.code {
                            CommandCode::Ping => {
                                response.value0 = command.arg0.wrapping_add(1);
                            }
                            CommandCode::Shutdown => {
                                request_shutdown = true;
                            }
                            CommandCode::QueryBundle
                            | CommandCode::Restore
                            | CommandCode::Publish
                            | CommandCode::Release => {
                                response.status = StatusCode::Invalid;
                            }
                        }
                        response
                    }) {
                        Ok(true) if request_shutdown => {
                            shutdown.notify_waiters();
                            break;
                        }
                        Ok(true) => {}
                        Ok(false) => thread::sleep(IDLE_POLL_INTERVAL),
                        Err(error) => {
                            error!("Local control request failed: {error}");
                            thread::sleep(IDLE_POLL_INTERVAL);
                        }
                    }
                }
                info!("Local control endpoint stopped: service={thread_service}");
            })
            .map_err(|error| TransportError::Thread(error.to_string()))?;

        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            error!("Local control thread panicked during shutdown");
        }
    }
}

impl Drop for LocalControlEndpoint {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use orbitkv_local::{CallOptions, Command, LocalClient, StatusCode};

    use super::*;

    fn service_name() -> String {
        format!(
            "orbitkv/test/server-lifecycle/{}/{}",
            std::process::id(),
            uuid::Uuid::new_v4().as_simple()
        )
    }

    #[test]
    fn endpoint_serves_ping_and_fences_old_sessions() {
        let shutdown = Arc::new(Notify::new());
        let service_name = service_name();
        let mut endpoint =
            LocalControlEndpoint::start(service_name.clone(), 17, Arc::clone(&shutdown)).unwrap();
        let client = LocalClient::connect(&service_name).unwrap();

        let mut ping = Command::ping(1, 17);
        ping.arg0 = 41;
        assert_eq!(
            client.call(ping, CallOptions::default()).unwrap().value0,
            42
        );
        assert_eq!(
            client
                .call(Command::ping(2, 16), CallOptions::default())
                .unwrap()
                .status,
            StatusCode::StaleSession
        );
        assert_eq!(
            client
                .call(
                    Command {
                        code: CommandCode::QueryBundle,
                        request_id: 3,
                        ..Command::ping(3, 17)
                    },
                    CallOptions::default(),
                )
                .unwrap()
                .status,
            StatusCode::Invalid
        );
        endpoint.stop();
    }

    #[test]
    fn shutdown_command_notifies_the_sidecar_lifecycle() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let shutdown = Arc::new(Notify::new());
        let service_name = service_name();
        let mut endpoint =
            LocalControlEndpoint::start(service_name.clone(), 23, Arc::clone(&shutdown)).unwrap();
        let client = LocalClient::connect(&service_name).unwrap();

        runtime.block_on(async {
            let call = tokio::task::spawn_blocking(move || {
                client.call(
                    Command {
                        code: CommandCode::Shutdown,
                        request_id: 3,
                        ..Command::ping(3, 23)
                    },
                    CallOptions::default(),
                )
            });
            tokio::time::timeout(Duration::from_secs(2), shutdown.notified())
                .await
                .unwrap();
            assert_eq!(call.await.unwrap().unwrap().status, StatusCode::Ok);
        });
        endpoint.stop();
    }
}
