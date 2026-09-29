use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

#[derive(Clone, Copy, Default)]
struct GateState {
    generation: u64,
    partitioned: bool,
    downstream_delay: Duration,
}

pub(crate) struct TcpGate {
    pub(crate) endpoint: String,
    state: watch::Sender<GateState>,
    admission: Arc<Mutex<()>>,
    active: Arc<AtomicUsize>,
    idle: Arc<Notify>,
    listener: JoinHandle<()>,
}

impl TcpGate {
    pub(crate) async fn start(target: &str) -> Self {
        let target: SocketAddr = target
            .strip_prefix("http://")
            .expect("test etcd endpoint uses http")
            .parse()
            .expect("test etcd endpoint is a socket address");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (state, _) = watch::channel(GateState::default());
        let listen_state = state.clone();
        let active = Arc::new(AtomicUsize::new(0));
        let idle = Arc::new(Notify::new());
        let admission = Arc::new(Mutex::new(()));
        let listen_active = Arc::clone(&active);
        let listen_idle = Arc::clone(&idle);
        let listen_admission = Arc::clone(&admission);
        let task = tokio::spawn(async move {
            loop {
                let Ok((client, _)) = listener.accept().await else {
                    return;
                };
                let state = listen_state.subscribe();
                let admission_guard = listen_admission.lock().unwrap();
                if state.borrow().partitioned {
                    drop(client);
                    continue;
                }
                listen_active.fetch_add(1, Ordering::Relaxed);
                let active = Arc::clone(&listen_active);
                let idle = Arc::clone(&listen_idle);
                tokio::spawn(async move {
                    let _guard = ActiveConnection { active, idle };
                    proxy(client, target, state).await;
                });
                drop(admission_guard);
            }
        });
        Self {
            endpoint,
            state,
            admission,
            active,
            idle,
            listener: task,
        }
    }

    pub(crate) fn set_downstream_delay(&self, delay: Duration) {
        self.state
            .send_modify(|state| state.downstream_delay = delay);
    }

    pub(crate) async fn partition(&self) {
        {
            let _admission = self.admission.lock().unwrap();
            self.state.send_modify(|state| {
                state.generation += 1;
                state.partitioned = true;
            });
        }
        wait_for_idle(&self.active, &self.idle).await;
    }

    pub(crate) fn heal(&self, downstream_delay: Duration) {
        let _admission = self.admission.lock().unwrap();
        self.state.send_modify(|state| {
            state.partitioned = false;
            state.downstream_delay = downstream_delay;
        });
    }

    pub(crate) async fn shutdown(self) {
        let Self {
            state,
            admission,
            active,
            idle,
            listener,
            ..
        } = self;
        {
            let _admission = admission.lock().unwrap();
            state.send_modify(|state| {
                state.generation += 1;
                state.partitioned = true;
            });
            listener.abort();
        }
        let _ = listener.await;
        wait_for_idle(&active, &idle).await;
    }
}

struct ActiveConnection {
    active: Arc<AtomicUsize>,
    idle: Arc<Notify>,
}

impl Drop for ActiveConnection {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::Release);
        self.idle.notify_waiters();
    }
}

async fn wait_for_idle(active: &AtomicUsize, idle: &Notify) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let notified = idle.notified();
            if active.load(Ordering::Acquire) == 0 {
                return;
            }
            notified.await;
        }
    })
    .await
    .expect("TCP gate connections did not drain");
}

async fn proxy(client: TcpStream, target: SocketAddr, state: watch::Receiver<GateState>) {
    let generation = state.borrow().generation;
    let mut connect_state = state.clone();
    let server = tokio::select! {
        result = TcpStream::connect(target) => {
            let Ok(server) = result else { return };
            server
        }
        _ = wait_for_disconnect(&mut connect_state, generation) => return,
    };
    let (client_read, client_write) = client.into_split();
    let (server_read, server_write) = server.into_split();
    tokio::select! {
        _ = copy_upstream(client_read, server_write, state.clone(), generation) => {}
        _ = copy_downstream(server_read, client_write, state, generation) => {}
    }
}

async fn wait_for_disconnect(state: &mut watch::Receiver<GateState>, generation: u64) {
    while !disconnected(*state.borrow(), generation) {
        if state.changed().await.is_err() {
            return;
        }
    }
}

async fn copy_upstream(
    mut source: tokio::net::tcp::OwnedReadHalf,
    mut destination: tokio::net::tcp::OwnedWriteHalf,
    mut state: watch::Receiver<GateState>,
    generation: u64,
) {
    let mut buffer = [0u8; 16 * 1024];
    loop {
        tokio::select! {
            changed = state.changed() => {
                if changed.is_err() || disconnected(*state.borrow(), generation) {
                    return;
                }
            }
            read = source.read(&mut buffer) => {
                let Ok(count) = read else { return };
                if count == 0 || destination.write_all(&buffer[..count]).await.is_err() {
                    return;
                }
            }
        }
    }
}

async fn copy_downstream(
    mut source: tokio::net::tcp::OwnedReadHalf,
    mut destination: tokio::net::tcp::OwnedWriteHalf,
    mut state: watch::Receiver<GateState>,
    generation: u64,
) {
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let count = tokio::select! {
            changed = state.changed() => {
                if changed.is_err() || disconnected(*state.borrow(), generation) {
                    return;
                }
                continue;
            }
            read = source.read(&mut buffer) => {
                let Ok(count) = read else { return };
                count
            }
        };
        if count == 0 {
            return;
        }
        let delay = state.borrow().downstream_delay;
        if !delay.is_zero() {
            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                changed = state.changed() => {
                    if changed.is_err() || disconnected(*state.borrow(), generation) {
                        return;
                    }
                }
            }
        }
        if destination.write_all(&buffer[..count]).await.is_err() {
            return;
        }
    }
}

fn disconnected(state: GateState, generation: u64) -> bool {
    state.partitioned || state.generation != generation
}
