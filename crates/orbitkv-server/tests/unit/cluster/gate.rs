use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
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
        let task = tokio::spawn(async move {
            loop {
                let Ok((client, _)) = listener.accept().await else {
                    return;
                };
                let state = listen_state.subscribe();
                if state.borrow().partitioned {
                    drop(client);
                    continue;
                }
                tokio::spawn(proxy(client, target, state));
            }
        });
        Self {
            endpoint,
            state,
            listener: task,
        }
    }

    pub(crate) fn set_downstream_delay(&self, delay: Duration) {
        self.state
            .send_modify(|state| state.downstream_delay = delay);
    }

    pub(crate) async fn partition(&self) {
        self.state.send_modify(|state| {
            state.generation += 1;
            state.partitioned = true;
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    pub(crate) fn heal(&self, downstream_delay: Duration) {
        self.state.send_modify(|state| {
            state.partitioned = false;
            state.downstream_delay = downstream_delay;
        });
    }

    pub(crate) async fn shutdown(self) {
        self.state.send_modify(|state| {
            state.generation += 1;
            state.partitioned = true;
        });
        self.listener.abort();
        let _ = self.listener.await;
    }
}

async fn proxy(client: TcpStream, target: SocketAddr, state: watch::Receiver<GateState>) {
    let Ok(server) = TcpStream::connect(target).await else {
        return;
    };
    let generation = state.borrow().generation;
    let (client_read, client_write) = client.into_split();
    let (server_read, server_write) = server.into_split();
    tokio::select! {
        _ = copy_upstream(client_read, server_write, state.clone(), generation) => {}
        _ = copy_downstream(server_read, client_write, state, generation) => {}
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
