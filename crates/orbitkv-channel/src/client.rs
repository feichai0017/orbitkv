use std::io::{Read, Write};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rustix::net::sockopt::socket_peercred;
use rustix::process::{PidfdFlags, pidfd_open};
use thiserror::Error;

use crate::lifecycle::{LIFECYCLE_HEADER_BYTES, LifecycleCommand, LifecycleHeader};
use crate::{
    BootstrapClient, BootstrapError, CallOptions, Command, CommandCode, PublishRequest,
    QueryBundleRequest, QueryBundleResponse, QueryCodecError, RESPONSE_FLAG_REQUEST_CONSUMED,
    ReleaseRequest, RestoreCommand, RestoreRequest, RestoreResponse, RestoreState, StatusCode,
    TransportClient, TransportError,
};

const RESTORE_NOTIFICATION_POLL_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(50);

#[derive(Debug, Error)]
pub enum ChannelError {
    #[error(transparent)]
    Bootstrap(#[from] BootstrapError),
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Codec(#[from] QueryCodecError),
    #[error("cache request returned {0:?}")]
    Status(StatusCode),
    #[error("cache session is ambiguous after a failed call; reconnect required")]
    SessionRequiresReconnect,
    #[error("restore operation {operation_id} timed out")]
    RestoreTimeout { operation_id: u64 },
    #[error("restore operation {operation_id} failed: {message}")]
    RestoreFailed { operation_id: u64, message: String },
    #[error("cannot observe Cache Manager process death; refusing to start a publish")]
    PublishPeerUnobservable,
    #[error("Cache Manager rejected lifecycle operation ({code}): {message}")]
    Lifecycle { code: u16, message: String },
}

pub struct ChannelClient {
    bootstrap: BootstrapClient,
    client: TransportClient,
    options: CallOptions,
    call_lock: Mutex<()>,
    lifecycle_lock: Mutex<()>,
    poisoned: AtomicBool,
    publish_peer: Option<OwnedFd>,
}

impl ChannelClient {
    pub fn connect(
        bootstrap_socket: impl AsRef<Path>,
        options: CallOptions,
    ) -> Result<Self, ChannelError> {
        let bootstrap = BootstrapClient::connect(bootstrap_socket)?;
        let publish_peer = socket_peercred(bootstrap.stream())
            .ok()
            .and_then(|credentials| pidfd_open(credentials.pid, PidfdFlags::empty()).ok());
        bootstrap
            .stream()
            .set_read_timeout(Some(options.timeout))
            .map_err(BootstrapError::Io)?;
        bootstrap
            .stream()
            .set_write_timeout(Some(options.timeout))
            .map_err(BootstrapError::Io)?;
        let client = TransportClient::connect(&bootstrap.info().service_name)?;
        Ok(Self {
            bootstrap,
            client,
            options,
            call_lock: Mutex::new(()),
            lifecycle_lock: Mutex::new(()),
            poisoned: AtomicBool::new(false),
            publish_peer,
        })
    }

    pub fn service_name(&self) -> &str {
        &self.bootstrap.info_ref().service_name
    }

    /// Registration and liveness metadata use UDS, independently of the hot descriptor slot.
    pub fn lifecycle(&self, command: LifecycleCommand, payload: &[u8]) -> Result<(), ChannelError> {
        let _guard = self
            .lifecycle_lock
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        if self.poisoned.load(Ordering::Acquire) {
            return Err(ChannelError::SessionRequiresReconnect);
        }
        let exchange = || -> std::io::Result<(u16, Vec<u8>)> {
            let header = LifecycleHeader {
                code: command as u16,
                epoch: self.session_epoch(),
                payload_len: payload.len(),
            }
            .encode()?;
            let mut stream = self.bootstrap.stream();
            let timeout = match command {
                LifecycleCommand::Register | LifecycleCommand::Unregister => {
                    self.options.timeout.max(Duration::from_secs(120))
                }
                _ => self.options.timeout,
            };
            stream.set_read_timeout(Some(timeout))?;
            stream.write_all(&header)?;
            stream.write_all(payload)?;
            let mut header = [0; LIFECYCLE_HEADER_BYTES];
            stream.read_exact(&mut header)?;
            let header = LifecycleHeader::decode(header)?;
            if header.epoch != self.session_epoch() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stale lifecycle session",
                ));
            }
            let mut body = vec![0; header.payload_len];
            stream.read_exact(&mut body)?;
            Ok((header.code, body))
        };
        match exchange() {
            Ok((0, body)) if body.is_empty() => Ok(()),
            Ok((0, _)) => Err(ChannelError::Lifecycle {
                code: 0,
                message: "unexpected lifecycle response body".to_string(),
            }),
            Ok((code, message)) => Err(ChannelError::Lifecycle {
                code,
                message: String::from_utf8_lossy(&message).into_owned(),
            }),
            Err(error) => {
                self.close();
                // A partial frame must never be reused as a new request.
                Err(BootstrapError::Io(error).into())
            }
        }
    }

    /// End the process-channel session explicitly, without waiting for client destruction.
    pub fn close(&self) {
        self.poisoned.store(true, Ordering::Release);
        let _ = self.bootstrap.stream().shutdown(std::net::Shutdown::Both);
    }

    pub fn session_epoch(&self) -> u64 {
        self.bootstrap.info_ref().session_epoch
    }

    /// Borrowed notification descriptor. It remains valid only while this
    /// client is alive and must not be closed by the caller.
    pub fn notification_fd(&self) -> RawFd {
        self.bootstrap.notification_fd().as_raw_fd()
    }

    pub fn query_bundle(
        &self,
        request_id: u64,
        request: &QueryBundleRequest,
    ) -> Result<QueryBundleResponse, ChannelError> {
        let payload = request.encode()?;
        let payload = self.call_descriptor(CommandCode::QueryBundle, request_id, &payload)?;
        match QueryBundleResponse::decode(&payload) {
            Ok(response) => Ok(response),
            Err(error) => {
                self.close();
                Err(error.into())
            }
        }
    }

    pub fn release(&self, request_id: u64, lease: Vec<u8>) -> Result<(), ChannelError> {
        let payload = ReleaseRequest { lease }.encode()?;
        let _ = self.call_descriptor(CommandCode::Release, request_id, &payload)?;
        Ok(())
    }

    pub fn publish(&self, request_id: u64, request: &PublishRequest) -> Result<(), ChannelError> {
        let peer = self
            .publish_peer
            .as_ref()
            .ok_or(ChannelError::PublishPeerUnobservable)?;
        let payloads = publish_payloads(request, self.bootstrap.info_ref().slot_capacity)?;
        for payload in payloads {
            // One logical publish can use several descriptor generations.
            // Every chunk retains the source pages until its D2H completes.
            let _ =
                self.call_descriptor_with_peer(CommandCode::Publish, request_id, &payload, peer)?;
        }
        Ok(())
    }

    pub fn restore_submit(
        &self,
        request_id: u64,
        request: &RestoreRequest,
    ) -> Result<u64, ChannelError> {
        let payload = RestoreCommand::Submit(request.clone()).encode()?;
        let payload = self.call_descriptor(CommandCode::Restore, request_id, &payload)?;
        let response = RestoreResponse::decode(&payload)?;
        match response.state {
            RestoreState::Pending => Ok(response.operation_id),
            RestoreState::Succeeded => Ok(response.operation_id),
            RestoreState::Failed => Err(ChannelError::RestoreFailed {
                operation_id: response.operation_id,
                message: response.message,
            }),
        }
    }

    pub fn restore_poll(
        &self,
        request_id: u64,
        operation_id: u64,
    ) -> Result<RestoreResponse, ChannelError> {
        let payload = RestoreCommand::Poll { operation_id }.encode()?;
        let payload = self.call_descriptor(CommandCode::Restore, request_id, &payload)?;
        Ok(RestoreResponse::decode(&payload)?)
    }

    pub fn restore_wait(
        &self,
        request_id: u64,
        operation_id: u64,
        timeout: std::time::Duration,
    ) -> Result<(), ChannelError> {
        let deadline = std::time::Instant::now() + timeout;
        let mut poll_request_id = request_id;
        loop {
            let response = self.restore_poll(poll_request_id, operation_id)?;
            match response.state {
                RestoreState::Succeeded => return Ok(()),
                RestoreState::Failed => {
                    return Err(ChannelError::RestoreFailed {
                        operation_id,
                        message: response.message,
                    });
                }
                RestoreState::Pending => {}
            }
            poll_request_id = poll_request_id
                .checked_add(1)
                .ok_or(ChannelError::SessionRequiresReconnect)?;
            let now = std::time::Instant::now();
            if now >= deadline {
                return Err(ChannelError::RestoreTimeout { operation_id });
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(RESTORE_NOTIFICATION_POLL_INTERVAL);
            let _ = self.bootstrap.wait_for_notification(wait)?;
        }
    }

    fn call_descriptor(
        &self,
        code: CommandCode,
        request_id: u64,
        payload: &[u8],
    ) -> Result<Vec<u8>, ChannelError> {
        self.call_descriptor_inner(code, request_id, payload, None)
    }

    fn call_descriptor_with_peer(
        &self,
        code: CommandCode,
        request_id: u64,
        payload: &[u8],
        peer: &OwnedFd,
    ) -> Result<Vec<u8>, ChannelError> {
        self.call_descriptor_inner(code, request_id, payload, Some(peer))
    }

    fn call_descriptor_inner(
        &self,
        code: CommandCode,
        request_id: u64,
        payload: &[u8],
        peer: Option<&OwnedFd>,
    ) -> Result<Vec<u8>, ChannelError> {
        let _call = self
            .call_lock
            .lock()
            .map_err(|_| BootstrapError::Arena(crate::ArenaError::Poisoned))?;
        if self.poisoned.load(Ordering::Acquire) {
            return Err(ChannelError::SessionRequiresReconnect);
        }
        let descriptor = self.bootstrap.write_request(payload)?;
        let info = self.bootstrap.info();
        let command = Command {
            code,
            request_id,
            session_epoch: info.session_epoch,
            descriptor,
            arg0: info.client_token,
            arg1: 0,
        };
        let call = match peer {
            Some(peer) => self
                .client
                .call_until_peer_exit(command, self.options, peer),
            None => self.client.call(command, self.options),
        };
        let response = match call {
            Ok(response) => response,
            Err(error) => {
                self.close();
                if let Some(peer) = peer
                    && !matches!(
                        error,
                        TransportError::PeerExited { .. } | TransportError::Send(_)
                    )
                {
                    TransportClient::wait_for_peer_exit(peer);
                }
                return Err(error.into());
            }
        };
        if response.status != StatusCode::Ok {
            if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED != 0 {
                self.bootstrap
                    .complete_request(descriptor)
                    .inspect_err(|_| self.close())?;
            } else {
                self.close();
            }
            return Err(ChannelError::Status(response.status));
        }
        if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED == 0 {
            self.close();
            return Err(ChannelError::SessionRequiresReconnect);
        }
        let payload = self
            .bootstrap
            .read_response(descriptor, response.descriptor)
            .inspect_err(|_| self.close())?;
        Ok(payload)
    }
}

fn publish_payloads(
    request: &PublishRequest,
    capacity: usize,
) -> Result<Vec<Vec<u8>>, ChannelError> {
    let payload = request.encode()?;
    if payload.len() <= capacity {
        return Ok(vec![payload]);
    }
    let too_large = |len| {
        ChannelError::Bootstrap(BootstrapError::Arena(crate::ArenaError::PayloadTooLarge {
            len,
            capacity,
        }))
    };
    let block_count = request
        .layers
        .iter()
        .map(|layer| layer.block_ids.len())
        .max()
        .unwrap_or(0);
    if block_count == 0 {
        return Err(too_large(payload.len()));
    }

    let mut payloads = Vec::new();
    let mut start = 0;
    while start < block_count {
        let mut low = start + 1;
        let mut high = block_count;
        let mut selected = None;
        while low <= high {
            let end = low + (high - low) / 2;
            // Slice the same block range across layers so each completed
            // chunk can seal full pages, including page-first registrations.
            let layers = request
                .layers
                .iter()
                .filter_map(|layer| {
                    let end = end.min(layer.block_ids.len());
                    (start < end).then(|| crate::PublishLayer {
                        layer_name: layer.layer_name.clone(),
                        block_ids: layer.block_ids[start..end].to_vec(),
                        block_hashes: layer.block_hashes[start..end].to_vec(),
                    })
                })
                .collect();
            let chunk = PublishRequest {
                instance_id: request.instance_id.clone(),
                tp_rank: request.tp_rank,
                pp_rank: request.pp_rank,
                device_id: request.device_id,
                layers,
            }
            .encode()?;
            if chunk.len() <= capacity {
                selected = Some((end, chunk));
                low = end + 1;
            } else {
                if end == start + 1 {
                    return Err(too_large(chunk.len()));
                }
                high = end - 1;
            }
        }
        let (end, chunk) = selected.ok_or_else(|| too_large(payload.len()))?;
        payloads.push(chunk);
        start = end;
    }
    Ok(payloads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PublishLayer;

    #[test]
    fn publish_chunks_preserve_all_layers_and_pages_within_the_slot_limit() {
        let request = PublishRequest {
            instance_id: "qwen3-8b".to_string(),
            tp_rank: 2,
            pp_rank: 1,
            device_id: 3,
            layers: (0..36)
                .map(|layer| PublishLayer {
                    layer_name: format!("model.layers.{layer}.attn"),
                    block_ids: (0..192).collect(),
                    block_hashes: (0..192).map(|page| vec![page as u8; 32]).collect(),
                })
                .collect(),
        };
        let capacity = crate::arena::DEFAULT_SLOT_CAPACITY;
        let payloads = publish_payloads(&request, capacity).unwrap();
        assert!(payloads.len() > 1);
        let mut reconstructed = request.clone();
        for layer in &mut reconstructed.layers {
            layer.block_ids.clear();
            layer.block_hashes.clear();
        }
        for payload in payloads {
            assert!(payload.len() <= capacity);
            let chunk = PublishRequest::decode(&payload).unwrap();
            assert_eq!(chunk.instance_id, request.instance_id);
            assert_eq!((chunk.tp_rank, chunk.pp_rank, chunk.device_id), (2, 1, 3));
            assert_eq!(chunk.layers.len(), request.layers.len());
            for (layer, original) in chunk.layers.into_iter().zip(&mut reconstructed.layers) {
                assert_eq!(layer.layer_name, original.layer_name);
                original.block_ids.extend(layer.block_ids);
                original.block_hashes.extend(layer.block_hashes);
            }
        }
        assert_eq!(reconstructed, request);
        assert!(publish_payloads(&request, 32).is_err());
        assert_eq!(
            publish_payloads(&request, usize::MAX).unwrap(),
            vec![request.encode().unwrap()]
        );
    }

    #[test]
    fn publish_chunks_preserve_ragged_cache_groups() {
        let request = PublishRequest {
            instance_id: "hybrid".to_string(),
            tp_rank: 0,
            pp_rank: 0,
            device_id: 0,
            layers: (0..6)
                .map(|index| PublishLayer {
                    layer_name: format!("layer.{index}"),
                    block_ids: (0..index * 4).collect(),
                    block_hashes: (0..index * 4)
                        .map(|page| vec![page as u8; 8 + page as usize])
                        .collect(),
                })
                .collect(),
        };
        let mut reconstructed = request.clone();
        for layer in &mut reconstructed.layers {
            layer.block_ids.clear();
            layer.block_hashes.clear();
        }
        for payload in publish_payloads(&request, 512).unwrap() {
            assert!(payload.len() <= 512);
            for layer in PublishRequest::decode(&payload).unwrap().layers {
                let original = reconstructed
                    .layers
                    .iter_mut()
                    .find(|candidate| candidate.layer_name == layer.layer_name)
                    .unwrap();
                original.block_ids.extend(layer.block_ids);
                original.block_hashes.extend(layer.block_hashes);
            }
        }
        assert_eq!(reconstructed, request);
    }
}
