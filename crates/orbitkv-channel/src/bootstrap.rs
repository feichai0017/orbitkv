use std::fs;
use std::io::{ErrorKind, IoSlice, IoSliceMut, Read, Write};
use std::mem::MaybeUninit;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rustix::event::{EventfdFlags, eventfd};
use rustix::net::sockopt::socket_peercred;
use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
};
use rustix::process::geteuid;
use thiserror::Error;

use crate::{ArenaError, DescriptorArena, DescriptorRef};

const BOOTSTRAP_MAGIC: u32 = 0x4f52_4242; // ORBB
// Version 2 requires lifecycle framing on the persistent bootstrap socket.
const BOOTSTRAP_VERSION: u16 = 2;
const BOOTSTRAP_BYTES: usize = 256;
const BOOTSTRAP_FD_COUNT: usize = 2;
const SERVICE_NAME_OFFSET: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerCredentials {
    pub pid: u32,
    pub uid: u32,
    pub gid: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapInfo {
    pub service_name: String,
    pub session_epoch: u64,
    pub arena_bytes: usize,
    pub slot_capacity: usize,
    pub slot_count: usize,
    pub slot_index: usize,
    pub initial_generation: u64,
    pub client_token: u64,
}

#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("bootstrap socket operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Arena(#[from] ArenaError),
    #[error("peer uid {actual} is not permitted; expected {expected}")]
    PeerUid { expected: u32, actual: u32 },
    #[error("invalid bootstrap magic: {0:#x}")]
    InvalidMagic(u32),
    #[error("unsupported bootstrap version: {0}")]
    UnsupportedVersion(u16),
    #[error("bootstrap payload length mismatch: expected {expected}, got {actual}")]
    PayloadLength { expected: usize, actual: usize },
    #[error("bootstrap expected {expected} file descriptors, got {actual}")]
    FileDescriptorCount { expected: usize, actual: usize },
    #[error("bootstrap ancillary buffer was too small")]
    AncillaryTruncated,
    #[error("bootstrap field {field} exceeds its local representation")]
    FieldOverflow { field: &'static str },
    #[error("iceoryx2 service name is too long for bootstrap: {0} bytes")]
    ServiceNameTooLong(usize),
    #[error("bootstrap iceoryx2 service name is not valid UTF-8")]
    InvalidServiceNameUtf8,
    #[error("all {0} descriptor slots are in use")]
    NoFreeSlots(usize),
    #[error("descriptor generation exhausted for slot {0}")]
    GenerationExhausted(usize),
    #[error("bootstrap client-token space is exhausted")]
    ClientTokenExhausted,
    #[error("descriptor belongs to slot {actual}, but client owns slot {expected}")]
    SlotMismatch { expected: usize, actual: usize },
    #[error("client token does not match the bootstrap session")]
    ClientTokenMismatch,
    #[error("unexpected request generation: expected {expected}, got {actual}")]
    UnexpectedGeneration { expected: u64, actual: u64 },
}

pub struct BootstrapServer {
    listener: UnixListener,
    socket_path: PathBuf,
    socket_identity: SocketIdentity,
    arena: DescriptorArena,
    service_name: String,
    session_epoch: u64,
    slots: Arc<Mutex<Vec<bool>>>,
    next_client_token: AtomicU64,
}

impl BootstrapServer {
    pub fn bind(
        socket_path: impl AsRef<Path>,
        service_name: impl Into<String>,
        session_epoch: u64,
        arena_bytes: usize,
        slot_capacity: usize,
    ) -> Result<Self, BootstrapError> {
        let socket_path = socket_path.as_ref().to_path_buf();
        let arena = DescriptorArena::create(session_epoch, arena_bytes, slot_capacity)?;
        let listener = bind_socket(&socket_path)?;
        if let Err(error) = fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600)) {
            drop(listener);
            let _ = fs::remove_file(&socket_path);
            return Err(error.into());
        }
        let socket_identity = SocketIdentity::from_metadata(&fs::symlink_metadata(&socket_path)?);
        let slots = Arc::new(Mutex::new(vec![false; arena.slot_count()]));
        Ok(Self {
            listener,
            socket_path,
            socket_identity,
            arena,
            service_name: service_name.into(),
            session_epoch,
            slots,
            next_client_token: AtomicU64::new(session_epoch.rotate_left(17) | 1),
        })
    }

    pub fn accept(&self) -> Result<BootstrapSession, BootstrapError> {
        let (stream, _) = self.listener.accept()?;
        let credentials = peer_credentials(&stream)?;
        let expected_uid = geteuid().as_raw();
        if credentials.uid != expected_uid {
            return Err(BootstrapError::PeerUid {
                expected: expected_uid,
                actual: credentials.uid,
            });
        }
        let (slot_index, initial_generation) = self.allocate_slot()?;
        let client_token = match self.next_client_token.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |token| token.checked_add(2),
        ) {
            Ok(token) => token,
            Err(_) => {
                self.release_slot(slot_index);
                return Err(BootstrapError::ClientTokenExhausted);
            }
        };
        let notification = match eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK) {
            Ok(notification) => notification,
            Err(error) => {
                self.release_slot(slot_index);
                return Err(std::io::Error::from(error).into());
            }
        };
        let info = BootstrapInfo {
            service_name: self.service_name.clone(),
            session_epoch: self.session_epoch,
            arena_bytes: self.arena.len(),
            slot_capacity: self.arena.slot_capacity(),
            slot_count: self.arena.slot_count(),
            slot_index,
            initial_generation,
            client_token,
        };
        if let Err(error) = send_bootstrap(&stream, &info, self.arena.file(), &notification) {
            self.release_slot(slot_index);
            return Err(error);
        }
        if let Err(error) = stream.set_nonblocking(true) {
            self.release_slot(slot_index);
            return Err(error.into());
        }
        Ok(BootstrapSession {
            stream,
            credentials,
            notification,
            slot_index,
            client_token,
            next_request_generation: initial_generation,
            slots: Arc::clone(&self.slots),
        })
    }

    pub fn arena(&self) -> &DescriptorArena {
        &self.arena
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<(), BootstrapError> {
        self.listener.set_nonblocking(nonblocking)?;
        Ok(())
    }

    pub fn try_accept(&self) -> Result<Option<BootstrapSession>, BootstrapError> {
        match self.accept() {
            Ok(session) => Ok(Some(session)),
            Err(BootstrapError::Io(error)) if error.kind() == std::io::ErrorKind::WouldBlock => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    pub fn descriptor_slot(&self, offset: u64) -> Result<usize, ArenaError> {
        self.arena.descriptor_slot(offset)
    }

    fn allocate_slot(&self) -> Result<(usize, u64), BootstrapError> {
        let mut slots = self.slots.lock().map_err(|_| ArenaError::Poisoned)?;
        let slot = slots
            .iter()
            .position(|active| !active)
            .ok_or(BootstrapError::NoFreeSlots(slots.len()))?;
        let current = self.arena.slot_generation(slot)?;
        let initial_generation = if current == 0 {
            1
        } else if current.is_multiple_of(2) {
            current
                .checked_add(1)
                .ok_or(BootstrapError::GenerationExhausted(slot))?
        } else {
            current
                .checked_add(2)
                .ok_or(BootstrapError::GenerationExhausted(slot))?
        };
        self.arena.reset_slot(slot, initial_generation)?;
        slots[slot] = true;
        Ok((slot, initial_generation))
    }

    fn release_slot(&self, slot: usize) {
        if let Ok(mut slots) = self.slots.lock()
            && let Some(active) = slots.get_mut(slot)
        {
            *active = false;
        }
    }
}

impl Drop for BootstrapServer {
    fn drop(&mut self) {
        if let Ok(metadata) = fs::symlink_metadata(&self.socket_path)
            && metadata.file_type().is_socket()
            && SocketIdentity::from_metadata(&metadata) == self.socket_identity
        {
            let _ = fs::remove_file(&self.socket_path);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SocketIdentity {
    device: u64,
    inode: u64,
}

impl SocketIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }
}

pub struct BootstrapSession {
    stream: UnixStream,
    credentials: PeerCredentials,
    notification: std::os::fd::OwnedFd,
    slot_index: usize,
    client_token: u64,
    next_request_generation: u64,
    slots: Arc<Mutex<Vec<bool>>>,
}

impl BootstrapSession {
    pub fn credentials(&self) -> PeerCredentials {
        self.credentials
    }

    pub fn stream(&self) -> &UnixStream {
        &self.stream
    }

    pub fn notification_fd(&self) -> &std::os::fd::OwnedFd {
        &self.notification
    }

    pub fn notify(&self) -> Result<(), BootstrapError> {
        let written = match rustix::io::write(&self.notification, &1u64.to_ne_bytes()) {
            Ok(written) => written,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) => return Err(std::io::Error::from(error).into()),
        };
        if written != std::mem::size_of::<u64>() {
            return Err(BootstrapError::PayloadLength {
                expected: std::mem::size_of::<u64>(),
                actual: written,
            });
        }
        Ok(())
    }

    pub fn slot_index(&self) -> usize {
        self.slot_index
    }

    pub fn client_token(&self) -> u64 {
        self.client_token
    }

    pub fn is_alive(&self) -> Result<bool, BootstrapError> {
        let mut byte = [0u8; 1];
        match rustix::net::recv(
            &self.stream,
            &mut byte,
            RecvFlags::PEEK | RecvFlags::DONTWAIT,
        ) {
            Ok((_, 0)) => Ok(false),
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(true),
            Err(error) => Err(std::io::Error::from(error).into()),
        }
    }

    pub fn validate_request(
        &mut self,
        descriptor: DescriptorRef,
        client_token: u64,
        descriptor_slot: usize,
    ) -> Result<(), BootstrapError> {
        if client_token != self.client_token {
            return Err(BootstrapError::ClientTokenMismatch);
        }
        if descriptor_slot != self.slot_index {
            return Err(BootstrapError::SlotMismatch {
                expected: self.slot_index,
                actual: descriptor_slot,
            });
        }
        if descriptor.generation != self.next_request_generation {
            return Err(BootstrapError::UnexpectedGeneration {
                expected: self.next_request_generation,
                actual: descriptor.generation,
            });
        }
        Ok(())
    }

    pub fn complete_request(&mut self) -> Result<(), BootstrapError> {
        self.next_request_generation = self
            .next_request_generation
            .checked_add(2)
            .ok_or(BootstrapError::GenerationExhausted(self.slot_index))?;
        Ok(())
    }
}

impl Drop for BootstrapSession {
    fn drop(&mut self) {
        if let Ok(mut slots) = self.slots.lock()
            && let Some(active) = slots.get_mut(self.slot_index)
        {
            *active = false;
        }
    }
}

pub struct BootstrapClient {
    stream: UnixStream,
    arena: DescriptorArena,
    notification: std::os::fd::OwnedFd,
    info: BootstrapInfo,
    next_generation: Mutex<u64>,
}

impl BootstrapClient {
    pub fn connect(socket_path: impl AsRef<Path>) -> Result<Self, BootstrapError> {
        let stream = UnixStream::connect(socket_path)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        let (info, mut fds) = receive_bootstrap(&stream)?;
        let notification = fds.pop().ok_or(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: 0,
        })?;
        let arena_fd = fds.pop().ok_or(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: 1,
        })?;
        let arena = DescriptorArena::open(arena_fd, info.session_epoch, info.arena_bytes)?;
        Ok(Self {
            stream,
            arena,
            notification,
            next_generation: Mutex::new(info.initial_generation),
            info,
        })
    }

    pub fn info(&self) -> BootstrapInfo {
        self.info.clone()
    }

    pub fn info_ref(&self) -> &BootstrapInfo {
        &self.info
    }

    pub fn arena(&self) -> &DescriptorArena {
        &self.arena
    }

    pub fn stream(&self) -> &UnixStream {
        &self.stream
    }

    pub fn notification_fd(&self) -> &std::os::fd::OwnedFd {
        &self.notification
    }

    pub fn wait_for_notification(&self, timeout: Duration) -> Result<bool, BootstrapError> {
        let timeout = rustix::event::Timespec {
            tv_sec: i64::try_from(timeout.as_secs()).map_err(|_| {
                BootstrapError::FieldOverflow {
                    field: "notification_timeout",
                }
            })?,
            tv_nsec: timeout.subsec_nanos().into(),
        };
        let mut fds = [rustix::event::PollFd::new(
            &self.notification,
            rustix::event::PollFlags::IN,
        )];
        if rustix::event::poll(&mut fds, Some(&timeout)).map_err(std::io::Error::from)? == 0 {
            return Ok(false);
        }
        let mut value = [0u8; 8];
        let read = match rustix::io::read(&self.notification, &mut value) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(false),
            Err(error) => return Err(std::io::Error::from(error).into()),
        };
        Ok(read == value.len())
    }

    pub fn write_request(&self, payload: &[u8]) -> Result<crate::DescriptorRef, BootstrapError> {
        let generation = self
            .next_generation
            .lock()
            .map_err(|_| ArenaError::Poisoned)?;
        let descriptor = self
            .arena
            .write_slot(self.info.slot_index, *generation, payload)?;
        Ok(descriptor)
    }

    pub fn read_response(
        &self,
        request: DescriptorRef,
        response: DescriptorRef,
    ) -> Result<Vec<u8>, BootstrapError> {
        let expected_generation = request
            .generation
            .checked_add(1)
            .ok_or(BootstrapError::GenerationExhausted(self.info.slot_index))?;
        if response.offset != request.offset {
            return Err(BootstrapError::SlotMismatch {
                expected: self.info.slot_index,
                actual: self.arena.descriptor_slot(response.offset)?,
            });
        }
        if response.generation != expected_generation {
            return Err(BootstrapError::UnexpectedGeneration {
                expected: expected_generation,
                actual: response.generation,
            });
        }
        let payload = self.arena.read(response)?;
        self.complete_request(request)?;
        Ok(payload)
    }

    pub fn complete_request(&self, request: DescriptorRef) -> Result<(), BootstrapError> {
        let mut generation = self
            .next_generation
            .lock()
            .map_err(|_| ArenaError::Poisoned)?;
        if *generation != request.generation {
            return Err(BootstrapError::UnexpectedGeneration {
                expected: *generation,
                actual: request.generation,
            });
        }
        *generation = generation
            .checked_add(2)
            .ok_or(BootstrapError::GenerationExhausted(self.info.slot_index))?;
        Ok(())
    }
}

fn peer_credentials(stream: &UnixStream) -> Result<PeerCredentials, BootstrapError> {
    let credentials = socket_peercred(stream).map_err(std::io::Error::from)?;
    Ok(PeerCredentials {
        pid: u32::try_from(credentials.pid.as_raw_nonzero().get())
            .map_err(|_| BootstrapError::FieldOverflow { field: "peer_pid" })?,
        uid: credentials.uid.as_raw(),
        gid: credentials.gid.as_raw(),
    })
}

fn bind_socket(path: &Path) -> Result<UnixListener, std::io::Error> {
    match UnixListener::bind(path) {
        Ok(listener) => Ok(listener),
        Err(error) if error.kind() == ErrorKind::AddrInUse => match UnixStream::connect(path) {
            Ok(_) => Err(error),
            Err(connect_error)
                if matches!(
                    connect_error.kind(),
                    ErrorKind::ConnectionRefused | ErrorKind::NotFound
                ) =>
            {
                let metadata = fs::symlink_metadata(path)?;
                if !metadata.file_type().is_socket() {
                    return Err(error);
                }
                fs::remove_file(path)?;
                UnixListener::bind(path)
            }
            Err(_) => Err(error),
        },
        Err(error) => Err(error),
    }
}

fn send_bootstrap(
    stream: &UnixStream,
    info: &BootstrapInfo,
    arena_fd: &impl std::os::fd::AsFd,
    notification_fd: &impl std::os::fd::AsFd,
) -> Result<(), BootstrapError> {
    let payload = encode_info(info)?;
    let borrowed = [arena_fd.as_fd(), notification_fd.as_fd()];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(BOOTSTRAP_FD_COUNT))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !control.push(SendAncillaryMessage::ScmRights(&borrowed)) {
        return Err(BootstrapError::AncillaryTruncated);
    }
    let sent = sendmsg(
        stream,
        &[IoSlice::new(&payload)],
        &mut control,
        SendFlags::empty(),
    )
    .map_err(std::io::Error::from)?;
    if sent == 0 {
        return Err(std::io::Error::from(std::io::ErrorKind::WriteZero).into());
    }
    if sent < payload.len() {
        let mut stream = stream;
        stream.write_all(&payload[sent..])?;
    }
    Ok(())
}

fn receive_bootstrap(
    stream: &UnixStream,
) -> Result<(BootstrapInfo, Vec<std::os::fd::OwnedFd>), BootstrapError> {
    let mut payload = [0u8; BOOTSTRAP_BYTES];
    let mut iov = [IoSliceMut::new(&mut payload)];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(BOOTSTRAP_FD_COUNT))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let message = recvmsg(stream, &mut iov, &mut control, RecvFlags::CMSG_CLOEXEC)
        .map_err(std::io::Error::from)?;
    if message.bytes == 0 {
        return Err(BootstrapError::PayloadLength {
            expected: BOOTSTRAP_BYTES,
            actual: 0,
        });
    }
    if message.flags.contains(rustix::net::ReturnFlags::CTRUNC) {
        return Err(BootstrapError::AncillaryTruncated);
    }
    let mut fds = Vec::new();
    for message in control.drain() {
        if let RecvAncillaryMessage::ScmRights(rights) = message {
            fds.extend(rights);
        }
    }
    if fds.len() != BOOTSTRAP_FD_COUNT {
        return Err(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: fds.len(),
        });
    }
    if message.bytes < BOOTSTRAP_BYTES {
        let mut stream = stream;
        stream.read_exact(&mut payload[message.bytes..])?;
    }
    Ok((decode_info(payload)?, fds))
}

fn encode_info(info: &BootstrapInfo) -> Result<[u8; BOOTSTRAP_BYTES], BootstrapError> {
    let service_name = info.service_name.as_bytes();
    if service_name.len() > BOOTSTRAP_BYTES - SERVICE_NAME_OFFSET {
        return Err(BootstrapError::ServiceNameTooLong(service_name.len()));
    }
    let mut bytes = [0u8; BOOTSTRAP_BYTES];
    bytes[0..4].copy_from_slice(&BOOTSTRAP_MAGIC.to_le_bytes());
    bytes[4..6].copy_from_slice(&BOOTSTRAP_VERSION.to_le_bytes());
    bytes[8..16].copy_from_slice(&info.session_epoch.to_le_bytes());
    bytes[16..24].copy_from_slice(
        &u64::try_from(info.arena_bytes)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "arena_bytes",
            })?
            .to_le_bytes(),
    );
    bytes[24..28].copy_from_slice(
        &u32::try_from(info.slot_capacity)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "slot_capacity",
            })?
            .to_le_bytes(),
    );
    bytes[28..32].copy_from_slice(
        &u32::try_from(info.slot_count)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "slot_count",
            })?
            .to_le_bytes(),
    );
    bytes[32..36].copy_from_slice(
        &u32::try_from(info.slot_index)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "slot_index",
            })?
            .to_le_bytes(),
    );
    bytes[36..38].copy_from_slice(
        &u16::try_from(service_name.len())
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "service_name_len",
            })?
            .to_le_bytes(),
    );
    bytes[40..48].copy_from_slice(&info.initial_generation.to_le_bytes());
    bytes[48..56].copy_from_slice(&info.client_token.to_le_bytes());
    bytes[SERVICE_NAME_OFFSET..SERVICE_NAME_OFFSET + service_name.len()]
        .copy_from_slice(service_name);
    Ok(bytes)
}

fn decode_info(bytes: [u8; BOOTSTRAP_BYTES]) -> Result<BootstrapInfo, BootstrapError> {
    let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("fixed slice"));
    if magic != BOOTSTRAP_MAGIC {
        return Err(BootstrapError::InvalidMagic(magic));
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().expect("fixed slice"));
    if version != BOOTSTRAP_VERSION {
        return Err(BootstrapError::UnsupportedVersion(version));
    }
    let service_name_len =
        u16::from_le_bytes(bytes[36..38].try_into().expect("fixed slice")) as usize;
    if service_name_len > BOOTSTRAP_BYTES - SERVICE_NAME_OFFSET {
        return Err(BootstrapError::ServiceNameTooLong(service_name_len));
    }
    let service_name = String::from_utf8(
        bytes[SERVICE_NAME_OFFSET..SERVICE_NAME_OFFSET + service_name_len].to_vec(),
    )
    .map_err(|_| BootstrapError::InvalidServiceNameUtf8)?;
    Ok(BootstrapInfo {
        service_name,
        session_epoch: u64::from_le_bytes(bytes[8..16].try_into().expect("fixed slice")),
        arena_bytes: usize::try_from(u64::from_le_bytes(
            bytes[16..24].try_into().expect("fixed slice"),
        ))
        .map_err(|_| BootstrapError::FieldOverflow {
            field: "arena_bytes",
        })?,
        slot_capacity: u32::from_le_bytes(bytes[24..28].try_into().expect("fixed slice")) as usize,
        slot_count: u32::from_le_bytes(bytes[28..32].try_into().expect("fixed slice")) as usize,
        slot_index: u32::from_le_bytes(bytes[32..36].try_into().expect("fixed slice")) as usize,
        initial_generation: u64::from_le_bytes(bytes[40..48].try_into().expect("fixed slice")),
        client_token: u64::from_le_bytes(bytes[48..56].try_into().expect("fixed slice")),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    fn connect_pair(server: &BootstrapServer, path: &Path) -> (BootstrapClient, BootstrapSession) {
        thread::scope(|scope| {
            let client = scope.spawn(|| BootstrapClient::connect(path).unwrap());
            let session = server.accept().unwrap();
            (client.join().unwrap(), session)
        })
    }

    #[test]
    fn bootstrap_passes_arena_and_notification_descriptors() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bootstrap.sock");
        let server = Arc::new(
            BootstrapServer::bind(&path, "orbitkv/test/bootstrap", 29, 16 * 1024, 1024).unwrap(),
        );
        let server_thread = {
            let server = Arc::clone(&server);
            thread::spawn(move || {
                let session = server.accept().unwrap();
                assert_eq!(session.credentials().uid, geteuid().as_raw());
                assert!(session.credentials().pid > 0);
                session.notify().unwrap();
                session
            })
        };

        let client = BootstrapClient::connect(&path).unwrap();
        assert_eq!(client.info().session_epoch, 29);
        assert_eq!(client.info().service_name, "orbitkv/test/bootstrap");
        assert_ne!(client.info().client_token, 0);
        assert!(
            client
                .wait_for_notification(Duration::from_secs(1))
                .unwrap()
        );
        let descriptor = client.write_request(b"query").unwrap();
        assert_eq!(
            server.descriptor_slot(descriptor.offset).unwrap(),
            client.info().slot_index
        );
        assert_eq!(server.arena().read(descriptor).unwrap(), b"query");
        let response = server.arena().write_response(descriptor, b"ready").unwrap();
        assert_eq!(
            client.read_response(descriptor, response).unwrap(),
            b"ready"
        );
        assert_eq!(response.generation, descriptor.generation + 1);

        drop(client);
        server_thread.join().unwrap();
    }

    #[test]
    fn reused_slot_gets_a_new_generation_and_old_request_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bootstrap.sock");
        let server =
            BootstrapServer::bind(&path, "orbitkv/test/reuse", 31, 16 * 1024, 4096).unwrap();

        let (first_client, mut first_session) = connect_pair(&server, &path);
        let first = first_client.write_request(b"first").unwrap();
        let first_slot = server.descriptor_slot(first.offset).unwrap();
        first_session
            .validate_request(first, first_client.info().client_token, first_slot)
            .unwrap();
        first_session.complete_request().unwrap();
        drop(first_session);
        drop(first_client);

        let (second_client, mut second_session) = connect_pair(&server, &path);
        assert_eq!(second_client.info().slot_index, first_slot);
        assert!(second_client.info().initial_generation > first.generation);
        assert!(!second_client.info().initial_generation.is_multiple_of(2));
        assert!(matches!(
            second_session.validate_request(first, second_client.info().client_token, first_slot),
            Err(BootstrapError::UnexpectedGeneration { .. })
        ));
    }

    #[test]
    fn failed_bind_does_not_remove_an_existing_socket() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bootstrap.sock");
        let first =
            BootstrapServer::bind(&path, "orbitkv/test/first", 41, 16 * 1024, 1024).unwrap();
        assert!(BootstrapServer::bind(&path, "orbitkv/test/second", 43, 16 * 1024, 1024).is_err());
        assert!(path.exists());
        drop(first);
        assert!(!path.exists());
    }

    #[test]
    fn bind_reclaims_a_stale_socket_after_owner_exit() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bootstrap.sock");
        drop(UnixListener::bind(&path).unwrap());
        assert!(path.exists());

        let server =
            BootstrapServer::bind(&path, "orbitkv/test/restart", 53, 16 * 1024, 1024).unwrap();
        assert!(path.exists());
        drop(server);
        assert!(!path.exists());
    }

    #[test]
    fn bind_never_removes_a_non_socket_path() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bootstrap.sock");
        fs::write(&path, b"keep").unwrap();

        assert!(
            BootstrapServer::bind(&path, "orbitkv/test/regular-file", 59, 16 * 1024, 1024).is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"keep");
    }

    #[test]
    fn session_rejects_wrong_identity_and_replayed_generation() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("bootstrap.sock");
        let server =
            BootstrapServer::bind(&path, "orbitkv/test/identity", 47, 16 * 1024, 1024).unwrap();
        let (client, mut session) = connect_pair(&server, &path);
        let descriptor = client.write_request(b"query").unwrap();
        let slot = server.descriptor_slot(descriptor.offset).unwrap();

        assert!(matches!(
            session.validate_request(descriptor, client.info().client_token + 1, slot),
            Err(BootstrapError::ClientTokenMismatch)
        ));
        assert!(matches!(
            session.validate_request(descriptor, client.info().client_token, slot + 1),
            Err(BootstrapError::SlotMismatch { .. })
        ));
        session
            .validate_request(descriptor, client.info().client_token, slot)
            .unwrap();
        session.complete_request().unwrap();
        assert!(matches!(
            session.validate_request(descriptor, client.info().client_token, slot),
            Err(BootstrapError::UnexpectedGeneration { .. })
        ));
    }
}
