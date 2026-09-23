use thiserror::Error;

const QUERY_REQUEST_MAGIC: u32 = 0x4f52_5151; // ORQQ
const QUERY_RESPONSE_MAGIC: u32 = 0x4f52_5152; // ORQR
const QUERY_POLL_MAGIC: u32 = 0x4f52_5150; // ORQP
const CANCEL_QUERY_MAGIC: u32 = 0x4f52_5143; // ORQC
const RELEASE_REQUEST_MAGIC: u32 = 0x4f52_4c51; // ORLQ
const PUBLISH_REQUEST_MAGIC: u32 = 0x4f52_5051; // ORPQ
const RESTORE_REQUEST_MAGIC: u32 = 0x4f52_5251; // ORRQ
const RESTORE_POLL_MAGIC: u32 = 0x4f52_5250; // ORRP
const RESTORE_RESPONSE_MAGIC: u32 = 0x4f52_5252; // ORRR
const QUERY_VERSION: u16 = 5;
const REQUEST_HEADER_BYTES: usize = 40;
const RESPONSE_HEADER_BYTES: usize = 24;
const RELEASE_HEADER_BYTES: usize = 12;
const PUBLISH_HEADER_BYTES: usize = 28;
const RESTORE_HEADER_BYTES: usize = 28;
const RESTORE_RESPONSE_BYTES: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CancelQueryRequest {
    pub ticket: QueryTicket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryTicket {
    pub operation_id: u64,
    pub revision: u64,
}

impl QueryTicket {
    fn encode_into(self, bytes: &mut Vec<u8>) -> Result<(), QueryCodecError> {
        if self.operation_id == 0 || self.revision == 0 {
            return Err(QueryCodecError::InvalidQueryTicket);
        }
        push_u64(bytes, self.operation_id);
        push_u64(bytes, self.revision);
        Ok(())
    }

    fn decode_from(decoder: &mut Decoder<'_>) -> Result<Self, QueryCodecError> {
        let ticket = Self {
            operation_id: decoder.u64()?,
            revision: decoder.u64()?,
        };
        if ticket.operation_id == 0 || ticket.revision == 0 {
            return Err(QueryCodecError::InvalidQueryTicket);
        }
        Ok(ticket)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryCommand {
    Submit(QueryBundleRequest),
    Poll(QueryTicket),
    Claim {
        ticket: QueryTicket,
        count_lookup: bool,
    },
}

impl QueryCommand {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        match self {
            Self::Submit(request) => request.encode(),
            Self::Poll(ticket) | Self::Claim { ticket, .. } => {
                let mut bytes = Vec::with_capacity(24);
                push_u32(&mut bytes, QUERY_POLL_MAGIC);
                push_u16(&mut bytes, QUERY_VERSION);
                let flags = match self {
                    Self::Claim { count_lookup, .. } => 1 | (u16::from(*count_lookup) << 1),
                    _ => 0,
                };
                push_u16(&mut bytes, flags);
                ticket.encode_into(&mut bytes)?;
                Ok(bytes)
            }
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        let magic = decoder.u32()?;
        if magic == QUERY_REQUEST_MAGIC {
            return QueryBundleRequest::decode(bytes).map(Self::Submit);
        }
        if magic != QUERY_POLL_MAGIC {
            return Err(QueryCodecError::InvalidMagic(magic));
        }
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags > 3 || flags == 2 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let ticket = QueryTicket::decode_from(&mut decoder)?;
        decoder.finish()?;
        Ok(if flags & 1 != 0 {
            Self::Claim {
                ticket,
                count_lookup: flags & 2 != 0,
            }
        } else {
            Self::Poll(ticket)
        })
    }
}

impl CancelQueryRequest {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, CANCEL_QUERY_MAGIC);
        push_u16(&mut bytes, QUERY_VERSION);
        push_u16(&mut bytes, 0);
        self.ticket.encode_into(&mut bytes)?;
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(CANCEL_QUERY_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let ticket = QueryTicket::decode_from(&mut decoder)?;
        decoder.finish()?;
        Ok(Self { ticket })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreLease {
    pub lease: Vec<u8>,
    pub block_ids_by_group: Vec<Vec<Option<u32>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreRequest {
    pub instance_id: String,
    pub tp_rank: u32,
    pub device_id: i32,
    pub layer_groups: Vec<Vec<String>>,
    pub loads: Vec<RestoreLease>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestoreCommand {
    Submit(RestoreRequest),
    Poll { operation_id: u64 },
}

impl RestoreCommand {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        match self {
            Self::Submit(request) => request.encode(),
            Self::Poll { operation_id } => {
                let mut bytes = Vec::with_capacity(16);
                push_u32(&mut bytes, RESTORE_POLL_MAGIC);
                push_u16(&mut bytes, QUERY_VERSION);
                push_u16(&mut bytes, 0);
                push_u64(&mut bytes, *operation_id);
                Ok(bytes)
            }
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let magic = bytes
            .get(0..4)
            .ok_or(QueryCodecError::Truncated)
            .map(|value| u32::from_le_bytes(value.try_into().expect("fixed slice")))?;
        if magic == RESTORE_REQUEST_MAGIC {
            return Ok(Self::Submit(RestoreRequest::decode(bytes)?));
        }
        if magic != RESTORE_POLL_MAGIC {
            return Err(QueryCodecError::InvalidMagic(magic));
        }
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(RESTORE_POLL_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let operation_id = decoder.u64()?;
        decoder.finish()?;
        if operation_id == 0 {
            return Err(QueryCodecError::ZeroOperationId);
        }
        Ok(Self::Poll { operation_id })
    }
}

impl RestoreRequest {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        let instance = self.instance_id.as_bytes();
        let mut bytes = Vec::with_capacity(RESTORE_HEADER_BYTES + instance.len());
        push_u32(&mut bytes, RESTORE_REQUEST_MAGIC);
        push_u16(&mut bytes, QUERY_VERSION);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, self.tp_rank);
        push_i32(&mut bytes, self.device_id);
        push_u32(&mut bytes, checked_u32(instance.len(), "instance_id")?);
        push_u32(
            &mut bytes,
            checked_u32(self.layer_groups.len(), "layer_groups")?,
        );
        push_u32(&mut bytes, checked_u32(self.loads.len(), "loads")?);
        bytes.extend_from_slice(instance);
        for group in &self.layer_groups {
            push_u32(&mut bytes, checked_u32(group.len(), "layer_group")?);
            for layer in group {
                push_bytes(&mut bytes, layer.as_bytes(), "layer_name")?;
            }
        }
        for load in &self.loads {
            push_bytes(&mut bytes, &load.lease, "lease")?;
            push_u32(
                &mut bytes,
                checked_u32(load.block_ids_by_group.len(), "block_ids_by_group")?,
            );
            for targets in &load.block_ids_by_group {
                push_u32(&mut bytes, checked_u32(targets.len(), "block_targets")?);
                for target in targets {
                    push_u32(&mut bytes, target.map_or(u32::MAX, |block_id| block_id));
                }
            }
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(RESTORE_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let tp_rank = decoder.u32()?;
        let device_id = decoder.i32()?;
        let instance_len = decoder.usize_u32()?;
        let group_count = decoder.usize_u32()?;
        let load_count = decoder.usize_u32()?;
        let instance_id = decoder.string(instance_len, "instance_id")?;
        if group_count > decoder.remaining() / 4 {
            return Err(QueryCodecError::Truncated);
        }
        let mut layer_groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            let layer_count = decoder.usize_u32()?;
            if layer_count > decoder.remaining() / 4 {
                return Err(QueryCodecError::Truncated);
            }
            let mut group = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let len = decoder.usize_u32()?;
                group.push(decoder.string(len, "layer_name")?);
            }
            layer_groups.push(group);
        }
        if load_count > decoder.remaining() / 8 {
            return Err(QueryCodecError::Truncated);
        }
        let mut loads = Vec::with_capacity(load_count);
        for _ in 0..load_count {
            let lease_len = decoder.usize_u32()?;
            let lease = decoder.bytes(lease_len)?.to_vec();
            if lease.is_empty() {
                return Err(QueryCodecError::EmptyLease);
            }
            let target_group_count = decoder.usize_u32()?;
            if target_group_count > decoder.remaining() / 4 {
                return Err(QueryCodecError::Truncated);
            }
            let mut block_ids_by_group = Vec::with_capacity(target_group_count);
            for _ in 0..target_group_count {
                let target_count = decoder.usize_u32()?;
                if target_count > decoder.remaining() / 4 {
                    return Err(QueryCodecError::Truncated);
                }
                let mut targets = Vec::with_capacity(target_count);
                for _ in 0..target_count {
                    let target = decoder.u32()?;
                    targets.push((target != u32::MAX).then_some(target));
                }
                block_ids_by_group.push(targets);
            }
            loads.push(RestoreLease {
                lease,
                block_ids_by_group,
            });
        }
        decoder.finish()?;
        Ok(Self {
            instance_id,
            tp_rank,
            device_id,
            layer_groups,
            loads,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum RestoreState {
    Pending = 1,
    Succeeded = 2,
    Failed = 3,
}

impl TryFrom<u16> for RestoreState {
    type Error = QueryCodecError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Pending),
            2 => Ok(Self::Succeeded),
            3 => Ok(Self::Failed),
            _ => Err(QueryCodecError::UnknownRestoreState(value)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreResponse {
    pub operation_id: u64,
    pub state: RestoreState,
    pub message: String,
}

impl RestoreResponse {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        let message = self.message.as_bytes();
        let mut bytes = Vec::with_capacity(RESTORE_RESPONSE_BYTES + message.len());
        push_u32(&mut bytes, RESTORE_RESPONSE_MAGIC);
        push_u16(&mut bytes, QUERY_VERSION);
        push_u16(&mut bytes, self.state as u16);
        push_u64(&mut bytes, self.operation_id);
        push_u32(&mut bytes, checked_u32(message.len(), "restore_message")?);
        push_u32(&mut bytes, 0);
        bytes.extend_from_slice(message);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(RESTORE_RESPONSE_MAGIC)?;
        decoder.expect_version()?;
        let state = RestoreState::try_from(decoder.u16()?)?;
        let operation_id = decoder.u64()?;
        let message_len = decoder.usize_u32()?;
        let reserved = decoder.u32()?;
        if reserved != 0 {
            return Err(QueryCodecError::InvalidReserved(reserved));
        }
        let message = decoder.string(message_len, "restore_message")?;
        decoder.finish()?;
        Ok(Self {
            operation_id,
            state,
            message,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishLayer {
    pub layer_name: String,
    pub block_ids: Vec<u32>,
    pub block_hashes: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishRequest {
    pub instance_id: String,
    pub tp_rank: u32,
    pub pp_rank: u32,
    pub device_id: i32,
    pub layers: Vec<PublishLayer>,
}

impl PublishRequest {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        validate_publish_layers(&self.layers)?;
        let instance = self.instance_id.as_bytes();
        let mut bytes = Vec::with_capacity(PUBLISH_HEADER_BYTES + instance.len());
        push_u32(&mut bytes, PUBLISH_REQUEST_MAGIC);
        push_u16(&mut bytes, QUERY_VERSION);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, self.tp_rank);
        push_u32(&mut bytes, self.pp_rank);
        push_i32(&mut bytes, self.device_id);
        push_u32(&mut bytes, checked_u32(instance.len(), "instance_id")?);
        push_u32(&mut bytes, checked_u32(self.layers.len(), "layers")?);
        bytes.extend_from_slice(instance);
        for layer in &self.layers {
            let name = layer.layer_name.as_bytes();
            push_u32(&mut bytes, checked_u32(name.len(), "layer_name")?);
            push_u32(&mut bytes, checked_u32(layer.block_ids.len(), "block_ids")?);
            bytes.extend_from_slice(name);
            for (block_id, hash) in layer.block_ids.iter().zip(&layer.block_hashes) {
                push_u32(&mut bytes, *block_id);
                push_u32(&mut bytes, checked_u32(hash.len(), "block_hash")?);
                bytes.extend_from_slice(hash);
            }
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(PUBLISH_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let tp_rank = decoder.u32()?;
        let pp_rank = decoder.u32()?;
        let device_id = decoder.i32()?;
        let instance_len = decoder.usize_u32()?;
        let layer_count = decoder.usize_u32()?;
        let instance_id = decoder.string(instance_len, "instance_id")?;
        if layer_count > decoder.remaining() / 8 {
            return Err(QueryCodecError::Truncated);
        }
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let name_len = decoder.usize_u32()?;
            let block_count = decoder.usize_u32()?;
            let layer_name = decoder.string(name_len, "layer_name")?;
            if block_count > decoder.remaining() / 8 {
                return Err(QueryCodecError::Truncated);
            }
            let mut block_ids = Vec::with_capacity(block_count);
            let mut block_hashes = Vec::with_capacity(block_count);
            for _ in 0..block_count {
                block_ids.push(decoder.u32()?);
                let hash_len = decoder.usize_u32()?;
                block_hashes.push(decoder.bytes(hash_len)?.to_vec());
            }
            layers.push(PublishLayer {
                layer_name,
                block_ids,
                block_hashes,
            });
        }
        decoder.finish()?;
        validate_publish_layers(&layers)?;
        Ok(Self {
            instance_id,
            tp_rank,
            pp_rank,
            device_id,
            layers,
        })
    }
}

fn validate_publish_layers(layers: &[PublishLayer]) -> Result<(), QueryCodecError> {
    for layer in layers {
        if layer.block_ids.len() != layer.block_hashes.len() {
            return Err(QueryCodecError::PublishShapeMismatch {
                layer: layer.layer_name.clone(),
                block_ids: layer.block_ids.len(),
                block_hashes: layer.block_hashes.len(),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseRequest {
    pub lease: Vec<u8>,
}

impl ReleaseRequest {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        let mut bytes = Vec::with_capacity(RELEASE_HEADER_BYTES + self.lease.len());
        push_u32(&mut bytes, RELEASE_REQUEST_MAGIC);
        push_u16(&mut bytes, QUERY_VERSION);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, checked_u32(self.lease.len(), "lease")?);
        bytes.extend_from_slice(&self.lease);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(RELEASE_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags != 0 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let lease_len = decoder.usize_u32()?;
        let lease = decoder.bytes(lease_len)?.to_vec();
        decoder.finish()?;
        if lease.is_empty() {
            return Err(QueryCodecError::EmptyLease);
        }
        Ok(Self { lease })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryBundleRequest {
    pub ticket: QueryTicket,
    pub instance_id: String,
    pub request_id: String,
    pub block_hashes: Vec<Vec<u8>>,
    pub group_id: u32,
    pub wait_for_full_prefix: bool,
    /// Best-effort preparation without a restore lease. Never waits for publication.
    pub warmup: bool,
    /// Metadata hints only: no payload read, byte reservation, or restore lease.
    pub discover: bool,
    /// Read a selected recovery range already counted by candidate discovery.
    pub materialize: bool,
    /// Keep a consumer-owned result at the Manager until its first demand poll.
    pub prepare: bool,
}

impl QueryBundleRequest {
    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        if self.request_id.is_empty() {
            return Err(QueryCodecError::EmptyRequestId);
        }
        let instance = self.instance_id.as_bytes();
        let request = self.request_id.as_bytes();
        let mut bytes = Vec::with_capacity(
            REQUEST_HEADER_BYTES
                + instance.len()
                + request.len()
                + self
                    .block_hashes
                    .iter()
                    .map(|hash| 4 + hash.len())
                    .sum::<usize>(),
        );
        push_u32(&mut bytes, QUERY_REQUEST_MAGIC);
        push_u16(&mut bytes, QUERY_VERSION);
        push_u16(
            &mut bytes,
            u16::from(self.wait_for_full_prefix)
                | (u16::from(self.warmup) << 1)
                | (u16::from(self.discover) << 2)
                | (u16::from(self.materialize) << 3)
                | (u16::from(self.prepare) << 4),
        );
        self.ticket.encode_into(&mut bytes)?;
        push_u32(&mut bytes, self.group_id);
        push_u32(&mut bytes, checked_u32(instance.len(), "instance_id")?);
        push_u32(&mut bytes, checked_u32(request.len(), "request_id")?);
        push_u32(
            &mut bytes,
            checked_u32(self.block_hashes.len(), "block_hashes")?,
        );
        bytes.extend_from_slice(instance);
        bytes.extend_from_slice(request);
        for hash in &self.block_hashes {
            push_u32(&mut bytes, checked_u32(hash.len(), "block_hash")?);
            bytes.extend_from_slice(hash);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(QUERY_REQUEST_MAGIC)?;
        decoder.expect_version()?;
        let flags = decoder.u16()?;
        if flags & !31 != 0 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let ticket = QueryTicket::decode_from(&mut decoder)?;
        let group_id = decoder.u32()?;
        let instance_len = decoder.usize_u32()?;
        let request_len = decoder.usize_u32()?;
        let hash_count = decoder.usize_u32()?;
        let instance_id = decoder.string(instance_len, "instance_id")?;
        let request_id = decoder.string(request_len, "request_id")?;
        if request_id.is_empty() {
            return Err(QueryCodecError::EmptyRequestId);
        }
        if hash_count > decoder.remaining() / 4 {
            return Err(QueryCodecError::Truncated);
        }
        let mut block_hashes = Vec::with_capacity(hash_count);
        for _ in 0..hash_count {
            let len = decoder.usize_u32()?;
            block_hashes.push(decoder.bytes(len)?.to_vec());
        }
        decoder.finish()?;
        Ok(Self {
            ticket,
            instance_id,
            request_id,
            block_hashes,
            group_id,
            wait_for_full_prefix: flags & 1 != 0,
            warmup: flags & 2 != 0,
            discover: flags & 4 != 0,
            materialize: flags & 8 != 0,
            prepare: flags & 16 != 0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum QueryOutcomeCode {
    Ready = 1,
    Loading = 2,
    Busy = 3,
    Candidates = 4,
}

impl TryFrom<u16> for QueryOutcomeCode {
    type Error = QueryCodecError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ready),
            2 => Ok(Self::Loading),
            3 => Ok(Self::Busy),
            4 => Ok(Self::Candidates),
            _ => Err(QueryCodecError::UnknownOutcome(value)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryBundleResponse {
    pub outcome: QueryOutcomeCode,
    pub num_hit_blocks: u64,
    pub lease: Vec<u8>,
    pub hit_positions: Vec<u32>,
}

impl QueryBundleResponse {
    pub fn loading() -> Self {
        Self {
            outcome: QueryOutcomeCode::Loading,
            num_hit_blocks: 0,
            lease: Vec::new(),
            hit_positions: Vec::new(),
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, QueryCodecError> {
        let mut bytes = Vec::with_capacity(
            RESPONSE_HEADER_BYTES + self.lease.len() + self.hit_positions.len() * 4,
        );
        push_u32(&mut bytes, QUERY_RESPONSE_MAGIC);
        push_u16(&mut bytes, QUERY_VERSION);
        push_u16(&mut bytes, self.outcome as u16);
        push_u64(&mut bytes, self.num_hit_blocks);
        push_u32(&mut bytes, checked_u32(self.lease.len(), "lease")?);
        push_u32(
            &mut bytes,
            checked_u32(self.hit_positions.len(), "hit_positions")?,
        );
        bytes.extend_from_slice(&self.lease);
        for position in &self.hit_positions {
            push_u32(&mut bytes, *position);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, QueryCodecError> {
        let mut decoder = Decoder::new(bytes);
        decoder.expect_magic(QUERY_RESPONSE_MAGIC)?;
        decoder.expect_version()?;
        let outcome = QueryOutcomeCode::try_from(decoder.u16()?)?;
        let num_hit_blocks = decoder.u64()?;
        let lease_len = decoder.usize_u32()?;
        let positions_len = decoder.usize_u32()?;
        let lease = decoder.bytes(lease_len)?.to_vec();
        if positions_len > decoder.remaining() / 4 {
            return Err(QueryCodecError::Truncated);
        }
        let mut hit_positions = Vec::with_capacity(positions_len);
        for _ in 0..positions_len {
            hit_positions.push(decoder.u32()?);
        }
        decoder.finish()?;
        if matches!(outcome, QueryOutcomeCode::Loading | QueryOutcomeCode::Busy)
            && (num_hit_blocks != 0 || !lease.is_empty() || !hit_positions.is_empty())
        {
            return Err(QueryCodecError::InvalidLoadingPayload);
        }
        if outcome == QueryOutcomeCode::Candidates
            && (!lease.is_empty()
                || num_hit_blocks != hit_positions.len() as u64
                || hit_positions.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(QueryCodecError::InvalidCandidates);
        }
        Ok(Self {
            outcome,
            num_hit_blocks,
            lease,
            hit_positions,
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum QueryCodecError {
    #[error("candidate hints must be ordered, unique and unleased")]
    InvalidCandidates,
    #[error("query operation and revision must be nonzero")]
    InvalidQueryTicket,
    #[error("query payload is truncated")]
    Truncated,
    #[error("query payload has {0} trailing bytes")]
    TrailingBytes(usize),
    #[error("invalid query magic: {0:#x}")]
    InvalidMagic(u32),
    #[error("unsupported query version: {0}")]
    UnsupportedVersion(u16),
    #[error("invalid query flags: {0:#x}")]
    InvalidFlags(u16),
    #[error("unknown query outcome: {0}")]
    UnknownOutcome(u16),
    #[error("query request_id must not be empty")]
    EmptyRequestId,
    #[error("release lease must not be empty")]
    EmptyLease,
    #[error("query field {field} is too large: {len}")]
    FieldTooLarge { field: &'static str, len: usize },
    #[error("query field {field} is not valid UTF-8")]
    InvalidUtf8 { field: &'static str },
    #[error("loading query outcome must not carry ready data")]
    InvalidLoadingPayload,
    #[error("publish layer {layer} has {block_ids} block ids but {block_hashes} block hashes")]
    PublishShapeMismatch {
        layer: String,
        block_ids: usize,
        block_hashes: usize,
    },
    #[error("unknown restore state: {0}")]
    UnknownRestoreState(u16),
    #[error("restore response reserved field must be zero, got {0}")]
    InvalidReserved(u32),
    #[error("restore operation id must be non-zero")]
    ZeroOperationId,
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_magic(&mut self, expected: u32) -> Result<(), QueryCodecError> {
        let actual = self.u32()?;
        if actual != expected {
            return Err(QueryCodecError::InvalidMagic(actual));
        }
        Ok(())
    }

    fn expect_version(&mut self) -> Result<(), QueryCodecError> {
        let version = self.u16()?;
        if version != QUERY_VERSION {
            return Err(QueryCodecError::UnsupportedVersion(version));
        }
        Ok(())
    }

    fn u16(&mut self) -> Result<u16, QueryCodecError> {
        Ok(u16::from_le_bytes(
            self.bytes(2)?.try_into().expect("fixed slice"),
        ))
    }

    fn u32(&mut self) -> Result<u32, QueryCodecError> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().expect("fixed slice"),
        ))
    }

    fn i32(&mut self) -> Result<i32, QueryCodecError> {
        Ok(i32::from_le_bytes(
            self.bytes(4)?.try_into().expect("fixed slice"),
        ))
    }

    fn u64(&mut self) -> Result<u64, QueryCodecError> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().expect("fixed slice"),
        ))
    }

    fn usize_u32(&mut self) -> Result<usize, QueryCodecError> {
        Ok(self.u32()? as usize)
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], QueryCodecError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(QueryCodecError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(QueryCodecError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn string(&mut self, len: usize, field: &'static str) -> Result<String, QueryCodecError> {
        String::from_utf8(self.bytes(len)?.to_vec())
            .map_err(|_| QueryCodecError::InvalidUtf8 { field })
    }

    fn finish(self) -> Result<(), QueryCodecError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(QueryCodecError::TrailingBytes(
                self.bytes.len() - self.offset,
            ))
        }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, QueryCodecError> {
    u32::try_from(value).map_err(|_| QueryCodecError::FieldTooLarge { field, len: value })
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_bytes(
    bytes: &mut Vec<u8>,
    value: &[u8],
    field: &'static str,
) -> Result<(), QueryCodecError> {
    push_u32(bytes, checked_u32(value.len(), field)?);
    bytes.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/cache_protocol.rs"]
mod tests;
