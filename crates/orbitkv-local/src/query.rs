use thiserror::Error;

const QUERY_REQUEST_MAGIC: u32 = 0x4f52_5151; // ORQQ
const QUERY_RESPONSE_MAGIC: u32 = 0x4f52_5152; // ORQR
const QUERY_VERSION: u16 = 1;
const REQUEST_HEADER_BYTES: usize = 24;
const RESPONSE_HEADER_BYTES: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryBundleRequest {
    pub instance_id: String,
    pub request_id: String,
    pub block_hashes: Vec<Vec<u8>>,
    pub group_id: u32,
    pub wait_for_full_prefix: bool,
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
        push_u16(&mut bytes, u16::from(self.wait_for_full_prefix));
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
        if flags & !1 != 0 {
            return Err(QueryCodecError::InvalidFlags(flags));
        }
        let group_id = decoder.u32()?;
        let instance_len = decoder.usize_u32()?;
        let request_len = decoder.usize_u32()?;
        let hash_count = decoder.usize_u32()?;
        let instance_id = decoder.string(instance_len, "instance_id")?;
        let request_id = decoder.string(request_len, "request_id")?;
        if request_id.is_empty() {
            return Err(QueryCodecError::EmptyRequestId);
        }
        let mut block_hashes = Vec::with_capacity(hash_count.min(1024));
        for _ in 0..hash_count {
            let len = decoder.usize_u32()?;
            block_hashes.push(decoder.bytes(len)?.to_vec());
        }
        decoder.finish()?;
        Ok(Self {
            instance_id,
            request_id,
            block_hashes,
            group_id,
            wait_for_full_prefix: flags & 1 != 0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum QueryOutcomeCode {
    Ready = 1,
    Loading = 2,
}

impl TryFrom<u16> for QueryOutcomeCode {
    type Error = QueryCodecError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ready),
            2 => Ok(Self::Loading),
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
        let mut hit_positions = Vec::with_capacity(positions_len.min(1024));
        for _ in 0..positions_len {
            hit_positions.push(decoder.u32()?);
        }
        decoder.finish()?;
        if outcome == QueryOutcomeCode::Loading
            && (num_hit_blocks != 0 || !lease.is_empty() || !hit_positions.is_empty())
        {
            return Err(QueryCodecError::InvalidLoadingPayload);
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
    #[error("query field {field} is too large: {len}")]
    FieldTooLarge { field: &'static str, len: usize },
    #[error("query field {field} is not valid UTF-8")]
    InvalidUtf8 { field: &'static str },
    #[error("loading query outcome must not carry ready data")]
    InvalidLoadingPayload,
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

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trip_preserves_variable_hashes() {
        let request = QueryBundleRequest {
            instance_id: "model-a".to_string(),
            request_id: "request-7".to_string(),
            block_hashes: vec![vec![1; 32], vec![2; 17]],
            group_id: 3,
            wait_for_full_prefix: true,
        };
        assert_eq!(
            QueryBundleRequest::decode(&request.encode().unwrap()).unwrap(),
            request
        );
    }

    #[test]
    fn response_round_trip_preserves_lease_and_positions() {
        let response = QueryBundleResponse {
            outcome: QueryOutcomeCode::Ready,
            num_hit_blocks: 2,
            lease: vec![3; 16],
            hit_positions: vec![1, 4],
        };
        assert_eq!(
            QueryBundleResponse::decode(&response.encode().unwrap()).unwrap(),
            response
        );
    }

    #[test]
    fn decoder_rejects_trailing_bytes_and_invalid_loading_payload() {
        let mut encoded = QueryBundleResponse::loading().encode().unwrap();
        encoded.push(0);
        assert_eq!(
            QueryBundleResponse::decode(&encoded),
            Err(QueryCodecError::TrailingBytes(1))
        );

        let invalid = QueryBundleResponse {
            outcome: QueryOutcomeCode::Loading,
            num_hit_blocks: 1,
            lease: Vec::new(),
            hit_positions: Vec::new(),
        };
        assert_eq!(
            QueryBundleResponse::decode(&invalid.encode().unwrap()),
            Err(QueryCodecError::InvalidLoadingPayload)
        );
    }
}
