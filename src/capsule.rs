use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

const KEY_MAGIC: &[u8] = b"orbitkv/capsule/v1\0";
const CHUNK_MARKER: u8 = 1;
const CAPSULE_SCHEMA: &str = "orbitkv.continuation-capsule.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ContentDigest(pub [u8; 32]);

impl ContentDigest {
    #[must_use]
    pub fn sha256(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(char::from(HEX[usize::from(byte >> 4)]));
            output.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        output
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CapsuleIdentity {
    pub namespace: Vec<u8>,
    pub model_fingerprint: ContentDigest,
    pub tokenizer_fingerprint: ContentDigest,
    pub adapter_fingerprint: ContentDigest,
    pub state_plan_fingerprint: ContentDigest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixChunk {
    pub digest: ContentDigest,
    pub end_token: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixPath {
    identity: CapsuleIdentity,
    chunk_tokens: u32,
    chunks: Vec<PrefixChunk>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CapsuleComponent {
    pub state_class: String,
    pub offset_bytes: u64,
    pub length_bytes: u64,
    pub checksum: ContentDigest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_start: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_end_exclusive: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CapsuleManifest {
    pub schema: String,
    pub capsule_id: ContentDigest,
    pub identity: CapsuleIdentity,
    pub prefix_token_count: u64,
    pub live_token_count: u64,
    pub payload_digest: ContentDigest,
    pub payload_bytes: u64,
    pub components: Vec<CapsuleComponent>,
    pub created_unix_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CapsuleComponentSpec {
    pub state_class: String,
    pub length_bytes: u64,
    #[serde(default)]
    pub token_start: Option<u64>,
    #[serde(default)]
    pub token_end_exclusive: Option<u64>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CapsuleError {
    #[error("capsule namespace is too long")]
    NamespaceTooLong,
    #[error("capsule chunk_tokens must be positive")]
    ZeroChunkTokens,
    #[error("capsule prefix must contain at least one chunk")]
    EmptyPrefix,
    #[error("capsule chunk boundary is outside the prefix")]
    InvalidChunkBoundary,
    #[error("capsule chunk end tokens must be strictly increasing")]
    NonIncreasingChunkBoundary,
    #[error("capsule prefix boundary must align with chunk_tokens")]
    MisalignedPrefixBoundary,
    #[error("capsule key exceeds the persistent ART key limit")]
    KeyTooLong,
    #[error("capsule components must be non-empty")]
    EmptyComponents,
    #[error("capsule components do not densely cover the payload")]
    InvalidComponentCoverage,
    #[error("capsule component checksum differs from payload bytes")]
    ComponentChecksumMismatch,
    #[error("capsule component token range is invalid")]
    InvalidComponentTokenRange,
    #[error("capsule manifest identity differs from its prefix key")]
    IdentityMismatch,
    #[error("capsule manifest prefix boundary differs from its prefix key")]
    PrefixBoundaryMismatch,
    #[error("capsule manifest schema is unsupported")]
    UnsupportedSchema,
    #[error("capsule id does not authenticate the manifest")]
    CapsuleIdMismatch,
    #[error("capsule payload length differs from its manifest")]
    PayloadLengthMismatch,
    #[error("capsule payload digest differs from its manifest")]
    PayloadDigestMismatch,
}

impl PrefixPath {
    /// Builds a content-addressed prefix path from token ids.
    ///
    /// # Errors
    ///
    /// Returns an error for zero chunk size, an empty prefix, invalid
    /// boundaries, or an encoded key larger than Holt's public key limit.
    pub fn from_token_ids(
        identity: CapsuleIdentity,
        chunk_tokens: u32,
        token_ids: &[u32],
    ) -> Result<Self, CapsuleError> {
        if chunk_tokens == 0 {
            return Err(CapsuleError::ZeroChunkTokens);
        }
        if token_ids.is_empty() {
            return Err(CapsuleError::EmptyPrefix);
        }
        let chunk_size = usize::try_from(chunk_tokens).map_err(|_| CapsuleError::KeyTooLong)?;
        let chunks = token_ids
            .chunks(chunk_size)
            .enumerate()
            .map(|(index, chunk)| {
                let mut bytes = Vec::with_capacity(std::mem::size_of_val(chunk));
                for token in chunk {
                    bytes.extend_from_slice(&token.to_le_bytes());
                }
                let end_token = index
                    .checked_mul(chunk_size)
                    .and_then(|start| start.checked_add(chunk.len()))
                    .and_then(|end| u64::try_from(end).ok())
                    .ok_or(CapsuleError::KeyTooLong)?;
                Ok(PrefixChunk {
                    digest: ContentDigest::sha256(&bytes),
                    end_token,
                })
            })
            .collect::<Result<Vec<_>, CapsuleError>>()?;
        Self::new(identity, chunk_tokens, chunks)
    }

    /// Builds a path from pre-hashed token chunks.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid geometry or an oversized encoded key.
    pub fn new(
        identity: CapsuleIdentity,
        chunk_tokens: u32,
        chunks: Vec<PrefixChunk>,
    ) -> Result<Self, CapsuleError> {
        if identity.namespace.len() > usize::from(u16::MAX) {
            return Err(CapsuleError::NamespaceTooLong);
        }
        if chunk_tokens == 0 {
            return Err(CapsuleError::ZeroChunkTokens);
        }
        if chunks.is_empty() {
            return Err(CapsuleError::EmptyPrefix);
        }
        if chunks
            .windows(2)
            .any(|pair| pair[0].end_token >= pair[1].end_token)
            || chunks[0].end_token == 0
        {
            return Err(CapsuleError::NonIncreasingChunkBoundary);
        }
        if chunks.iter().enumerate().any(|(index, chunk)| {
            let Some(chunk_number) = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
            else {
                return true;
            };
            let Some(full_boundary) = chunk_number.checked_mul(u64::from(chunk_tokens)) else {
                return true;
            };
            if index + 1 == chunks.len() {
                let previous_boundary = full_boundary - u64::from(chunk_tokens);
                chunk.end_token <= previous_boundary || chunk.end_token > full_boundary
            } else {
                chunk.end_token != full_boundary
            }
        }) {
            return Err(CapsuleError::MisalignedPrefixBoundary);
        }
        let path = Self {
            identity,
            chunk_tokens,
            chunks,
        };
        path.catalog_key_at(path.chunks.len())?;
        Ok(path)
    }

    #[must_use]
    pub fn identity(&self) -> &CapsuleIdentity {
        &self.identity
    }

    #[must_use]
    pub const fn chunk_tokens(&self) -> u32 {
        self.chunk_tokens
    }

    #[must_use]
    pub fn chunks(&self) -> &[PrefixChunk] {
        &self.chunks
    }

    #[must_use]
    pub fn token_count(&self) -> u64 {
        self.chunks.last().map_or(0, |chunk| chunk.end_token)
    }

    #[must_use]
    pub fn is_chunk_aligned(&self) -> bool {
        self.token_count()
            .is_multiple_of(u64::from(self.chunk_tokens))
    }

    /// Encodes one exact capsule boundary after `chunk_count` chunks.
    ///
    /// # Errors
    ///
    /// Returns an error when the boundary is invalid or exceeds Holt's key limit.
    pub fn catalog_key_at(&self, chunk_count: usize) -> Result<Vec<u8>, CapsuleError> {
        if chunk_count == 0 || chunk_count > self.chunks.len() {
            return Err(CapsuleError::InvalidChunkBoundary);
        }
        let mut key = Vec::with_capacity(
            KEY_MAGIC.len() + 2 + self.identity.namespace.len() + 4 * 32 + 4 + chunk_count * 33,
        );
        key.extend_from_slice(KEY_MAGIC);
        key.extend_from_slice(
            &u16::try_from(self.identity.namespace.len())
                .map_err(|_| CapsuleError::NamespaceTooLong)?
                .to_be_bytes(),
        );
        key.extend_from_slice(&self.identity.namespace);
        for digest in [
            self.identity.model_fingerprint,
            self.identity.tokenizer_fingerprint,
            self.identity.adapter_fingerprint,
            self.identity.state_plan_fingerprint,
        ] {
            key.extend_from_slice(&digest.0);
        }
        key.extend_from_slice(&self.chunk_tokens.to_be_bytes());
        for chunk in &self.chunks[..chunk_count] {
            key.push(CHUNK_MARKER);
            key.extend_from_slice(&chunk.digest.0);
        }
        if key.len() > usize::from(u16::MAX) {
            return Err(CapsuleError::KeyTooLong);
        }
        Ok(key)
    }

    /// Returns the number of complete chunks encoded by a catalog key.
    ///
    /// # Errors
    ///
    /// Returns an error when the key does not belong to this path or ends
    /// between chunk boundaries.
    pub fn chunk_count_for_catalog_key(&self, key: &[u8]) -> Result<usize, CapsuleError> {
        let base = self.catalog_key_at(1)?;
        let base_len = base
            .len()
            .checked_sub(33)
            .ok_or(CapsuleError::InvalidChunkBoundary)?;
        if key.len() < base_len || key.get(..base_len) != base.get(..base_len) {
            return Err(CapsuleError::IdentityMismatch);
        }
        let suffix_len = key.len() - base_len;
        if suffix_len == 0 || !suffix_len.is_multiple_of(33) {
            return Err(CapsuleError::InvalidChunkBoundary);
        }
        let chunk_count = suffix_len / 33;
        if chunk_count > self.chunks.len() || self.catalog_key_at(chunk_count)? != key {
            return Err(CapsuleError::PrefixBoundaryMismatch);
        }
        Ok(chunk_count)
    }

    /// Returns the exact path ending after `chunk_count` chunks.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested boundary is outside this path.
    pub fn prefix_at(&self, chunk_count: usize) -> Result<Self, CapsuleError> {
        if chunk_count == 0 || chunk_count > self.chunks.len() {
            return Err(CapsuleError::InvalidChunkBoundary);
        }
        Self::new(
            self.identity.clone(),
            self.chunk_tokens,
            self.chunks[..chunk_count].to_vec(),
        )
    }

    /// Returns exact boundary keys from deepest to shallowest.
    ///
    /// # Errors
    ///
    /// Returns an error if any encoded boundary exceeds Holt's key limit.
    pub fn candidate_keys_desc(&self) -> Result<Vec<Vec<u8>>, CapsuleError> {
        (1..=self.chunks.len())
            .rev()
            .map(|count| self.catalog_key_at(count))
            .collect()
    }
}

impl CapsuleManifest {
    /// Constructs and validates one immutable continuation-capsule manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when components do not densely cover the payload.
    pub fn new(
        path: &PrefixPath,
        live_token_count: u64,
        payload: &[u8],
        components: Vec<CapsuleComponent>,
        created_unix_ms: u64,
    ) -> Result<Self, CapsuleError> {
        validate_components(payload, &components)?;
        validate_component_ranges(path.token_count(), &components)?;
        let payload_digest = ContentDigest::sha256(payload);
        Ok(Self {
            schema: CAPSULE_SCHEMA.into(),
            capsule_id: capsule_id(path, payload_digest, live_token_count, &components)?,
            identity: path.identity.clone(),
            prefix_token_count: path.token_count(),
            live_token_count,
            payload_digest,
            payload_bytes: u64::try_from(payload.len())
                .map_err(|_| CapsuleError::InvalidComponentCoverage)?,
            components,
            created_unix_ms,
        })
    }

    /// Validates that this manifest belongs to the supplied prefix path.
    ///
    /// # Errors
    ///
    /// Returns an error for identity or boundary mismatch.
    pub fn validate_for_path(&self, path: &PrefixPath) -> Result<(), CapsuleError> {
        if self.schema != CAPSULE_SCHEMA {
            return Err(CapsuleError::UnsupportedSchema);
        }
        if self.identity != *path.identity() {
            return Err(CapsuleError::IdentityMismatch);
        }
        if self.prefix_token_count != path.token_count() {
            return Err(CapsuleError::PrefixBoundaryMismatch);
        }
        validate_component_ranges(path.token_count(), &self.components)?;
        if self.capsule_id
            != capsule_id(
                path,
                self.payload_digest,
                self.live_token_count,
                &self.components,
            )?
        {
            return Err(CapsuleError::CapsuleIdMismatch);
        }
        Ok(())
    }

    /// Validates the complete manifest and immutable payload.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid path binding, payload digest, length,
    /// component coverage, or component checksum.
    pub fn validate(&self, path: &PrefixPath, payload: &[u8]) -> Result<(), CapsuleError> {
        self.validate_for_path(path)?;
        if self.payload_bytes
            != u64::try_from(payload.len()).map_err(|_| CapsuleError::PayloadLengthMismatch)?
        {
            return Err(CapsuleError::PayloadLengthMismatch);
        }
        if self.payload_digest != ContentDigest::sha256(payload) {
            return Err(CapsuleError::PayloadDigestMismatch);
        }
        validate_components(payload, &self.components)
    }
}

/// Builds dense component metadata from ordered component lengths.
///
/// # Errors
///
/// Returns an error when lengths overflow, do not cover the payload, or name
/// an empty component.
pub fn build_capsule_components(
    payload: &[u8],
    specs: &[CapsuleComponentSpec],
) -> Result<Vec<CapsuleComponent>, CapsuleError> {
    if specs.is_empty() {
        return Err(CapsuleError::EmptyComponents);
    }
    let mut offset_bytes = 0_u64;
    let mut components = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.state_class.is_empty() {
            return Err(CapsuleError::InvalidComponentCoverage);
        }
        let end = offset_bytes
            .checked_add(spec.length_bytes)
            .ok_or(CapsuleError::InvalidComponentCoverage)?;
        let start =
            usize::try_from(offset_bytes).map_err(|_| CapsuleError::InvalidComponentCoverage)?;
        let end_usize = usize::try_from(end).map_err(|_| CapsuleError::InvalidComponentCoverage)?;
        let bytes = payload
            .get(start..end_usize)
            .ok_or(CapsuleError::InvalidComponentCoverage)?;
        components.push(CapsuleComponent {
            state_class: spec.state_class.clone(),
            offset_bytes,
            length_bytes: spec.length_bytes,
            checksum: ContentDigest::sha256(bytes),
            token_start: spec.token_start,
            token_end_exclusive: spec.token_end_exclusive,
        });
        offset_bytes = end;
    }
    if offset_bytes
        != u64::try_from(payload.len()).map_err(|_| CapsuleError::InvalidComponentCoverage)?
    {
        return Err(CapsuleError::InvalidComponentCoverage);
    }
    Ok(components)
}

fn capsule_id(
    path: &PrefixPath,
    payload_digest: ContentDigest,
    live_token_count: u64,
    components: &[CapsuleComponent],
) -> Result<ContentDigest, CapsuleError> {
    let mut material = path.catalog_key_at(path.chunks.len())?;
    material.extend_from_slice(&payload_digest.0);
    material.extend_from_slice(&live_token_count.to_le_bytes());
    for component in components {
        let name = component.state_class.as_bytes();
        material.extend_from_slice(
            &u64::try_from(name.len())
                .map_err(|_| CapsuleError::InvalidComponentCoverage)?
                .to_le_bytes(),
        );
        material.extend_from_slice(name);
        material.extend_from_slice(&component.offset_bytes.to_le_bytes());
        material.extend_from_slice(&component.length_bytes.to_le_bytes());
        material.extend_from_slice(&component.checksum.0);
        if let (Some(start), Some(end)) = (component.token_start, component.token_end_exclusive) {
            material.extend_from_slice(b"orbitkv-component-token-range-v1");
            material.extend_from_slice(&start.to_le_bytes());
            material.extend_from_slice(&end.to_le_bytes());
        }
    }
    Ok(ContentDigest::sha256(&material))
}

fn validate_component_ranges(
    prefix_token_count: u64,
    components: &[CapsuleComponent],
) -> Result<(), CapsuleError> {
    for component in components {
        match (component.token_start, component.token_end_exclusive) {
            (None, None) => {}
            (Some(start), Some(end)) if start < end && end <= prefix_token_count => {}
            _ => return Err(CapsuleError::InvalidComponentTokenRange),
        }
    }
    Ok(())
}

fn validate_components(
    payload: &[u8],
    components: &[CapsuleComponent],
) -> Result<(), CapsuleError> {
    if components.is_empty() {
        return Err(CapsuleError::EmptyComponents);
    }
    let mut expected_offset = 0_u64;
    for component in components {
        if component.offset_bytes != expected_offset {
            return Err(CapsuleError::InvalidComponentCoverage);
        }
        let end = component
            .offset_bytes
            .checked_add(component.length_bytes)
            .ok_or(CapsuleError::InvalidComponentCoverage)?;
        let start = usize::try_from(component.offset_bytes)
            .map_err(|_| CapsuleError::InvalidComponentCoverage)?;
        let end_usize = usize::try_from(end).map_err(|_| CapsuleError::InvalidComponentCoverage)?;
        let bytes = payload
            .get(start..end_usize)
            .ok_or(CapsuleError::InvalidComponentCoverage)?;
        if ContentDigest::sha256(bytes) != component.checksum {
            return Err(CapsuleError::ComponentChecksumMismatch);
        }
        expected_offset = end;
    }
    if expected_offset
        != u64::try_from(payload.len()).map_err(|_| CapsuleError::InvalidComponentCoverage)?
    {
        return Err(CapsuleError::InvalidComponentCoverage);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> CapsuleIdentity {
        CapsuleIdentity {
            namespace: b"tenant-a".to_vec(),
            model_fingerprint: ContentDigest::sha256(b"model"),
            tokenizer_fingerprint: ContentDigest::sha256(b"tokenizer"),
            adapter_fingerprint: ContentDigest::sha256(b"adapter"),
            state_plan_fingerprint: ContentDigest::sha256(b"state-plan"),
        }
    }

    #[test]
    fn token_prefix_paths_share_art_bytes() {
        let short = PrefixPath::from_token_ids(identity(), 4, &[1, 2, 3, 4]).unwrap();
        let long = PrefixPath::from_token_ids(identity(), 4, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let short_key = short.catalog_key_at(1).unwrap();
        let long_first = long.catalog_key_at(1).unwrap();
        assert_eq!(short_key, long_first);
        let long_key = long.catalog_key_at(2).unwrap();
        assert!(long_key.starts_with(&short_key));
    }

    #[test]
    fn manifest_requires_dense_component_checksums() {
        let path = PrefixPath::from_token_ids(identity(), 4, &[1, 2, 3, 4]).unwrap();
        let payload = b"abcdefgh";
        let components = vec![
            CapsuleComponent {
                state_class: "k".into(),
                offset_bytes: 0,
                length_bytes: 4,
                checksum: ContentDigest::sha256(&payload[..4]),
                token_start: None,
                token_end_exclusive: None,
            },
            CapsuleComponent {
                state_class: "v".into(),
                offset_bytes: 4,
                length_bytes: 4,
                checksum: ContentDigest::sha256(&payload[4..]),
                token_start: None,
                token_end_exclusive: None,
            },
        ];
        let manifest = CapsuleManifest::new(&path, 4, payload, components, 1).unwrap();
        manifest.validate_for_path(&path).unwrap();
    }

    #[test]
    fn component_token_ranges_are_authenticated_and_bounded() {
        let path = PrefixPath::from_token_ids(identity(), 4, &[1, 2, 3, 4]).unwrap();
        let payload = b"abcdefgh";
        let components = build_capsule_components(
            payload,
            &[
                CapsuleComponentSpec {
                    state_class: "full-kv".into(),
                    length_bytes: 4,
                    token_start: Some(0),
                    token_end_exclusive: Some(4),
                },
                CapsuleComponentSpec {
                    state_class: "swa-kv".into(),
                    length_bytes: 4,
                    token_start: Some(2),
                    token_end_exclusive: Some(4),
                },
            ],
        )
        .unwrap();
        let manifest = CapsuleManifest::new(&path, 4, payload, components, 1).unwrap();
        manifest.validate(&path, payload).unwrap();

        let mut invalid = manifest.clone();
        invalid.components[1].token_end_exclusive = Some(5);
        assert_eq!(
            invalid.validate_for_path(&path),
            Err(CapsuleError::InvalidComponentTokenRange)
        );
    }
}
