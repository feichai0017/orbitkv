use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use holt::{DB, Durability, Tree, TreeConfig};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::{CapsuleError, CapsuleManifest, ContentDigest, PrefixPath};

const CAPSULE_TREE: &str = "orbitkv/capsules";
const HOLT_VALUE_LIMIT: usize = 65_535;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapsulePublish {
    Published,
    AlreadyPresent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoredCapsule {
    pub manifest: CapsuleManifest,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub struct HoltCapsuleStore {
    db: DB,
    capsules: Tree,
    objects_dir: PathBuf,
}

#[derive(Debug, Error)]
pub enum HoltCapsuleError {
    #[error("capsule validation failed: {0}")]
    Capsule(#[from] CapsuleError),
    #[error("Holt catalog operation failed: {0}")]
    Holt(#[from] holt::Error),
    #[error("capsule payload I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("capsule catalog encoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("capsule catalog value exceeds Holt's value limit")]
    CatalogValueTooLarge,
    #[error("capsules may only be published at complete token-chunk boundaries")]
    UnalignedPublishBoundary,
    #[error("a different capsule is already published at this prefix boundary")]
    PrefixConflict,
    #[error("capsule payload object is invalid")]
    InvalidPayloadObject,
    #[error("capsule payload object is missing")]
    MissingPayload,
}

impl HoltCapsuleStore {
    /// Opens the only persistent Capsule store used by `OrbitKV`.
    ///
    /// Holt stores the prefix catalog. Immutable KV payload bytes are stored
    /// under the same root as content-addressed files because Holt values are
    /// intentionally metadata-sized.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory, Holt database, or named trees
    /// cannot be opened.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, HoltCapsuleError> {
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        let objects_dir = root.join("objects");
        fs::create_dir_all(&objects_dir)?;
        sync_directory(root)?;

        let mut config = TreeConfig::new(root.join("holt"));
        config.durability = Durability::Wal { sync: true };
        config.checkpoint.enabled = false;
        let db = DB::open(config)?;
        let capsules = db.open_or_create_tree(CAPSULE_TREE)?;

        Ok(Self {
            db,
            capsules,
            objects_dir,
        })
    }

    /// Publishes one immutable Capsule using payload-first ordering.
    ///
    /// The payload file is made durable before one Holt conditional write
    /// publishes the prefix manifest.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid Capsule bytes, conflicting prefix state,
    /// storage failures, or catalog values that exceed Holt's limit.
    pub fn publish(
        &self,
        path: &PrefixPath,
        manifest: &CapsuleManifest,
        payload: &[u8],
    ) -> Result<CapsulePublish, HoltCapsuleError> {
        if !path.is_chunk_aligned() {
            return Err(HoltCapsuleError::UnalignedPublishBoundary);
        }
        manifest.validate(path, payload)?;
        self.write_payload(manifest.payload_digest, payload)?;

        let manifest_bytes = encode_catalog_value(manifest)?;
        let capsule_key = path.catalog_key_at(path.chunks().len())?;

        if let Some(existing) = self.capsules.get(&capsule_key)? {
            return self.reconcile_existing(path, manifest, payload, &existing);
        }

        if self.capsules.put_if_absent(&capsule_key, &manifest_bytes)? {
            return Ok(CapsulePublish::Published);
        }

        let existing = self
            .capsules
            .get(&capsule_key)?
            .ok_or(HoltCapsuleError::PrefixConflict)?;
        self.reconcile_existing(path, manifest, payload, &existing)
    }

    /// Restores the deepest published Capsule boundary in `path`.
    ///
    /// Candidate boundaries are checked from deepest to shallowest. Every hit
    /// is authenticated against its exact path, Holt object reference, and
    /// immutable payload before it is returned.
    ///
    /// # Errors
    ///
    /// Returns an error when a published catalog entry or payload is corrupt
    /// or cannot be read.
    pub fn restore_deepest(
        &self,
        path: &PrefixPath,
    ) -> Result<Option<RestoredCapsule>, HoltCapsuleError> {
        let Some((candidate_path, manifest)) = self.lookup_deepest(path)? else {
            return Ok(None);
        };
        let payload = self.restore_payload(&candidate_path, &manifest)?;
        Ok(Some(RestoredCapsule { manifest, payload }))
    }

    /// Finds the deepest published Capsule manifest without reading its payload.
    ///
    /// The returned manifest is authenticated against the exact prefix path.
    /// Callers that consume the payload must still verify its size and digest.
    ///
    /// # Errors
    ///
    /// Returns an error when a catalog entry is malformed or belongs to a
    /// different path.
    pub fn lookup_deepest(
        &self,
        path: &PrefixPath,
    ) -> Result<Option<(PrefixPath, CapsuleManifest)>, HoltCapsuleError> {
        let query_key = path.catalog_key_at(path.chunks().len())?;
        let Some(record) = self.capsules.longest_prefix_record(&query_key)? else {
            return Ok(None);
        };
        let chunk_count = path.chunk_count_for_catalog_key(&record.key)?;
        let candidate_path = path.prefix_at(chunk_count)?;
        let manifest: CapsuleManifest = serde_json::from_slice(&record.value)?;
        manifest.validate_for_path(&candidate_path)?;
        Ok(Some((candidate_path, manifest)))
    }

    /// Forces a Holt checkpoint after all previously acknowledged publishes.
    ///
    /// # Errors
    ///
    /// Returns an error when Holt cannot flush the checkpoint.
    pub fn checkpoint(&self) -> Result<(), HoltCapsuleError> {
        self.db.checkpoint()?;
        Ok(())
    }

    fn reconcile_existing(
        &self,
        path: &PrefixPath,
        manifest: &CapsuleManifest,
        payload: &[u8],
        existing: &[u8],
    ) -> Result<CapsulePublish, HoltCapsuleError> {
        if existing != serde_json::to_vec(manifest)? {
            return Err(HoltCapsuleError::PrefixConflict);
        }
        let persisted = self.read_payload(manifest.payload_digest)?;
        manifest.validate(path, &persisted)?;
        if persisted != payload {
            return Err(HoltCapsuleError::PrefixConflict);
        }
        Ok(CapsulePublish::AlreadyPresent)
    }

    fn restore_payload(
        &self,
        path: &PrefixPath,
        manifest: &CapsuleManifest,
    ) -> Result<Vec<u8>, HoltCapsuleError> {
        manifest.validate_for_path(path)?;
        let payload = self.read_payload(manifest.payload_digest)?;
        manifest.validate(path, &payload)?;
        Ok(payload)
    }

    fn write_payload(&self, digest: ContentDigest, payload: &[u8]) -> Result<(), HoltCapsuleError> {
        let path = self.object_path(digest);
        let directory = path
            .parent()
            .ok_or_else(|| io::Error::other("capsule object has no parent directory"))?;
        fs::create_dir_all(directory)?;
        sync_directory(&self.objects_dir)?;

        if path.exists() {
            return validate_payload_file(&path, digest, payload.len());
        }

        let temp = temporary_object_path(directory, digest);
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        file.write_all(payload)?;
        file.sync_all()?;
        drop(file);

        match fs::hard_link(&temp, &path) {
            Ok(()) => {
                fs::remove_file(&temp)?;
                sync_directory(directory)?;
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                fs::remove_file(&temp)?;
                validate_payload_file(&path, digest, payload.len())
            }
            Err(error) => {
                let _ = fs::remove_file(&temp);
                Err(error.into())
            }
        }
    }

    fn read_payload(&self, digest: ContentDigest) -> Result<Vec<u8>, HoltCapsuleError> {
        let path = self.object_path(digest);
        match fs::read(path) {
            Ok(payload) => Ok(payload),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Err(HoltCapsuleError::MissingPayload)
            }
            Err(error) => Err(error.into()),
        }
    }

    fn object_path(&self, digest: ContentDigest) -> PathBuf {
        let hex = digest.to_hex();
        self.objects_dir
            .join(&hex[..2])
            .join(format!("{}.capsule", &hex[2..]))
    }
}

fn encode_catalog_value<T: Serialize>(value: &T) -> Result<Vec<u8>, HoltCapsuleError> {
    let encoded = serde_json::to_vec(value)?;
    if encoded.len() > HOLT_VALUE_LIMIT {
        return Err(HoltCapsuleError::CatalogValueTooLarge);
    }
    Ok(encoded)
}

fn temporary_object_path(directory: &Path, digest: ContentDigest) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    directory.join(format!(
        ".{}.{}.{}.tmp",
        digest.to_hex(),
        std::process::id(),
        sequence
    ))
}

fn validate_payload_file(
    path: &Path,
    expected_digest: ContentDigest,
    expected_bytes: usize,
) -> Result<(), HoltCapsuleError> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let expected_bytes =
        u64::try_from(expected_bytes).map_err(|_| HoltCapsuleError::InvalidPayloadObject)?;
    if metadata.len() != expected_bytes {
        return Err(HoltCapsuleError::InvalidPayloadObject);
    }
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    if ContentDigest(hasher.finalize().into()) != expected_digest {
        return Err(HoltCapsuleError::InvalidPayloadObject);
    }
    Ok(())
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::{CapsuleComponent, CapsuleIdentity};

    fn identity(label: &[u8]) -> CapsuleIdentity {
        CapsuleIdentity {
            namespace: b"tenant-a".to_vec(),
            model_fingerprint: ContentDigest::sha256(label),
            tokenizer_fingerprint: ContentDigest::sha256(b"tokenizer"),
            adapter_fingerprint: ContentDigest::sha256(b"adapter"),
            state_plan_fingerprint: ContentDigest::sha256(b"state-plan"),
        }
    }

    fn capsule(
        identity: CapsuleIdentity,
        tokens: &[u32],
        payload: &[u8],
    ) -> (PrefixPath, CapsuleManifest) {
        let path = PrefixPath::from_token_ids(identity, 4, tokens).unwrap();
        let components = vec![CapsuleComponent {
            state_class: "local-kv".into(),
            offset_bytes: 0,
            length_bytes: u64::try_from(payload.len()).unwrap(),
            checksum: ContentDigest::sha256(payload),
        }];
        let manifest =
            CapsuleManifest::new(&path, path.token_count(), payload, components, 1).unwrap();
        (path, manifest)
    }

    #[test]
    fn publish_reopen_and_restore_deepest_prefix() {
        let directory = TempDir::new().unwrap();
        let short_payload = b"persistent-kv-state";
        let long_payload = b"deeper-persistent-kv-state";
        let (short_path, short_manifest) =
            capsule(identity(b"model"), &[1, 2, 3, 4], short_payload);
        let long_path =
            PrefixPath::from_token_ids(identity(b"model"), 4, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let (_, long_manifest) =
            capsule(identity(b"model"), &[1, 2, 3, 4, 5, 6, 7, 8], long_payload);

        {
            let store = HoltCapsuleStore::open(directory.path()).unwrap();
            assert_eq!(
                store
                    .publish(&short_path, &short_manifest, short_payload)
                    .unwrap(),
                CapsulePublish::Published
            );
            assert_eq!(
                store
                    .publish(&short_path, &short_manifest, short_payload)
                    .unwrap(),
                CapsulePublish::AlreadyPresent
            );
            assert_eq!(
                store
                    .publish(&long_path, &long_manifest, long_payload)
                    .unwrap(),
                CapsulePublish::Published
            );
        }

        let store = HoltCapsuleStore::open(directory.path()).unwrap();
        let restored = store.restore_deepest(&long_path).unwrap().unwrap();
        assert_eq!(restored.manifest, long_manifest);
        assert_eq!(restored.payload, long_payload);
    }

    #[test]
    fn payload_without_holt_publication_is_invisible() {
        let directory = TempDir::new().unwrap();
        let payload = b"orphan-payload";
        let (path, manifest) = capsule(identity(b"model"), &[1, 2, 3, 4], payload);
        let store = HoltCapsuleStore::open(directory.path()).unwrap();
        store
            .write_payload(manifest.payload_digest, payload)
            .unwrap();

        assert!(store.restore_deepest(&path).unwrap().is_none());
    }

    #[test]
    fn model_and_state_plan_identity_are_part_of_the_lookup_path() {
        let directory = TempDir::new().unwrap();
        let payload = b"model-specific-state";
        let (path, manifest) = capsule(identity(b"model-a"), &[1, 2, 3, 4], payload);
        let store = HoltCapsuleStore::open(directory.path()).unwrap();
        store.publish(&path, &manifest, payload).unwrap();

        let stale_model =
            PrefixPath::from_token_ids(identity(b"model-b"), 4, &[1, 2, 3, 4]).unwrap();
        assert!(store.restore_deepest(&stale_model).unwrap().is_none());

        let mut stale_plan_identity = identity(b"model-a");
        stale_plan_identity.state_plan_fingerprint = ContentDigest::sha256(b"stale-plan");
        let stale_plan = PrefixPath::from_token_ids(stale_plan_identity, 4, &[1, 2, 3, 4]).unwrap();
        assert!(store.restore_deepest(&stale_plan).unwrap().is_none());
    }

    #[test]
    fn partial_query_prefix_finds_last_complete_capsule_chunk() {
        let directory = TempDir::new().unwrap();
        let payload = b"aligned-state";
        let (published_path, manifest) = capsule(identity(b"model"), &[1, 2, 3, 4], payload);
        let query_path =
            PrefixPath::from_token_ids(identity(b"model"), 4, &[1, 2, 3, 4, 5, 6]).unwrap();
        let store = HoltCapsuleStore::open(directory.path()).unwrap();
        store.publish(&published_path, &manifest, payload).unwrap();

        let restored = store.restore_deepest(&query_path).unwrap().unwrap();
        assert_eq!(restored.manifest, manifest);
    }

    #[test]
    fn partial_chunk_publish_is_rejected() {
        let directory = TempDir::new().unwrap();
        let payload = b"partial-state";
        let (path, manifest) = capsule(identity(b"model"), &[1, 2, 3, 4, 5, 6], payload);
        let store = HoltCapsuleStore::open(directory.path()).unwrap();

        assert!(matches!(
            store.publish(&path, &manifest, payload),
            Err(HoltCapsuleError::UnalignedPublishBoundary)
        ));
        assert!(store.restore_deepest(&path).unwrap().is_none());
    }

    #[test]
    fn payload_larger_than_holt_value_limit_round_trips() {
        let directory = TempDir::new().unwrap();
        let payload = vec![0x5a; HOLT_VALUE_LIMIT + 1];
        let (path, manifest) = capsule(identity(b"model"), &[1, 2, 3, 4], &payload);
        let store = HoltCapsuleStore::open(directory.path()).unwrap();

        store.publish(&path, &manifest, &payload).unwrap();
        let restored = store.restore_deepest(&path).unwrap().unwrap();
        assert_eq!(restored.payload, payload);
    }

    #[test]
    fn corrupted_published_payload_fails_closed() {
        let directory = TempDir::new().unwrap();
        let payload = b"immutable-state";
        let (path, manifest) = capsule(identity(b"model"), &[1, 2, 3, 4], payload);
        let store = HoltCapsuleStore::open(directory.path()).unwrap();
        store.publish(&path, &manifest, payload).unwrap();

        fs::write(
            store.object_path(manifest.payload_digest),
            b"corrupted-state",
        )
        .unwrap();
        assert!(matches!(
            store.restore_deepest(&path),
            Err(HoltCapsuleError::Capsule(
                CapsuleError::PayloadDigestMismatch
            ))
        ));
    }

    #[test]
    fn conflicting_capsule_at_same_prefix_is_rejected() {
        let directory = TempDir::new().unwrap();
        let first_payload = b"first-state";
        let second_payload = b"second-state";
        let (path, first_manifest) = capsule(identity(b"model"), &[1, 2, 3, 4], first_payload);
        let (_, second_manifest) = capsule(identity(b"model"), &[1, 2, 3, 4], second_payload);
        let store = HoltCapsuleStore::open(directory.path()).unwrap();
        store
            .publish(&path, &first_manifest, first_payload)
            .unwrap();

        assert!(matches!(
            store.publish(&path, &second_manifest, second_payload),
            Err(HoltCapsuleError::PrefixConflict)
        ));
        let restored = store.restore_deepest(&path).unwrap().unwrap();
        assert_eq!(restored.payload, first_payload);
    }
}
