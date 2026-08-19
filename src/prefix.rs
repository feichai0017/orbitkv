use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::{
    CapsuleError, CapsuleManifest, ContentDigest, PhysicalStateBindingReceipt, PrefixPath,
};

const PREFIX_OBJECT_DOMAIN: &[u8] = b"orbitkv/prefix-object/v1\0";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PrefixObjectId(pub ContentDigest);

impl PrefixObjectId {
    /// Derives the logical object identity from a complete token-prefix path.
    ///
    /// # Errors
    ///
    /// Returns an error when the path cannot be encoded as a catalog key.
    pub fn from_path(path: &PrefixPath) -> Result<Self, PrefixError> {
        let mut material = Vec::from(PREFIX_OBJECT_DOMAIN);
        material.extend_from_slice(&path.catalog_key_at(path.chunks().len())?);
        Ok(Self(ContentDigest(Sha256::digest(material).into())))
    }

    #[must_use]
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct PrefixTokenRange {
    pub start: u64,
    pub end_exclusive: u64,
}

impl PrefixTokenRange {
    /// Creates one non-empty half-open token range.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or reversed range.
    pub const fn new(start: u64, end_exclusive: u64) -> Result<Self, PrefixError> {
        if start >= end_exclusive {
            return Err(PrefixError::InvalidComponentRange);
        }
        Ok(Self {
            start,
            end_exclusive,
        })
    }

    #[must_use]
    pub const fn token_count(self) -> u64 {
        self.end_exclusive - self.start
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct PrefixComponentSpec {
    pub state_class: String,
    pub token_range: PrefixTokenRange,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixComponentCompleteness {
    Missing,
    Complete,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum PrefixDeviceState {
    Absent,
    Resident { physical_binding_id: String },
    Tombstoned,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistentPrefixComponent {
    pub capsule_id: ContentDigest,
    pub payload_digest: ContentDigest,
    pub component_checksum: ContentDigest,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixAvailability {
    DeviceReady,
    Restorable,
    Incomplete,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixComponentSnapshot {
    pub spec: PrefixComponentSpec,
    pub device: PrefixDeviceState,
    pub device_completeness: PrefixComponentCompleteness,
    pub persistent: Option<PersistentPrefixComponent>,
    pub persistent_completeness: PrefixComponentCompleteness,
    pub lease_count: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixObjectSnapshot {
    pub schema: String,
    pub object_id: PrefixObjectId,
    pub prefix_token_count: u64,
    pub availability: PrefixAvailability,
    pub components: Vec<PrefixComponentSnapshot>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PrefixLeaseId(pub u64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrefixLease {
    pub lease_id: PrefixLeaseId,
    pub object_id: PrefixObjectId,
    pub state_classes: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct PrefixRuntimeStats {
    pub objects: u64,
    pub device_ready_objects: u64,
    pub restorable_objects: u64,
    pub incomplete_objects: u64,
    pub active_leases: u64,
    pub tombstoned_components: u64,
}

#[derive(Clone, Debug)]
struct PrefixComponent {
    spec: PrefixComponentSpec,
    device: PrefixDeviceState,
    persistent: Option<PersistentPrefixComponent>,
    leases: BTreeSet<PrefixLeaseId>,
}

#[derive(Clone, Debug)]
struct PrefixObject {
    prefix_token_count: u64,
    components: BTreeMap<String, PrefixComponent>,
}

#[derive(Clone, Debug)]
pub struct PrefixRuntime {
    objects: BTreeMap<PrefixObjectId, PrefixObject>,
    leases: BTreeMap<PrefixLeaseId, PrefixLease>,
    next_lease_id: u64,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PrefixError {
    #[error("prefix token count must be positive")]
    EmptyPrefix,
    #[error("prefix components must not be empty")]
    EmptyComponents,
    #[error("prefix component name must not be empty")]
    EmptyComponentName,
    #[error("prefix component token range is invalid")]
    InvalidComponentRange,
    #[error("prefix component {0:?} is duplicated")]
    DuplicateComponent(String),
    #[error("prefix object {0:?} is already declared with different geometry")]
    ConflictingObject(PrefixObjectId),
    #[error("prefix object {0:?} is unknown")]
    UnknownObject(PrefixObjectId),
    #[error("prefix component {state_class:?} is unknown for object {object_id:?}")]
    UnknownComponent {
        object_id: PrefixObjectId,
        state_class: String,
    },
    #[error("prefix component {state_class:?} is not device resident")]
    ComponentNotResident { state_class: String },
    #[error("prefix component {state_class:?} has no authenticated persistent copy")]
    ComponentNotRestorable { state_class: String },
    #[error("prefix component {state_class:?} is protected by {lease_count} lease(s)")]
    ComponentLeased {
        state_class: String,
        lease_count: u64,
    },
    #[error("prefix lease component set must not be empty")]
    EmptyLease,
    #[error("prefix lease component {0:?} is duplicated")]
    DuplicateLeaseComponent(String),
    #[error("prefix lease generation exhausted")]
    LeaseGenerationExhausted,
    #[error("prefix lease {0:?} is unknown")]
    UnknownLease(PrefixLeaseId),
    #[error("physical binding receipt does not match the prefix object")]
    MismatchedBindingReceipt,
    #[error("persistent Prefix snapshot is invalid")]
    InvalidPersistentSnapshot,
    #[error("Capsule component {0:?} has no exact token range")]
    MissingCapsuleComponentRange(String),
    #[error(transparent)]
    Capsule(#[from] CapsuleError),
}

impl Default for PrefixRuntime {
    fn default() -> Self {
        Self {
            objects: BTreeMap::new(),
            leases: BTreeMap::new(),
            next_lease_id: 1,
        }
    }
}

impl PrefixRuntime {
    /// Declares one logical Prefix object and its required state components.
    ///
    /// Repeating an identical declaration is idempotent. A declaration never
    /// implies that physical bytes are resident.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid component geometry or a conflicting
    /// declaration of the same object identity.
    pub fn declare(
        &mut self,
        object_id: PrefixObjectId,
        prefix_token_count: u64,
        mut components: Vec<PrefixComponentSpec>,
    ) -> Result<(), PrefixError> {
        validate_specs(prefix_token_count, &mut components)?;
        let geometry = components
            .iter()
            .map(|component| (component.state_class.clone(), component.clone()))
            .collect::<BTreeMap<_, _>>();
        if let Some(existing) = self.objects.get(&object_id) {
            let existing_geometry = existing
                .components
                .iter()
                .map(|(name, component)| (name.clone(), component.spec.clone()))
                .collect::<BTreeMap<_, _>>();
            if existing.prefix_token_count == prefix_token_count && existing_geometry == geometry {
                return Ok(());
            }
            return Err(PrefixError::ConflictingObject(object_id));
        }
        let components = geometry
            .into_iter()
            .map(|(name, spec)| {
                (
                    name,
                    PrefixComponent {
                        spec,
                        device: PrefixDeviceState::Absent,
                        persistent: None,
                        leases: BTreeSet::new(),
                    },
                )
            })
            .collect();
        self.objects.insert(
            object_id,
            PrefixObject {
                prefix_token_count,
                components,
            },
        );
        Ok(())
    }

    /// Registers an authenticated Holt Capsule as a complete persistent copy.
    ///
    /// Every Capsule component must carry an exact token range. The logical
    /// object is declared idempotently before its persistent copies are made
    /// visible.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid Capsule identity, duplicate components, or
    /// missing component ranges.
    pub fn register_capsule(
        &mut self,
        path: &PrefixPath,
        manifest: &CapsuleManifest,
    ) -> Result<PrefixObjectId, PrefixError> {
        manifest.validate_for_path(path)?;
        let object_id = PrefixObjectId::from_path(path)?;
        let mut specs = Vec::with_capacity(manifest.components.len());
        for component in &manifest.components {
            let (Some(start), Some(end_exclusive)) =
                (component.token_start, component.token_end_exclusive)
            else {
                return Err(PrefixError::MissingCapsuleComponentRange(
                    component.state_class.clone(),
                ));
            };
            specs.push(PrefixComponentSpec {
                state_class: component.state_class.clone(),
                token_range: PrefixTokenRange::new(start, end_exclusive)?,
            });
        }
        self.declare(object_id, manifest.prefix_token_count, specs)?;
        let object = self
            .objects
            .get_mut(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?;
        for component in &manifest.components {
            let state = object.components.get_mut(&component.state_class).ok_or(
                PrefixError::UnknownComponent {
                    object_id,
                    state_class: component.state_class.clone(),
                },
            )?;
            state.persistent = Some(PersistentPrefixComponent {
                capsule_id: manifest.capsule_id,
                payload_digest: manifest.payload_digest,
                component_checksum: component.checksum,
                offset_bytes: component.offset_bytes,
                length_bytes: component.length_bytes,
            });
        }
        Ok(object_id)
    }

    /// Imports a snapshot produced by an authenticated persistent store.
    ///
    /// Only persistent-only snapshots are accepted. Device residency and
    /// leases must be established by this runtime through binding and lease
    /// operations rather than trusted from an external snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot contains device state, leases,
    /// incomplete persistent components, or inconsistent availability.
    pub fn register_persistent_snapshot(
        &mut self,
        snapshot: &PrefixObjectSnapshot,
    ) -> Result<PrefixObjectId, PrefixError> {
        if snapshot.schema != "orbitkv.prefix-object-snapshot.v1"
            || snapshot.availability != PrefixAvailability::Restorable
            || snapshot.components.is_empty()
            || snapshot.components.iter().any(|component| {
                !matches!(component.device, PrefixDeviceState::Absent)
                    || component.device_completeness != PrefixComponentCompleteness::Missing
                    || component.persistent.is_none()
                    || component.persistent_completeness != PrefixComponentCompleteness::Complete
                    || component.lease_count != 0
            })
        {
            return Err(PrefixError::InvalidPersistentSnapshot);
        }
        self.declare(
            snapshot.object_id,
            snapshot.prefix_token_count,
            snapshot
                .components
                .iter()
                .map(|component| component.spec.clone())
                .collect(),
        )?;
        let object = self
            .objects
            .get_mut(&snapshot.object_id)
            .ok_or(PrefixError::UnknownObject(snapshot.object_id))?;
        for component in &snapshot.components {
            object
                .components
                .get_mut(&component.spec.state_class)
                .ok_or_else(|| PrefixError::UnknownComponent {
                    object_id: snapshot.object_id,
                    state_class: component.spec.state_class.clone(),
                })?
                .persistent
                .clone_from(&component.persistent);
        }
        Ok(snapshot.object_id)
    }

    /// Marks one exact component as device resident after physical commit.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown components or an empty binding identity.
    pub fn mark_device_resident(
        &mut self,
        object_id: PrefixObjectId,
        state_class: &str,
        physical_binding_id: impl Into<String>,
    ) -> Result<(), PrefixError> {
        let physical_binding_id = physical_binding_id.into();
        if physical_binding_id.is_empty() {
            return Err(PrefixError::MismatchedBindingReceipt);
        }
        self.component_mut(object_id, state_class)?.device = PrefixDeviceState::Resident {
            physical_binding_id,
        };
        Ok(())
    }

    /// Atomically publishes all device components proven by one binding
    /// receipt. No component is changed if preflight validation fails.
    ///
    /// # Errors
    ///
    /// Returns an error unless the receipt contains exactly every required
    /// component with matching ranges and ready payloads.
    pub fn commit_binding(
        &mut self,
        object_id: PrefixObjectId,
        receipt: &PhysicalStateBindingReceipt,
    ) -> Result<(), PrefixError> {
        self.validate_binding(object_id, receipt)?;
        let bindings = receipt
            .components
            .iter()
            .map(|component| {
                (
                    component.state_class.clone(),
                    component.physical_binding_id.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let object = self
            .objects
            .get_mut(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?;
        for (state_class, physical_binding_id) in bindings {
            object
                .components
                .get_mut(&state_class)
                .ok_or(PrefixError::MismatchedBindingReceipt)?
                .device = PrefixDeviceState::Resident {
                physical_binding_id,
            };
        }
        Ok(())
    }

    /// Validates a complete physical binding without changing Prefix state.
    ///
    /// # Errors
    ///
    /// Returns an error unless the receipt contains exactly every required
    /// component with matching ranges and ready payloads.
    pub fn validate_binding(
        &self,
        object_id: PrefixObjectId,
        receipt: &PhysicalStateBindingReceipt,
    ) -> Result<(), PrefixError> {
        let object = self
            .objects
            .get(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?;
        let mut bindings = BTreeMap::new();
        for component in &receipt.components {
            if !component.payload_ready
                || component.physical_binding_id.is_empty()
                || bindings
                    .insert(
                        component.state_class.clone(),
                        component.physical_binding_id.clone(),
                    )
                    .is_some()
            {
                return Err(PrefixError::MismatchedBindingReceipt);
            }
            let Some(expected) = object.components.get(&component.state_class) else {
                return Err(PrefixError::MismatchedBindingReceipt);
            };
            if component.token_start != expected.spec.token_range.start
                || component.token_end_exclusive != expected.spec.token_range.end_exclusive
                || component.physical_tokens != expected.spec.token_range.token_count()
            {
                return Err(PrefixError::MismatchedBindingReceipt);
            }
        }
        if bindings.len() != object.components.len() {
            return Err(PrefixError::MismatchedBindingReceipt);
        }
        for state_class in object.components.keys() {
            if !bindings.contains_key(state_class) {
                return Err(PrefixError::MismatchedBindingReceipt);
            }
        }
        Ok(())
    }

    /// Acquires a shared lease over selected device-resident components.
    ///
    /// # Errors
    ///
    /// Returns an error when any requested component is missing, duplicated,
    /// or not device resident. Lease acquisition is atomic.
    pub fn acquire(
        &mut self,
        object_id: PrefixObjectId,
        state_classes: &[String],
    ) -> Result<PrefixLease, PrefixError> {
        if state_classes.is_empty() {
            return Err(PrefixError::EmptyLease);
        }
        let requested = state_classes.iter().cloned().collect::<BTreeSet<_>>();
        if requested.len() != state_classes.len() {
            let duplicate = state_classes
                .iter()
                .find(|name| state_classes.iter().filter(|other| *other == *name).count() > 1)
                .cloned()
                .unwrap_or_default();
            return Err(PrefixError::DuplicateLeaseComponent(duplicate));
        }
        let object = self
            .objects
            .get(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?;
        for state_class in &requested {
            let component =
                object
                    .components
                    .get(state_class)
                    .ok_or(PrefixError::UnknownComponent {
                        object_id,
                        state_class: state_class.clone(),
                    })?;
            if !matches!(component.device, PrefixDeviceState::Resident { .. }) {
                return Err(PrefixError::ComponentNotResident {
                    state_class: state_class.clone(),
                });
            }
        }
        let lease_id = PrefixLeaseId(self.next_lease_id);
        self.next_lease_id = self
            .next_lease_id
            .checked_add(1)
            .ok_or(PrefixError::LeaseGenerationExhausted)?;
        let mut state_classes = requested.into_iter().collect::<Vec<_>>();
        state_classes.sort();
        let lease = PrefixLease {
            lease_id,
            object_id,
            state_classes,
        };
        let object = self
            .objects
            .get_mut(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?;
        for state_class in &lease.state_classes {
            object
                .components
                .get_mut(state_class)
                .ok_or(PrefixError::UnknownComponent {
                    object_id,
                    state_class: state_class.clone(),
                })?
                .leases
                .insert(lease_id);
        }
        self.leases.insert(lease_id, lease.clone());
        Ok(lease)
    }

    /// Releases one component early while retaining the lease's other state.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown lease or a component outside that lease.
    pub fn release_component(
        &mut self,
        lease_id: PrefixLeaseId,
        state_class: &str,
    ) -> Result<(), PrefixError> {
        let lease = self
            .leases
            .get_mut(&lease_id)
            .ok_or(PrefixError::UnknownLease(lease_id))?;
        let Some(index) = lease
            .state_classes
            .iter()
            .position(|name| name == state_class)
        else {
            return Err(PrefixError::UnknownComponent {
                object_id: lease.object_id,
                state_class: state_class.into(),
            });
        };
        let object_id = lease.object_id;
        lease.state_classes.remove(index);
        self.component_mut(object_id, state_class)?
            .leases
            .remove(&lease_id);
        Ok(())
    }

    /// Reattaches one resident component to an existing lease.
    ///
    /// This is the rollback operation for a component release performed before
    /// an engine-side physical transition. It never creates a new lease.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown lease, duplicate component, or a
    /// component that is no longer device resident.
    pub fn attach_component(
        &mut self,
        lease_id: PrefixLeaseId,
        state_class: &str,
    ) -> Result<(), PrefixError> {
        let lease = self
            .leases
            .get(&lease_id)
            .ok_or(PrefixError::UnknownLease(lease_id))?;
        if lease.state_classes.iter().any(|name| name == state_class) {
            return Err(PrefixError::DuplicateLeaseComponent(state_class.into()));
        }
        let object_id = lease.object_id;
        let component = self
            .objects
            .get(&object_id)
            .and_then(|object| object.components.get(state_class))
            .ok_or_else(|| PrefixError::UnknownComponent {
                object_id,
                state_class: state_class.into(),
            })?;
        if !matches!(component.device, PrefixDeviceState::Resident { .. }) {
            return Err(PrefixError::ComponentNotResident {
                state_class: state_class.into(),
            });
        }
        self.component_mut(object_id, state_class)?
            .leases
            .insert(lease_id);
        let lease = self
            .leases
            .get_mut(&lease_id)
            .ok_or(PrefixError::UnknownLease(lease_id))?;
        lease.state_classes.push(state_class.into());
        lease.state_classes.sort();
        Ok(())
    }

    /// Releases every component protected by a lease.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown lease.
    pub fn release(&mut self, lease_id: PrefixLeaseId) -> Result<(), PrefixError> {
        let lease = self
            .leases
            .remove(&lease_id)
            .ok_or(PrefixError::UnknownLease(lease_id))?;
        for state_class in lease.state_classes {
            self.component_mut(lease.object_id, &state_class)?
                .leases
                .remove(&lease_id);
        }
        Ok(())
    }

    /// Reclaims one device component while preserving its logical tombstone.
    ///
    /// # Errors
    ///
    /// Returns an error while any lease still protects the component.
    pub fn tombstone(
        &mut self,
        object_id: PrefixObjectId,
        state_class: &str,
    ) -> Result<(), PrefixError> {
        let component = self.component_mut(object_id, state_class)?;
        if !component.leases.is_empty() {
            return Err(PrefixError::ComponentLeased {
                state_class: state_class.into(),
                lease_count: component.leases.len() as u64,
            });
        }
        component.device = PrefixDeviceState::Tombstoned;
        Ok(())
    }

    /// Returns the persistent components needed to restore exact device state.
    ///
    /// # Errors
    ///
    /// Returns an error if any missing device component lacks an authenticated
    /// persistent copy.
    pub fn restore_plan(
        &self,
        object_id: PrefixObjectId,
    ) -> Result<Vec<PrefixComponentSpec>, PrefixError> {
        let object = self
            .objects
            .get(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?;
        let mut restore = Vec::new();
        for component in object.components.values() {
            if matches!(component.device, PrefixDeviceState::Resident { .. }) {
                continue;
            }
            if component.persistent.is_none() {
                return Err(PrefixError::ComponentNotRestorable {
                    state_class: component.spec.state_class.clone(),
                });
            }
            restore.push(component.spec.clone());
        }
        Ok(restore)
    }

    /// Returns a deterministic snapshot for validation and adapter protocols.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown object.
    pub fn snapshot(&self, object_id: PrefixObjectId) -> Result<PrefixObjectSnapshot, PrefixError> {
        let object = self
            .objects
            .get(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?;
        Ok(snapshot(object_id, object))
    }

    #[must_use]
    pub fn stats(&self) -> PrefixRuntimeStats {
        let mut stats = PrefixRuntimeStats {
            objects: self.objects.len() as u64,
            active_leases: self.leases.len() as u64,
            ..PrefixRuntimeStats::default()
        };
        for (&object_id, object) in &self.objects {
            match availability(object) {
                PrefixAvailability::DeviceReady => {
                    stats.device_ready_objects = stats.device_ready_objects.saturating_add(1);
                }
                PrefixAvailability::Restorable => {
                    stats.restorable_objects = stats.restorable_objects.saturating_add(1);
                }
                PrefixAvailability::Incomplete => {
                    stats.incomplete_objects = stats.incomplete_objects.saturating_add(1);
                }
            }
            stats.tombstoned_components = stats.tombstoned_components.saturating_add(
                snapshot(object_id, object)
                    .components
                    .iter()
                    .filter(|component| matches!(component.device, PrefixDeviceState::Tombstoned))
                    .count() as u64,
            );
        }
        stats
    }

    fn component_mut(
        &mut self,
        object_id: PrefixObjectId,
        state_class: &str,
    ) -> Result<&mut PrefixComponent, PrefixError> {
        self.objects
            .get_mut(&object_id)
            .ok_or(PrefixError::UnknownObject(object_id))?
            .components
            .get_mut(state_class)
            .ok_or_else(|| PrefixError::UnknownComponent {
                object_id,
                state_class: state_class.into(),
            })
    }
}

fn validate_specs(
    prefix_token_count: u64,
    components: &mut [PrefixComponentSpec],
) -> Result<(), PrefixError> {
    if prefix_token_count == 0 {
        return Err(PrefixError::EmptyPrefix);
    }
    if components.is_empty() {
        return Err(PrefixError::EmptyComponents);
    }
    components.sort();
    let mut names = BTreeSet::new();
    for component in components {
        if component.state_class.is_empty() {
            return Err(PrefixError::EmptyComponentName);
        }
        if component.token_range.start >= component.token_range.end_exclusive
            || component.token_range.end_exclusive > prefix_token_count
        {
            return Err(PrefixError::InvalidComponentRange);
        }
        if !names.insert(component.state_class.clone()) {
            return Err(PrefixError::DuplicateComponent(
                component.state_class.clone(),
            ));
        }
    }
    Ok(())
}

fn snapshot(object_id: PrefixObjectId, object: &PrefixObject) -> PrefixObjectSnapshot {
    PrefixObjectSnapshot {
        schema: "orbitkv.prefix-object-snapshot.v1".into(),
        object_id,
        prefix_token_count: object.prefix_token_count,
        availability: availability(object),
        components: object
            .components
            .values()
            .map(|component| PrefixComponentSnapshot {
                spec: component.spec.clone(),
                device: component.device.clone(),
                device_completeness: if matches!(
                    component.device,
                    PrefixDeviceState::Resident { .. }
                ) {
                    PrefixComponentCompleteness::Complete
                } else {
                    PrefixComponentCompleteness::Missing
                },
                persistent: component.persistent.clone(),
                persistent_completeness: if component.persistent.is_some() {
                    PrefixComponentCompleteness::Complete
                } else {
                    PrefixComponentCompleteness::Missing
                },
                lease_count: component.leases.len() as u64,
            })
            .collect(),
    }
}

fn availability(object: &PrefixObject) -> PrefixAvailability {
    if object
        .components
        .values()
        .all(|component| matches!(component.device, PrefixDeviceState::Resident { .. }))
    {
        return PrefixAvailability::DeviceReady;
    }
    if object.components.values().all(|component| {
        matches!(component.device, PrefixDeviceState::Resident { .. })
            || component.persistent.is_some()
    }) {
        return PrefixAvailability::Restorable;
    }
    PrefixAvailability::Incomplete
}

#[cfg(test)]
mod tests {
    use crate::{
        CapsuleComponentSpec, CapsuleIdentity, PhysicalStateBindingComponentReceipt,
        build_capsule_components,
    };

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

    fn capsule() -> (PrefixPath, CapsuleManifest) {
        let path = PrefixPath::from_token_ids(identity(), 4, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let payload = b"fullswa!";
        let components = build_capsule_components(
            payload,
            &[
                CapsuleComponentSpec {
                    state_class: "full-kv".into(),
                    length_bytes: 4,
                    token_start: Some(0),
                    token_end_exclusive: Some(8),
                },
                CapsuleComponentSpec {
                    state_class: "swa-kv".into(),
                    length_bytes: 4,
                    token_start: Some(4),
                    token_end_exclusive: Some(8),
                },
            ],
        )
        .unwrap();
        let manifest = CapsuleManifest::new(&path, 8, payload, components, 1).unwrap();
        (path, manifest)
    }

    fn binding_receipt(
        runtime: &PrefixRuntime,
        object_id: PrefixObjectId,
    ) -> PhysicalStateBindingReceipt {
        let snapshot = runtime.snapshot(object_id).unwrap();
        PhysicalStateBindingReceipt {
            schema: "orbitkv.physical-state-binding-receipt.v1".into(),
            plan_fingerprint: "sha256:test".into(),
            binding_id: 1,
            backend_transaction_id: "test-transaction".into(),
            components: snapshot
                .components
                .iter()
                .map(|component| PhysicalStateBindingComponentReceipt {
                    state_class: component.spec.state_class.clone(),
                    token_start: component.spec.token_range.start,
                    token_end_exclusive: component.spec.token_range.end_exclusive,
                    physical_tokens: component.spec.token_range.token_count(),
                    physical_binding_id: format!("{}-binding", component.spec.state_class),
                    payload_ready: true,
                })
                .collect(),
        }
    }

    #[test]
    fn capsule_declares_exact_restorable_components() {
        let (path, manifest) = capsule();
        let mut runtime = PrefixRuntime::default();
        let object_id = runtime.register_capsule(&path, &manifest).unwrap();
        let snapshot = runtime.snapshot(object_id).unwrap();

        assert_eq!(snapshot.availability, PrefixAvailability::Restorable);
        assert_eq!(snapshot.components.len(), 2);
        assert!(snapshot.components.iter().all(|component| {
            component.device_completeness == PrefixComponentCompleteness::Missing
                && component.persistent_completeness == PrefixComponentCompleteness::Complete
        }));
        assert_eq!(
            runtime.restore_plan(object_id).unwrap(),
            snapshot
                .components
                .iter()
                .map(|component| component.spec.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn full_can_remain_leased_while_swa_is_tombstoned() {
        let (path, manifest) = capsule();
        let mut runtime = PrefixRuntime::default();
        let object_id = runtime.register_capsule(&path, &manifest).unwrap();
        runtime
            .commit_binding(object_id, &binding_receipt(&runtime, object_id))
            .unwrap();
        let lease = runtime
            .acquire(object_id, &["full-kv".into(), "swa-kv".into()])
            .unwrap();

        runtime.release_component(lease.lease_id, "swa-kv").unwrap();
        runtime.tombstone(object_id, "swa-kv").unwrap();
        let snapshot = runtime.snapshot(object_id).unwrap();
        let full = snapshot
            .components
            .iter()
            .find(|component| component.spec.state_class == "full-kv")
            .unwrap();
        let swa = snapshot
            .components
            .iter()
            .find(|component| component.spec.state_class == "swa-kv")
            .unwrap();

        assert_eq!(full.lease_count, 1);
        assert!(matches!(full.device, PrefixDeviceState::Resident { .. }));
        assert_eq!(swa.lease_count, 0);
        assert_eq!(swa.device, PrefixDeviceState::Tombstoned);
        assert_eq!(snapshot.availability, PrefixAvailability::Restorable);
        assert_eq!(
            runtime.restore_plan(object_id).unwrap(),
            vec![swa.spec.clone()]
        );
        runtime.release(lease.lease_id).unwrap();
    }

    #[test]
    fn leased_component_cannot_be_tombstoned() {
        let (path, manifest) = capsule();
        let mut runtime = PrefixRuntime::default();
        let object_id = runtime.register_capsule(&path, &manifest).unwrap();
        runtime
            .commit_binding(object_id, &binding_receipt(&runtime, object_id))
            .unwrap();
        let lease = runtime
            .acquire(object_id, &["full-kv".into(), "swa-kv".into()])
            .unwrap();

        assert_eq!(
            runtime.tombstone(object_id, "swa-kv"),
            Err(PrefixError::ComponentLeased {
                state_class: "swa-kv".into(),
                lease_count: 1,
            })
        );
        assert_eq!(
            runtime.snapshot(object_id).unwrap().availability,
            PrefixAvailability::DeviceReady
        );
        runtime.release(lease.lease_id).unwrap();
    }

    #[test]
    fn binding_commit_is_atomic_across_components() {
        let (path, manifest) = capsule();
        let mut runtime = PrefixRuntime::default();
        let object_id = runtime.register_capsule(&path, &manifest).unwrap();
        let mut receipt = binding_receipt(&runtime, object_id);
        receipt.components[1].payload_ready = false;

        assert_eq!(
            runtime.commit_binding(object_id, &receipt),
            Err(PrefixError::MismatchedBindingReceipt)
        );
        let snapshot = runtime.snapshot(object_id).unwrap();
        assert!(
            snapshot
                .components
                .iter()
                .all(|component| { matches!(component.device, PrefixDeviceState::Absent) })
        );
    }

    #[test]
    fn missing_persistent_component_fails_closed() {
        let object_id = PrefixObjectId(ContentDigest::sha256(b"prefix"));
        let mut runtime = PrefixRuntime::default();
        runtime
            .declare(
                object_id,
                8,
                vec![
                    PrefixComponentSpec {
                        state_class: "full-kv".into(),
                        token_range: PrefixTokenRange::new(0, 8).unwrap(),
                    },
                    PrefixComponentSpec {
                        state_class: "swa-kv".into(),
                        token_range: PrefixTokenRange::new(4, 8).unwrap(),
                    },
                ],
            )
            .unwrap();
        runtime
            .mark_device_resident(object_id, "full-kv", "full-binding")
            .unwrap();

        assert_eq!(
            runtime.restore_plan(object_id),
            Err(PrefixError::ComponentNotRestorable {
                state_class: "swa-kv".into(),
            })
        );
        assert_eq!(
            runtime.snapshot(object_id).unwrap().availability,
            PrefixAvailability::Incomplete
        );
    }

    #[test]
    fn persistent_snapshot_does_not_import_device_or_lease_state() {
        let (path, manifest) = capsule();
        let mut source = PrefixRuntime::default();
        let object_id = source.register_capsule(&path, &manifest).unwrap();
        let persistent = source.snapshot(object_id).unwrap();
        let mut runtime = PrefixRuntime::default();
        runtime.register_persistent_snapshot(&persistent).unwrap();
        assert_eq!(runtime.snapshot(object_id).unwrap(), persistent);

        let mut invalid = persistent;
        invalid.components[0].device = PrefixDeviceState::Resident {
            physical_binding_id: "untrusted".into(),
        };
        invalid.components[0].device_completeness = PrefixComponentCompleteness::Complete;
        invalid.availability = PrefixAvailability::DeviceReady;
        assert_eq!(
            runtime.register_persistent_snapshot(&invalid),
            Err(PrefixError::InvalidPersistentSnapshot)
        );
    }

    #[test]
    fn persistent_registration_is_idempotent_after_device_binding() {
        let (path, manifest) = capsule();
        let mut source = PrefixRuntime::default();
        let object_id = source.register_capsule(&path, &manifest).unwrap();
        let persistent = source.snapshot(object_id).unwrap();
        let mut runtime = PrefixRuntime::default();
        runtime.register_persistent_snapshot(&persistent).unwrap();
        runtime
            .commit_binding(object_id, &binding_receipt(&runtime, object_id))
            .unwrap();
        let lease = runtime
            .acquire(object_id, &["full-kv".into(), "swa-kv".into()])
            .unwrap();

        runtime.register_persistent_snapshot(&persistent).unwrap();
        let snapshot = runtime.snapshot(object_id).unwrap();
        assert_eq!(snapshot.availability, PrefixAvailability::DeviceReady);
        assert!(snapshot.components.iter().all(|component| {
            component.device_completeness == PrefixComponentCompleteness::Complete
                && component.persistent_completeness == PrefixComponentCompleteness::Complete
                && component.lease_count == 1
        }));
        runtime.release(lease.lease_id).unwrap();
    }

    #[test]
    fn sole_component_release_can_be_rolled_back() {
        let object_id = PrefixObjectId(ContentDigest::sha256(b"pure-swa-prefix"));
        let mut runtime = PrefixRuntime::default();
        runtime
            .declare(
                object_id,
                8,
                vec![PrefixComponentSpec {
                    state_class: "swa".into(),
                    token_range: PrefixTokenRange::new(4, 8).unwrap(),
                }],
            )
            .unwrap();
        runtime
            .mark_device_resident(object_id, "swa", "swa-binding")
            .unwrap();
        let lease = runtime.acquire(object_id, &["swa".into()]).unwrap();

        runtime.release_component(lease.lease_id, "swa").unwrap();
        assert_eq!(
            runtime.snapshot(object_id).unwrap().components[0].lease_count,
            0
        );
        runtime.attach_component(lease.lease_id, "swa").unwrap();
        assert_eq!(
            runtime.snapshot(object_id).unwrap().components[0].lease_count,
            1
        );
        runtime.release(lease.lease_id).unwrap();
    }
}
