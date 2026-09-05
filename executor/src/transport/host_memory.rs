use std::{
    collections::{BTreeMap, VecDeque, btree_map::Entry},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use orbitkv::{ExternalReplica, ExternalReplicaDeletionEvidence, ExternalTransferCompletion};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use crate::{ExternalRestoreBatch, ExternalTransferBatch, KvComponent};

use super::{
    ExternalCopyChecksum, ExternalKvTransport, ExternalTransferOperation, ExternalTransferOutcome,
    ExternalTransportError, TransferObservation,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostFaultPoint {
    BeforeMutation,
    AfterMutation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostTransportFault {
    pub operation: ExternalTransferOperation,
    pub point: HostFaultPoint,
}

/// A shared byte region registered under one executor backend domain.
#[derive(Clone, Debug)]
pub struct HostTensorRegion {
    backend_domain: u16,
    layer: u32,
    component: KvComponent,
    base_offset: u64,
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl HostTensorRegion {
    #[must_use]
    pub fn new(
        backend_domain: u16,
        layer: u32,
        component: KvComponent,
        base_offset: u64,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            backend_domain,
            layer,
            component,
            base_offset,
            bytes: Arc::new(Mutex::new(bytes)),
        }
    }

    #[must_use]
    pub const fn backend_domain(&self) -> u16 {
        self.backend_domain
    }

    #[must_use]
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    #[must_use]
    pub const fn component(&self) -> KvComponent {
        self.component
    }

    #[must_use]
    pub const fn base_offset(&self) -> u64 {
        self.base_offset
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.lock().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.lock().is_empty()
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<u8> {
        self.bytes.lock().clone()
    }
}

#[derive(Clone, Debug)]
struct HostObject {
    base_offset: u64,
    bytes: Vec<u8>,
}

/// Deterministic reference transport backed by real host byte buffers.
///
/// It is intentionally simple rather than fast: exports stage a complete object
/// before publication, restores stage all reads before touching destinations,
/// and injected faults expose the same unobserved/ambiguous split required from
/// production transports.
#[derive(Debug)]
pub struct HostMemoryTransport {
    completion_domain: u64,
    next_completion: AtomicU64,
    regions: Mutex<BTreeMap<(u16, u32, KvComponent), HostTensorRegion>>,
    objects: Mutex<BTreeMap<(u64, u64), HostObject>>,
    faults: Mutex<VecDeque<HostTransportFault>>,
}

impl HostMemoryTransport {
    #[must_use]
    pub fn new(completion_domain: u64) -> Self {
        Self {
            completion_domain,
            next_completion: AtomicU64::new(1),
            regions: Mutex::new(BTreeMap::new()),
            objects: Mutex::new(BTreeMap::new()),
            faults: Mutex::new(VecDeque::new()),
        }
    }

    /// Registers one non-empty logical tensor-address range.
    ///
    /// # Errors
    ///
    /// Rejects zero domains, empty/overflowing regions, and duplicate domains.
    pub fn register_region(&self, region: HostTensorRegion) -> Result<(), ExternalTransportError> {
        if region.backend_domain == 0
            || region.is_empty()
            || region
                .base_offset
                .checked_add(region.len() as u64)
                .is_none()
        {
            return Err(error(
                ExternalTransferOperation::Register,
                TransferObservation::Unobserved,
                "invalid registered host region",
            ));
        }
        let mut regions = self.regions.lock();
        match regions.entry((region.backend_domain, region.layer, region.component)) {
            Entry::Vacant(entry) => {
                entry.insert(region);
            }
            Entry::Occupied(_) => {
                return Err(error(
                    ExternalTransferOperation::Register,
                    TransferObservation::Unobserved,
                    "tensor region is already registered",
                ));
            }
        }
        Ok(())
    }

    /// Queues a deterministic one-shot fault for the next matching operation.
    pub fn inject_fault(&self, fault: HostTransportFault) {
        self.faults.lock().push_back(fault);
    }

    #[must_use]
    pub fn object_bytes(&self, storage_domain: u64, object_index: u64) -> Option<Vec<u8>> {
        self.objects
            .lock()
            .get(&(storage_domain, object_index))
            .map(|object| object.bytes.clone())
    }

    fn completion(
        &self,
        transfer_id: orbitkv::ExternalTransferId,
    ) -> Result<ExternalTransferCompletion, ExternalTransportError> {
        let completion_value = self
            .next_completion
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                error(
                    ExternalTransferOperation::Complete,
                    TransferObservation::Ambiguous,
                    "transport completion identity exhausted",
                )
            })?;
        if self.completion_domain == 0 {
            return Err(error(
                ExternalTransferOperation::Complete,
                TransferObservation::Ambiguous,
                "transport completion identity exhausted",
            ));
        }
        Ok(ExternalTransferCompletion {
            transfer_id,
            completion_domain: self.completion_domain,
            completion_value,
            confirmed: true,
        })
    }

    fn take_fault(&self, operation: ExternalTransferOperation, point: HostFaultPoint) -> bool {
        let mut faults = self.faults.lock();
        let Some(position) = faults
            .iter()
            .position(|fault| fault.operation == operation && fault.point == point)
        else {
            return false;
        };
        faults.remove(position);
        true
    }
}

#[async_trait]
impl ExternalKvTransport for HostMemoryTransport {
    async fn export(
        &self,
        batch: &ExternalTransferBatch,
    ) -> Result<ExternalTransferOutcome, ExternalTransportError> {
        let (key, object) = stage_export(batch, &self.regions.lock())?;
        let checksums = checksums_from_export(batch, &object)?;
        if self.take_fault(
            ExternalTransferOperation::Export,
            HostFaultPoint::BeforeMutation,
        ) {
            return Err(error(
                ExternalTransferOperation::Export,
                TransferObservation::Unobserved,
                "injected before object publication",
            ));
        }
        let mut objects = self.objects.lock();
        if objects.contains_key(&key) {
            return Err(error(
                ExternalTransferOperation::Export,
                TransferObservation::Unobserved,
                "external object already exists",
            ));
        }
        objects.insert(key, object);
        drop(objects);
        if self.take_fault(
            ExternalTransferOperation::Export,
            HostFaultPoint::AfterMutation,
        ) {
            return Err(error(
                ExternalTransferOperation::Export,
                TransferObservation::Ambiguous,
                "injected after object publication",
            ));
        }
        ExternalTransferOutcome::confirmed(
            ExternalTransferOperation::Export,
            self.completion(batch.transfer_id)?,
            checksums,
        )
    }

    async fn restore(
        &self,
        batch: &ExternalRestoreBatch,
    ) -> Result<ExternalTransferOutcome, ExternalTransportError> {
        let staged = stage_restore(batch, &self.objects.lock())?;
        let checksums = checksums_from_restore(batch, &staged)?;
        validate_restore_checksums(batch, &checksums)?;
        if self.take_fault(
            ExternalTransferOperation::Restore,
            HostFaultPoint::BeforeMutation,
        ) {
            return Err(error(
                ExternalTransferOperation::Restore,
                TransferObservation::Unobserved,
                "injected before destination writes",
            ));
        }
        commit_restore(batch, &staged, &self.regions.lock())?;
        if self.take_fault(
            ExternalTransferOperation::Restore,
            HostFaultPoint::AfterMutation,
        ) {
            return Err(error(
                ExternalTransferOperation::Restore,
                TransferObservation::Ambiguous,
                "injected after destination writes",
            ));
        }
        ExternalTransferOutcome::confirmed(
            ExternalTransferOperation::Restore,
            self.completion(batch.transfer_id)?,
            checksums,
        )
    }

    async fn delete(
        &self,
        replica: &ExternalReplica,
    ) -> Result<ExternalReplicaDeletionEvidence, ExternalTransportError> {
        let key = (replica.target.storage_domain, replica.target.object_index);
        {
            let objects = self.objects.lock();
            let object = objects.get(&key).ok_or_else(|| {
                error(
                    ExternalTransferOperation::Delete,
                    TransferObservation::Unobserved,
                    "external object is unknown",
                )
            })?;
            if object.base_offset != replica.target.base_offset
                || object.bytes.len() as u64 != replica.total_bytes
            {
                return Err(error(
                    ExternalTransferOperation::Delete,
                    TransferObservation::Unobserved,
                    "external object geometry differs from replica",
                ));
            }
        }
        if self.take_fault(
            ExternalTransferOperation::Delete,
            HostFaultPoint::BeforeMutation,
        ) {
            return Err(error(
                ExternalTransferOperation::Delete,
                TransferObservation::Unobserved,
                "injected before object deletion",
            ));
        }
        self.objects.lock().remove(&key);
        if self.take_fault(
            ExternalTransferOperation::Delete,
            HostFaultPoint::AfterMutation,
        ) {
            return Err(error(
                ExternalTransferOperation::Delete,
                TransferObservation::Ambiguous,
                "injected after object deletion",
            ));
        }
        Ok(ExternalReplicaDeletionEvidence {
            key: replica.key,
            target: replica.target,
            deleted: true,
        })
    }
}

fn stage_export(
    batch: &ExternalTransferBatch,
    regions: &BTreeMap<(u16, u32, KvComponent), HostTensorRegion>,
) -> Result<((u64, u64), HostObject), ExternalTransportError> {
    let first = batch
        .spans
        .first()
        .ok_or_else(|| invalid(ExternalTransferOperation::Export))?;
    let key = (
        first.destination_storage_domain,
        first.destination_object_index,
    );
    let base_offset = first.destination_offset;
    let total = usize::try_from(batch.total_bytes)
        .map_err(|_| invalid(ExternalTransferOperation::Export))?;
    let mut object = vec![0_u8; total];
    let mut expected_offset = base_offset;
    for span in &batch.spans {
        if (
            span.destination_storage_domain,
            span.destination_object_index,
        ) != key
            || span.destination_offset != expected_offset
        {
            return Err(invalid(ExternalTransferOperation::Export));
        }
        let source = region_slice(
            regions,
            span.source_backend_domain,
            span.layer,
            span.component,
            span.source_tensor_offset,
            span.bytes,
            ExternalTransferOperation::Export,
        )?;
        let begin = offset_in_object(
            span.destination_offset,
            base_offset,
            span.bytes,
            total,
            ExternalTransferOperation::Export,
        )?;
        object[begin..begin + source.len()].copy_from_slice(&source);
        expected_offset = expected_offset
            .checked_add(span.bytes)
            .ok_or_else(|| invalid(ExternalTransferOperation::Export))?;
    }
    if expected_offset.checked_sub(base_offset) != Some(batch.total_bytes) {
        return Err(invalid(ExternalTransferOperation::Export));
    }
    Ok((
        key,
        HostObject {
            base_offset,
            bytes: object,
        },
    ))
}

fn stage_restore(
    batch: &ExternalRestoreBatch,
    objects: &BTreeMap<(u64, u64), HostObject>,
) -> Result<Vec<Vec<u8>>, ExternalTransportError> {
    let mut staged = Vec::with_capacity(batch.spans.len());
    for span in &batch.spans {
        let object = objects
            .get(&(span.source_storage_domain, span.source_object_index))
            .ok_or_else(|| invalid(ExternalTransferOperation::Restore))?;
        let begin = offset_in_object(
            span.source_offset,
            object.base_offset,
            span.bytes,
            object.bytes.len(),
            ExternalTransferOperation::Restore,
        )?;
        staged.push(
            object.bytes[begin
                ..begin
                    + usize::try_from(span.bytes)
                        .map_err(|_| invalid(ExternalTransferOperation::Restore))?]
                .to_vec(),
        );
    }
    if staged.iter().map(Vec::len).sum::<usize>()
        != usize::try_from(batch.total_bytes)
            .map_err(|_| invalid(ExternalTransferOperation::Restore))?
    {
        return Err(invalid(ExternalTransferOperation::Restore));
    }
    Ok(staged)
}

fn commit_restore(
    batch: &ExternalRestoreBatch,
    staged: &[Vec<u8>],
    regions: &BTreeMap<(u16, u32, KvComponent), HostTensorRegion>,
) -> Result<(), ExternalTransportError> {
    for span in &batch.spans {
        validate_region_write(
            regions,
            span.destination_backend_domain,
            span.layer,
            span.component,
            span.destination_tensor_offset,
            span.bytes,
        )?;
    }
    for (span, bytes) in batch.spans.iter().zip(staged) {
        write_region(
            regions,
            span.destination_backend_domain,
            span.layer,
            span.component,
            span.destination_tensor_offset,
            bytes,
        )?;
    }
    Ok(())
}

fn validate_restore_checksums(
    batch: &ExternalRestoreBatch,
    checksums: &[ExternalCopyChecksum],
) -> Result<(), ExternalTransportError> {
    for checksum in checksums {
        let mut spans = batch
            .spans
            .iter()
            .filter(|span| span.copy_index == checksum.copy_index);
        let expected = spans
            .next()
            .map(|span| span.expected_checksum)
            .ok_or_else(|| invalid(ExternalTransferOperation::Restore))?;
        if checksum.checksum != expected || spans.any(|span| span.expected_checksum != expected) {
            return Err(error(
                ExternalTransferOperation::Restore,
                TransferObservation::Unobserved,
                "external object checksum mismatch",
            ));
        }
    }
    Ok(())
}

fn checksums_from_export(
    batch: &ExternalTransferBatch,
    object: &HostObject,
) -> Result<Box<[ExternalCopyChecksum]>, ExternalTransportError> {
    checksums_by_copy(
        &batch.spans,
        |span| span.copy_index,
        |span| {
            let begin = offset_in_object(
                span.destination_offset,
                object.base_offset,
                span.bytes,
                object.bytes.len(),
                ExternalTransferOperation::Export,
            )?;
            Ok(object.bytes[begin
                ..begin
                    + usize::try_from(span.bytes)
                        .map_err(|_| invalid(ExternalTransferOperation::Export))?]
                .to_vec())
        },
        ExternalTransferOperation::Export,
    )
}

fn checksums_from_restore(
    batch: &ExternalRestoreBatch,
    staged: &[Vec<u8>],
) -> Result<Box<[ExternalCopyChecksum]>, ExternalTransportError> {
    if batch.spans.len() != staged.len() {
        return Err(invalid(ExternalTransferOperation::Restore));
    }
    checksums_by_copy(
        &batch.spans.iter().zip(staged).collect::<Vec<_>>(),
        |(span, _)| span.copy_index,
        |(_, bytes)| Ok((*bytes).clone()),
        ExternalTransferOperation::Restore,
    )
}

fn checksums_by_copy<T, I, B>(
    spans: &[T],
    index: I,
    bytes: B,
    operation: ExternalTransferOperation,
) -> Result<Box<[ExternalCopyChecksum]>, ExternalTransportError>
where
    I: Fn(&T) -> u32,
    B: Fn(&T) -> Result<Vec<u8>, ExternalTransportError>,
{
    let mut output = Vec::new();
    let mut current = None;
    let mut hasher = Sha256::new();
    for span in spans {
        let copy_index = index(span);
        if current != Some(copy_index) {
            if let Some(previous) = current {
                output.push(ExternalCopyChecksum {
                    copy_index: previous,
                    checksum: hasher.finalize_reset().into(),
                });
            }
            if usize::try_from(copy_index).ok() != Some(output.len()) {
                return Err(invalid(operation));
            }
            current = Some(copy_index);
        }
        hasher.update(bytes(span)?);
    }
    if let Some(copy_index) = current {
        output.push(ExternalCopyChecksum {
            copy_index,
            checksum: hasher.finalize().into(),
        });
    }
    if output.is_empty() {
        return Err(invalid(operation));
    }
    Ok(output.into_boxed_slice())
}

fn region_slice(
    regions: &BTreeMap<(u16, u32, KvComponent), HostTensorRegion>,
    domain: u16,
    layer: u32,
    component: KvComponent,
    offset: u64,
    bytes: u64,
    operation: ExternalTransferOperation,
) -> Result<Vec<u8>, ExternalTransportError> {
    let region = regions
        .get(&(domain, layer, component))
        .ok_or_else(|| invalid(operation))?;
    let data = region.bytes.lock();
    let begin = offset_in_object(offset, region.base_offset, bytes, data.len(), operation)?;
    let count = usize::try_from(bytes).map_err(|_| invalid(operation))?;
    Ok(data[begin..begin + count].to_vec())
}

fn write_region(
    regions: &BTreeMap<(u16, u32, KvComponent), HostTensorRegion>,
    domain: u16,
    layer: u32,
    component: KvComponent,
    offset: u64,
    bytes: &[u8],
) -> Result<(), ExternalTransportError> {
    let region = regions
        .get(&(domain, layer, component))
        .ok_or_else(|| invalid(ExternalTransferOperation::Restore))?;
    let mut data = region.bytes.lock();
    let count =
        u64::try_from(bytes.len()).map_err(|_| invalid(ExternalTransferOperation::Restore))?;
    let begin = offset_in_object(
        offset,
        region.base_offset,
        count,
        data.len(),
        ExternalTransferOperation::Restore,
    )?;
    data[begin..begin + bytes.len()].copy_from_slice(bytes);
    Ok(())
}

fn validate_region_write(
    regions: &BTreeMap<(u16, u32, KvComponent), HostTensorRegion>,
    domain: u16,
    layer: u32,
    component: KvComponent,
    offset: u64,
    bytes: u64,
) -> Result<(), ExternalTransportError> {
    let region = regions
        .get(&(domain, layer, component))
        .ok_or_else(|| invalid(ExternalTransferOperation::Restore))?;
    offset_in_object(
        offset,
        region.base_offset,
        bytes,
        region.bytes.lock().len(),
        ExternalTransferOperation::Restore,
    )?;
    Ok(())
}

fn offset_in_object(
    offset: u64,
    base_offset: u64,
    bytes: u64,
    total: usize,
    operation: ExternalTransferOperation,
) -> Result<usize, ExternalTransportError> {
    let begin = offset
        .checked_sub(base_offset)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid(operation))?;
    let count = usize::try_from(bytes).map_err(|_| invalid(operation))?;
    if begin.checked_add(count).is_none_or(|end| end > total) {
        return Err(invalid(operation));
    }
    Ok(begin)
}

fn invalid(operation: ExternalTransferOperation) -> ExternalTransportError {
    error(
        operation,
        TransferObservation::Unobserved,
        "transfer geometry is invalid or unregistered",
    )
}

fn error(
    operation: ExternalTransferOperation,
    observation: TransferObservation,
    reason: &'static str,
) -> ExternalTransportError {
    match observation {
        TransferObservation::Unobserved => ExternalTransportError::unobserved(operation, reason),
        TransferObservation::Ambiguous => ExternalTransportError::ambiguous(operation, reason),
    }
}
