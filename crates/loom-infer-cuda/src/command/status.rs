//! Retained host storage and decoder registrations for device status packets.

use super::CommandError;
use crate::device_status::{DeviceStatusDecoder, DeviceStatusProtocolError, STATUS_PACKET_WORDS};
use cuda_core::sys::CUdeviceptr;
use cuda_core::{CudaContext, PinnedHostBuffer};
use loom_infer::ContractError;
use std::sync::Arc;

pub(crate) struct DeviceStatusState {
    host: PinnedHostBuffer<i32>,
    pending: Vec<PendingDeviceStatus>,
    capacity: usize,
}

impl DeviceStatusState {
    pub(super) fn new(context: &Arc<CudaContext>, capacity: usize) -> Result<Self, CommandError> {
        let words = capacity
            .checked_mul(STATUS_PACKET_WORDS)
            .ok_or(CommandError::DeviceStatusCapacityOverflow)?;
        Ok(Self {
            host: PinnedHostBuffer::zeroed(context, words)?,
            pending: Vec::with_capacity(capacity),
            capacity,
        })
    }

    pub(super) fn reserve(
        &mut self,
        source: CUdeviceptr,
        decoder: DeviceStatusDecoder,
    ) -> Result<usize, CommandError> {
        if self.pending.len() == self.capacity {
            return Err(CommandError::DeviceStatusCapacityExceeded {
                capacity: self.capacity,
            });
        }
        if self.pending.iter().any(|pending| pending.source == source) {
            return Err(CommandError::DuplicateDeviceStatusSource);
        }
        let index = self.pending.len();
        self.pending.push(PendingDeviceStatus { source, decoder });
        Ok(index)
    }

    pub(super) fn cancel_last(&mut self, index: usize) {
        assert_eq!(
            index + 1,
            self.pending.len(),
            "only the latest device status reservation can be cancelled"
        );
        self.pending.pop();
    }

    pub(super) fn len(&self) -> usize {
        self.pending.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub(super) fn pending(&self, index: usize) -> PendingDeviceStatus {
        self.pending[index]
    }

    pub(super) fn host_mut(&mut self) -> &mut PinnedHostBuffer<i32> {
        &mut self.host
    }

    pub(crate) fn decode(&self) -> Result<Option<ContractError>, DeviceStatusProtocolError> {
        decode_status_packets(&self.pending, &self.host)
    }

    pub(crate) fn clear(&mut self) {
        self.pending.clear();
    }
}

#[derive(Clone, Copy)]
pub(super) struct PendingDeviceStatus {
    source: CUdeviceptr,
    decoder: DeviceStatusDecoder,
}

impl PendingDeviceStatus {
    pub(super) const fn source(self) -> CUdeviceptr {
        self.source
    }
}

fn decode_status_packets(
    pending: &[PendingDeviceStatus],
    packets: &[i32],
) -> Result<Option<ContractError>, DeviceStatusProtocolError> {
    let mut first_rejection = None;
    for (index, pending) in pending.iter().enumerate() {
        let start = index * STATUS_PACKET_WORDS;
        let end = start + STATUS_PACKET_WORDS;
        if let Some(error) = pending.decoder.decode(&packets[start..end])? {
            first_rejection.get_or_insert(error);
        }
    }
    Ok(first_rejection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_status::{AppendMapKind, STATUS_NON_EXCLUSIVE_APPEND_TARGET, STATUS_SUCCESS};

    #[test]
    fn protocol_failure_after_rejection_takes_priority() {
        let decoder = DeviceStatusDecoder::paged_append(AppendMapKind::Requests, 2, 2, 4, 2, 16);
        let pending = [
            PendingDeviceStatus { source: 0, decoder },
            PendingDeviceStatus { source: 1, decoder },
        ];
        let packets = [
            STATUS_NON_EXCLUSIVE_APPEND_TARGET,
            1,
            2,
            0,
            0,
            STATUS_SUCCESS,
            1,
            0,
            0,
            0,
        ];
        assert!(decode_status_packets(&pending, &packets).is_err());
    }
}
