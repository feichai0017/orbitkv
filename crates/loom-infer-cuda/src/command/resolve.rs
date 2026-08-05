//! Checked conversion from opaque handles to disjoint device-buffer borrows.

use super::binding::{LeaseError, ResolveElement};
use super::{BindingElement, CommandError, CommandScope, Read, Write};
use cuda_core::{CudaStream, DeviceBuffer};

impl CommandScope<'_> {
    pub(crate) fn resolve_rrw<A, B, C>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Write<C>,
    ) -> Result<ResolvedRrw<'_, A, B, C>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
    {
        self.validate_resolve_request(
            &[first.set_id, second.set_id, third.set_id],
            &[first.slot, second.slot, third.slot],
        )?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        let [first_lease, second_lease, third_lease] = bindings
            .leases
            .get_disjoint_mut([first.slot, second.slot, third.slot])
            .expect("validated binding slots are pairwise disjoint");
        let first_buffer =
            A::read(first_lease).map_err(|error| map_lease_error(error, first.slot))?;
        let second_buffer =
            B::read(second_lease).map_err(|error| map_lease_error(error, second.slot))?;
        let third_buffer =
            C::write(third_lease).map_err(|error| map_lease_error(error, third.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrw {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
        })
    }

    pub(crate) fn resolve_rrww<A, B, C, D>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Write<C>,
        fourth: Write<D>,
    ) -> Result<ResolvedRrww<'_, A, B, C, D>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
    {
        let slots = [first.slot, second.slot, third.slot, fourth.slot];
        self.validate_resolve_request(
            &[first.set_id, second.set_id, third.set_id, fourth.set_id],
            &slots,
        )?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        let [first_lease, second_lease, third_lease, fourth_lease] = bindings
            .leases
            .get_disjoint_mut(slots)
            .expect("validated binding slots are pairwise disjoint");
        let first_buffer =
            A::read(first_lease).map_err(|error| map_lease_error(error, first.slot))?;
        let second_buffer =
            B::read(second_lease).map_err(|error| map_lease_error(error, second.slot))?;
        let third_buffer =
            C::write(third_lease).map_err(|error| map_lease_error(error, third.slot))?;
        let fourth_buffer =
            D::write(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
        })
    }

    pub(crate) fn resolve_rrrww<A, B, C, D, E>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Write<D>,
        fifth: Write<E>,
    ) -> Result<ResolvedRrrww<'_, A, B, C, D, E>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
    {
        let slots = [first.slot, second.slot, third.slot, fourth.slot, fifth.slot];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
            ],
            &slots,
        )?;
        let bindings = self
            .bindings
            .as_mut()
            .expect("live command scope has bindings");
        let [
            first_lease,
            second_lease,
            third_lease,
            fourth_lease,
            fifth_lease,
        ] = bindings
            .leases
            .get_disjoint_mut(slots)
            .expect("validated binding slots are pairwise disjoint");
        let first_buffer =
            A::read(first_lease).map_err(|error| map_lease_error(error, first.slot))?;
        let second_buffer =
            B::read(second_lease).map_err(|error| map_lease_error(error, second.slot))?;
        let third_buffer =
            C::read(third_lease).map_err(|error| map_lease_error(error, third.slot))?;
        let fourth_buffer =
            D::write(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::write(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
        })
    }

    fn validate_resolve_request(
        &self,
        set_ids: &[u64],
        slots: &[usize],
    ) -> Result<(), CommandError> {
        let bindings = self
            .bindings
            .as_ref()
            .expect("live command scope has bindings");
        validate_binding_request(bindings.set_id, bindings.leases.len(), set_ids, slots)
    }
}

fn validate_binding_request(
    expected_set_id: u64,
    binding_count: usize,
    set_ids: &[u64],
    slots: &[usize],
) -> Result<(), CommandError> {
    if set_ids.iter().any(|&set_id| set_id != expected_set_id) {
        return Err(CommandError::BindingSetMismatch);
    }
    for (index, &slot) in slots.iter().enumerate() {
        if slots[..index].contains(&slot) {
            return Err(CommandError::DuplicateBindingSlot);
        }
        if slot >= binding_count {
            return Err(CommandError::BindingSlotOutOfRange {
                slot,
                bindings: binding_count,
            });
        }
    }
    Ok(())
}

fn map_lease_error(error: LeaseError, slot: usize) -> CommandError {
    match error {
        LeaseError::ElementMismatch => CommandError::BindingTypeMismatch { slot },
        LeaseError::ReadOnly => CommandError::BindingIsReadOnly { slot },
        LeaseError::Vacant => CommandError::BindingSlotVacant { slot },
    }
}

pub(crate) struct ResolvedRrw<'scope, A: BindingElement, B: BindingElement, C: BindingElement> {
    pub(crate) stream: &'scope CudaStream,
    pub(crate) first: &'scope DeviceBuffer<A>,
    pub(crate) second: &'scope DeviceBuffer<B>,
    pub(crate) third: &'scope mut DeviceBuffer<C>,
}

pub(crate) struct ResolvedRrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
> {
    pub(crate) stream: &'scope CudaStream,
    pub(crate) first: &'scope DeviceBuffer<A>,
    pub(crate) second: &'scope DeviceBuffer<B>,
    pub(crate) third: &'scope mut DeviceBuffer<C>,
    pub(crate) fourth: &'scope mut DeviceBuffer<D>,
}

pub(crate) struct ResolvedRrrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
> {
    pub(crate) stream: &'scope CudaStream,
    pub(crate) first: &'scope DeviceBuffer<A>,
    pub(crate) second: &'scope DeviceBuffer<B>,
    pub(crate) third: &'scope DeviceBuffer<C>,
    pub(crate) fourth: &'scope mut DeviceBuffer<D>,
    pub(crate) fifth: &'scope mut DeviceBuffer<E>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_request_accepts_matching_distinct_slots() {
        assert!(validate_binding_request(7, 3, &[7, 7, 7], &[0, 1, 2]).is_ok());
    }

    #[test]
    fn binding_request_rejects_a_handle_from_another_set() {
        assert!(matches!(
            validate_binding_request(7, 3, &[7, 8, 7], &[0, 1, 2]),
            Err(CommandError::BindingSetMismatch)
        ));
    }

    #[test]
    fn binding_request_rejects_duplicate_slots_before_resolution() {
        assert!(matches!(
            validate_binding_request(7, 3, &[7, 7, 7], &[0, 1, 1]),
            Err(CommandError::DuplicateBindingSlot)
        ));
    }

    #[test]
    fn binding_request_reports_the_first_out_of_range_slot() {
        assert!(matches!(
            validate_binding_request(7, 3, &[7, 7], &[0, 3]),
            Err(CommandError::BindingSlotOutOfRange {
                slot: 3,
                bindings: 3,
            })
        ));
    }
}
