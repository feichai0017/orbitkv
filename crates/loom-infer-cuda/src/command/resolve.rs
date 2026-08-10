//! Checked conversion from opaque handles to disjoint device-region borrows.

use super::binding::{LeaseError, ResolveElement};
use super::{BindingElement, CommandError, CommandScope, Read, Write};
use crate::device_status::STATUS_PACKET_WORDS;
use crate::memory::{DeviceRegion, ReadWriteDeviceRegion};
use cuda_core::CudaStream;
use std::sync::Arc;

impl CommandScope<'_> {
    pub(super) fn resolve_device_status_source(
        &mut self,
        status: Read<i32>,
    ) -> Result<cuda_core::sys::CUdeviceptr, CommandError> {
        self.validate_resolve_request(&[status.set_id], &[status.slot])?;
        let bindings = self
            .bindings
            .as_ref()
            .expect("live command scope has bindings");
        let lease = &bindings.leases[status.slot];
        let region = <i32 as ResolveElement>::read(lease)
            .map_err(|error| map_lease_error(error, status.slot))?;
        if region.len() < STATUS_PACKET_WORDS {
            return Err(CommandError::DeviceStatusSourceTooShort);
        }
        Ok(region.cu_deviceptr())
    }

    pub(crate) fn resolve_rww<A, B, C>(
        &mut self,
        first: Read<A>,
        second: Write<B>,
        third: Write<C>,
    ) -> Result<ResolvedRww<'_, A, B, C>, CommandError>
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
            B::write(second_lease).map_err(|error| map_lease_error(error, second.slot))?;
        let third_buffer =
            C::write(third_lease).map_err(|error| map_lease_error(error, third.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
        })
    }

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

    pub(crate) fn resolve_rrrwww<A, B, C, D, E, F>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Write<D>,
        fifth: Write<E>,
        sixth: Write<F>,
    ) -> Result<ResolvedRrrwww<'_, A, B, C, D, E, F>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
        F: ResolveElement,
    {
        let slots = [
            first.slot,
            second.slot,
            third.slot,
            fourth.slot,
            fifth.slot,
            sixth.slot,
        ];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
                sixth.set_id,
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
            sixth_lease,
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
        let sixth_buffer =
            F::write(sixth_lease).map_err(|error| map_lease_error(error, sixth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrwww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
            sixth: sixth_buffer,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_rrrrw<A, B, C, D, E>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Read<D>,
        fifth: Write<E>,
    ) -> Result<ResolvedRrrrw<'_, A, B, C, D, E>, CommandError>
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
            D::read(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::write(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrrw {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_rrrrwww<A, B, C, D, E, F, G>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Read<D>,
        fifth: Write<E>,
        sixth: Write<F>,
        seventh: Write<G>,
    ) -> Result<ResolvedRrrrwww<'_, A, B, C, D, E, F, G>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
        F: ResolveElement,
        G: ResolveElement,
    {
        let slots = [
            first.slot,
            second.slot,
            third.slot,
            fourth.slot,
            fifth.slot,
            sixth.slot,
            seventh.slot,
        ];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
                sixth.set_id,
                seventh.set_id,
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
            sixth_lease,
            seventh_lease,
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
            D::read(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::write(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let sixth_buffer =
            F::write(sixth_lease).map_err(|error| map_lease_error(error, sixth.slot))?;
        let seventh_buffer =
            G::write(seventh_lease).map_err(|error| map_lease_error(error, seventh.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrrwww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
            sixth: sixth_buffer,
            seventh: seventh_buffer,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_rrrrrw<A, B, C, D, E, F>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Read<D>,
        fifth: Read<E>,
        sixth: Write<F>,
    ) -> Result<ResolvedRrrrrw<'_, A, B, C, D, E, F>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
        F: ResolveElement,
    {
        let slots = [
            first.slot,
            second.slot,
            third.slot,
            fourth.slot,
            fifth.slot,
            sixth.slot,
        ];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
                sixth.set_id,
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
            sixth_lease,
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
            D::read(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::read(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let sixth_buffer =
            F::write(sixth_lease).map_err(|error| map_lease_error(error, sixth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrrrw {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
            sixth: sixth_buffer,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_rrrrrww<A, B, C, D, E, F, G>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Read<D>,
        fifth: Read<E>,
        sixth: Write<F>,
        seventh: Write<G>,
    ) -> Result<ResolvedRrrrrww<'_, A, B, C, D, E, F, G>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
        F: ResolveElement,
        G: ResolveElement,
    {
        let slots = [
            first.slot,
            second.slot,
            third.slot,
            fourth.slot,
            fifth.slot,
            sixth.slot,
            seventh.slot,
        ];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
                sixth.set_id,
                seventh.set_id,
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
            sixth_lease,
            seventh_lease,
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
            D::read(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::read(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let sixth_buffer =
            F::write(sixth_lease).map_err(|error| map_lease_error(error, sixth.slot))?;
        let seventh_buffer =
            G::write(seventh_lease).map_err(|error| map_lease_error(error, seventh.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrrrww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
            sixth: sixth_buffer,
            seventh: seventh_buffer,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_rrrrrrw<A, B, C, D, E, F, G>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Read<D>,
        fifth: Read<E>,
        sixth: Read<F>,
        seventh: Write<G>,
    ) -> Result<ResolvedRrrrrrw<'_, A, B, C, D, E, F, G>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
        F: ResolveElement,
        G: ResolveElement,
    {
        let slots = [
            first.slot,
            second.slot,
            third.slot,
            fourth.slot,
            fifth.slot,
            sixth.slot,
            seventh.slot,
        ];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
                sixth.set_id,
                seventh.set_id,
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
            sixth_lease,
            seventh_lease,
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
            D::read(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::read(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let sixth_buffer =
            F::read(sixth_lease).map_err(|error| map_lease_error(error, sixth.slot))?;
        let seventh_buffer =
            G::write(seventh_lease).map_err(|error| map_lease_error(error, seventh.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrrrrw {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
            sixth: sixth_buffer,
            seventh: seventh_buffer,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_rrrrrrww<A, B, C, D, E, F, G, H>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Read<D>,
        fifth: Read<E>,
        sixth: Read<F>,
        seventh: Write<G>,
        eighth: Write<H>,
    ) -> Result<ResolvedRrrrrrww<'_, A, B, C, D, E, F, G, H>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
        F: ResolveElement,
        G: ResolveElement,
        H: ResolveElement,
    {
        let slots = [
            first.slot,
            second.slot,
            third.slot,
            fourth.slot,
            fifth.slot,
            sixth.slot,
            seventh.slot,
            eighth.slot,
        ];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
                sixth.set_id,
                seventh.set_id,
                eighth.set_id,
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
            sixth_lease,
            seventh_lease,
            eighth_lease,
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
            D::read(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::read(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let sixth_buffer =
            F::read(sixth_lease).map_err(|error| map_lease_error(error, sixth.slot))?;
        let seventh_buffer =
            G::write(seventh_lease).map_err(|error| map_lease_error(error, seventh.slot))?;
        let eighth_buffer =
            H::write(eighth_lease).map_err(|error| map_lease_error(error, eighth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrrrrww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
            sixth: sixth_buffer,
            seventh: seventh_buffer,
            eighth: eighth_buffer,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_rrrrrrrww<A, B, C, D, E, F, G, H, I>(
        &mut self,
        first: Read<A>,
        second: Read<B>,
        third: Read<C>,
        fourth: Read<D>,
        fifth: Read<E>,
        sixth: Read<F>,
        seventh: Read<G>,
        eighth: Write<H>,
        ninth: Write<I>,
    ) -> Result<ResolvedRrrrrrrww<'_, A, B, C, D, E, F, G, H, I>, CommandError>
    where
        A: ResolveElement,
        B: ResolveElement,
        C: ResolveElement,
        D: ResolveElement,
        E: ResolveElement,
        F: ResolveElement,
        G: ResolveElement,
        H: ResolveElement,
        I: ResolveElement,
    {
        let slots = [
            first.slot,
            second.slot,
            third.slot,
            fourth.slot,
            fifth.slot,
            sixth.slot,
            seventh.slot,
            eighth.slot,
            ninth.slot,
        ];
        self.validate_resolve_request(
            &[
                first.set_id,
                second.set_id,
                third.set_id,
                fourth.set_id,
                fifth.set_id,
                sixth.set_id,
                seventh.set_id,
                eighth.set_id,
                ninth.set_id,
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
            sixth_lease,
            seventh_lease,
            eighth_lease,
            ninth_lease,
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
            D::read(fourth_lease).map_err(|error| map_lease_error(error, fourth.slot))?;
        let fifth_buffer =
            E::read(fifth_lease).map_err(|error| map_lease_error(error, fifth.slot))?;
        let sixth_buffer =
            F::read(sixth_lease).map_err(|error| map_lease_error(error, sixth.slot))?;
        let seventh_buffer =
            G::read(seventh_lease).map_err(|error| map_lease_error(error, seventh.slot))?;
        let eighth_buffer =
            H::write(eighth_lease).map_err(|error| map_lease_error(error, eighth.slot))?;
        let ninth_buffer =
            I::write(ninth_lease).map_err(|error| map_lease_error(error, ninth.slot))?;
        let queue = self.queue.as_ref().expect("live command scope has a queue");

        Ok(ResolvedRrrrrrrww {
            stream: &queue.stream,
            first: first_buffer,
            second: second_buffer,
            third: third_buffer,
            fourth: fourth_buffer,
            fifth: fifth_buffer,
            sixth: sixth_buffer,
            seventh: seventh_buffer,
            eighth: eighth_buffer,
            ninth: ninth_buffer,
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

pub(crate) struct ResolvedRww<'scope, A: BindingElement, B: BindingElement, C: BindingElement> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope mut ReadWriteDeviceRegion<B>,
    pub(crate) third: &'scope mut ReadWriteDeviceRegion<C>,
}

pub(crate) struct ResolvedRrw<'scope, A: BindingElement, B: BindingElement, C: BindingElement> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope mut ReadWriteDeviceRegion<C>,
}

pub(crate) struct ResolvedRrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope mut ReadWriteDeviceRegion<C>,
    pub(crate) fourth: &'scope mut ReadWriteDeviceRegion<D>,
}

pub(crate) struct ResolvedRrrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope mut ReadWriteDeviceRegion<D>,
    pub(crate) fifth: &'scope mut ReadWriteDeviceRegion<E>,
}

pub(crate) struct ResolvedRrrwww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
    F: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope mut ReadWriteDeviceRegion<D>,
    pub(crate) fifth: &'scope mut ReadWriteDeviceRegion<E>,
    pub(crate) sixth: &'scope mut ReadWriteDeviceRegion<F>,
}

#[allow(clippy::type_complexity)]
pub(crate) struct ResolvedRrrrw<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope DeviceRegion<D>,
    pub(crate) fifth: &'scope mut ReadWriteDeviceRegion<E>,
}

#[allow(clippy::type_complexity)]
pub(crate) struct ResolvedRrrrwww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
    F: BindingElement,
    G: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope DeviceRegion<D>,
    pub(crate) fifth: &'scope mut ReadWriteDeviceRegion<E>,
    pub(crate) sixth: &'scope mut ReadWriteDeviceRegion<F>,
    pub(crate) seventh: &'scope mut ReadWriteDeviceRegion<G>,
}

#[allow(clippy::type_complexity)]
pub(crate) struct ResolvedRrrrrw<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
    F: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope DeviceRegion<D>,
    pub(crate) fifth: &'scope DeviceRegion<E>,
    pub(crate) sixth: &'scope mut ReadWriteDeviceRegion<F>,
}

#[allow(clippy::type_complexity)]
pub(crate) struct ResolvedRrrrrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
    F: BindingElement,
    G: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope DeviceRegion<D>,
    pub(crate) fifth: &'scope DeviceRegion<E>,
    pub(crate) sixth: &'scope mut ReadWriteDeviceRegion<F>,
    pub(crate) seventh: &'scope mut ReadWriteDeviceRegion<G>,
}

#[allow(clippy::type_complexity)]
pub(crate) struct ResolvedRrrrrrw<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
    F: BindingElement,
    G: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope DeviceRegion<D>,
    pub(crate) fifth: &'scope DeviceRegion<E>,
    pub(crate) sixth: &'scope DeviceRegion<F>,
    pub(crate) seventh: &'scope mut ReadWriteDeviceRegion<G>,
}

#[allow(clippy::type_complexity)]
pub(crate) struct ResolvedRrrrrrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
    F: BindingElement,
    G: BindingElement,
    H: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope DeviceRegion<D>,
    pub(crate) fifth: &'scope DeviceRegion<E>,
    pub(crate) sixth: &'scope DeviceRegion<F>,
    pub(crate) seventh: &'scope mut ReadWriteDeviceRegion<G>,
    pub(crate) eighth: &'scope mut ReadWriteDeviceRegion<H>,
}

#[allow(clippy::type_complexity)]
pub(crate) struct ResolvedRrrrrrrww<
    'scope,
    A: BindingElement,
    B: BindingElement,
    C: BindingElement,
    D: BindingElement,
    E: BindingElement,
    F: BindingElement,
    G: BindingElement,
    H: BindingElement,
    I: BindingElement,
> {
    pub(crate) stream: &'scope Arc<CudaStream>,
    pub(crate) first: &'scope DeviceRegion<A>,
    pub(crate) second: &'scope DeviceRegion<B>,
    pub(crate) third: &'scope DeviceRegion<C>,
    pub(crate) fourth: &'scope DeviceRegion<D>,
    pub(crate) fifth: &'scope DeviceRegion<E>,
    pub(crate) sixth: &'scope DeviceRegion<F>,
    pub(crate) seventh: &'scope DeviceRegion<G>,
    pub(crate) eighth: &'scope mut ReadWriteDeviceRegion<H>,
    pub(crate) ninth: &'scope mut ReadWriteDeviceRegion<I>,
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
