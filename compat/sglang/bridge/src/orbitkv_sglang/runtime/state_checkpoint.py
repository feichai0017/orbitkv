from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StatePoolConfig:
    engine_epoch: int
    pool_epoch: int
    byte_count: int
    pool_id: int
    slot_count: int


@dataclass(frozen=True, slots=True)
class StateSlotLease:
    engine_epoch: int
    pool_epoch: int
    generation: int
    slot_id: int
    pool_id: int


@dataclass(frozen=True, slots=True)
class StateTransitionLease:
    engine_epoch: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class StateRetirementLease:
    engine_epoch: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class StatePoolIdentity:
    engine_epoch: int
    pool_epoch: int
    byte_count: int
    pool_id: int
    slot_count: int


@dataclass(frozen=True, slots=True)
class StatePoolStats:
    identity: StatePoolIdentity
    free_slots: int
    reserved_slots: int
    relocating_slots: int
    live_slots: int
    retiring_slots: int
    quarantined_slots: int
    active_owners: int
    pending_transitions: int
    pending_retirements: int


@dataclass(frozen=True, slots=True)
class StateCopyIntent:
    transition: StateTransitionLease
    owner_id: int
    source: StateSlotLease | None
    destination: StateSlotLease
    byte_count: int


@dataclass(frozen=True, slots=True)
class StateCopyReceipt:
    transition: StateTransitionLease
    source: StateSlotLease | None
    destination: StateSlotLease
    byte_count: int
    observed: int = 1
    written: int = 1


@dataclass(frozen=True, slots=True)
class StateCompletionReceipt:
    engine_epoch: int
    completion_domain: int
    completion_value: int
    confirmed: int = 1


@dataclass(frozen=True, slots=True)
class StateRetirementCertificate:
    retirement: StateRetirementLease
    slot: StateSlotLease
    byte_count: int
    completion_domain: int
    completion_value: int


@dataclass(frozen=True, slots=True)
class StatePublication:
    owner_id: int
    slot: StateSlotLease
    retirement: StateRetirementCertificate | None


__all__ = [name for name in globals() if name.startswith("State")]
