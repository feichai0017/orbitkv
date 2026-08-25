"""Reference implementation of the engine-neutral OrbitKV adapter SPI.

The implementation owns no allocation policy.  It projects manager-issued,
generation-bearing addresses into external byte buffers or optional contiguous
Torch tensors.  It is deliberately synchronous on CPU; CUDA tensors use a
recorded event but this reference path does not claim copy/compute overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from threading import RLock
from typing import Iterable, Sequence
from uuid import uuid4

from orbitkv_runtime import (
    AdapterCapabilities,
    ArenaRegistration,
    BackendPageAddress,
    BackendTokenAddress,
    CompletionEvidence,
    CompletionFence,
    DataPlaneOperation,
    DataPlaneEvidence,
    MirrorEvidence,
    PageMirrorKey,
    PageMirrorUpdate,
    ReclamationCertificate,
    ResolvedPageAddress,
    ResolvedTokenAddress,
    ReuseEvidence,
    TokenCopy,
    TokenMirrorKey,
    TokenMirrorUpdate,
    TokenWrite,
    TokenWriteReceipt,
)


class AdapterError(RuntimeError):
    """Base class for reference-adapter failures."""


class AdapterPreflightError(AdapterError):
    """A complete operation was rejected before external state changed."""


class AdapterPoisonedError(AdapterError):
    """The adapter observed an uncertain or post-mutation failure."""


@dataclass(frozen=True, slots=True)
class ArenaBinding:
    registration: ArenaRegistration
    storage: object


@dataclass(slots=True)
class _FenceState:
    fence: CompletionFence
    event: object | None
    purpose: str
    covered_pages: tuple[BackendPageAddress, ...]
    ordered_after_fence_id: int | None = None
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class _Timeline:
    completion_domain: int
    completion_value: int
    device_key: str
    cuda_arena: _ArenaView | None
    stream: object | None


class _ArenaView:
    def __init__(self, binding: ArenaBinding) -> None:
        self.registration = binding.registration
        self.storage = binding.storage
        self.is_cuda = False
        self.device = None
        expected = (
            self.registration.page_count
            * self.registration.page_tokens
            * self.registration.token_bytes
        )
        try:
            view = memoryview(self.storage)
        except TypeError:
            view = None
        if view is not None:
            if view.readonly:
                raise AdapterPreflightError("arena storage must be writable")
            if not view.c_contiguous:
                raise AdapterPreflightError("arena storage must be contiguous")
            try:
                self._bytes = view.cast("B")
            except TypeError as error:
                raise AdapterPreflightError(
                    "arena storage must expose a byte-castable buffer"
                ) from error
            if self._bytes.nbytes != expected:
                raise AdapterPreflightError(
                    f"arena byte size {self._bytes.nbytes} does not match {expected}"
                )
            self._tensor = None
            self._torch = None
            return

        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise AdapterPreflightError(
                "Torch tensor storage requires the optional torch dependency"
            ) from error
        if not isinstance(self.storage, torch.Tensor):
            raise AdapterPreflightError("unrecognized Torch arena object")
        if self.storage.layout != torch.strided or not self.storage.is_contiguous():
            raise AdapterPreflightError(
                "Torch arena must be a contiguous strided tensor"
            )
        if self.storage.requires_grad:
            raise AdapterPreflightError("Torch arena must not require gradients")
        self.device = self.storage.device
        if self.device.type not in ("cpu", "cuda"):
            raise AdapterPreflightError(
                "Torch arena device must be CPU or CUDA"
            )
        actual = int(self.storage.numel()) * int(self.storage.element_size())
        if actual != expected:
            raise AdapterPreflightError(
                f"arena byte size {actual} does not match {expected}"
            )
        try:
            self._tensor = self.storage.detach().view(torch.uint8).reshape(
                self.registration.page_count,
                self.registration.page_tokens,
                self.registration.token_bytes,
            )
        except (RuntimeError, TypeError) as error:
            raise AdapterPreflightError(
                "Torch arena cannot be represented as contiguous bytes"
            ) from error
        self._bytes = None
        self._torch = torch
        self.is_cuda = self.device.type == "cuda"

    def read(self, page_index: int, token_offset: int) -> bytes:
        if self._bytes is not None:
            begin = (
                page_index * self.registration.page_tokens + token_offset
            ) * self.registration.token_bytes
            return bytes(self._bytes[begin : begin + self.registration.token_bytes])
        row = self._tensor[page_index, token_offset]
        if self.is_cuda:
            row = row.cpu()
        return bytes(row.tolist())

    def write(self, page_index: int, token_offset: int, payload: object) -> None:
        if self._bytes is not None:
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise TypeError("buffer arena write requires bytes-like payload")
            begin = (
                page_index * self.registration.page_tokens + token_offset
            ) * self.registration.token_bytes
            self._bytes[begin : begin + self.registration.token_bytes] = payload
            return
        value = (
            payload
            if isinstance(payload, self._torch.Tensor)
            else self._torch.tensor(
                list(payload), dtype=self._torch.uint8, device=self.device
            )
        )
        self._tensor[page_index, token_offset].copy_(value)

    def capture(self, page_index: int, token_offset: int) -> object:
        if self._bytes is not None:
            return self.read(page_index, token_offset)
        return self._tensor[page_index, token_offset].clone()


@dataclass(frozen=True, slots=True)
class _ResolvedToken:
    public: ResolvedTokenAddress
    arena: _ArenaView


class ReferencePagedAdapter:
    """Production-reusable tensor-arena adapter and executable oracle.

    The caller supplies storage and manager-issued identities.  The adapter
    performs byte-exact copies and maintains checked page/token mirror
    projections, but never allocates pages or decides that a page is dead.
    """

    def __init__(
        self,
        arenas: Sequence[ArenaBinding],
        *,
        adapter_id: str | None = None,
    ) -> None:
        if not arenas:
            raise AdapterPreflightError("at least one arena is required")
        self._lock = RLock()
        if adapter_id is not None and (
            not isinstance(adapter_id, str) or not adapter_id
        ):
            raise AdapterPreflightError("adapter_id must be a nonempty string")
        instance_nonce = uuid4().hex
        self._adapter_id = (
            f"{adapter_id}:{instance_nonce}"
            if adapter_id is not None
            else f"reference-{instance_nonce}"
        )
        self._arenas: dict[int, _ArenaView] = {}
        engine_epochs = set()
        domain_classes = set()
        for binding in arenas:
            registration = binding.registration
            if registration.pool_id in self._arenas:
                raise AdapterPreflightError("duplicate pool_id registration")
            identity = (registration.backend_domain, registration.class_id)
            if identity in domain_classes:
                raise AdapterPreflightError(
                    "duplicate backend-domain/class registration"
                )
            self._arenas[registration.pool_id] = _ArenaView(binding)
            engine_epochs.add(registration.engine_epoch)
            domain_classes.add(identity)
        if len(engine_epochs) != 1:
            raise AdapterPreflightError(
                "all arenas must belong to one engine epoch"
            )
        self._engine_epoch = engine_epochs.pop()
        self._poison_reason: str | None = None
        self._page_mirrors: dict[PageMirrorKey, BackendPageAddress] = {}
        self._token_mirrors: dict[TokenMirrorKey, BackendTokenAddress] = {}
        self._generations: dict[tuple[int, int], int] = {}
        self._reuse_allowed_after: dict[tuple[int, int], int] = {}
        self._fences: dict[int, _FenceState] = {}
        self._domain_values: dict[int, int] = {}
        self._domain_devices: dict[int, str] = {}
        self._domain_last_events: dict[int, object] = {}
        self._latest_page_fences: dict[tuple[int, int, int], int] = {}
        self._next_fence_id = 1
        self._next_operation_id = 1
        self._next_evidence_id = 1
        self._pending_reuse: dict[int, ReuseEvidence] = {}
        self._acknowledged_evidence: set[int] = set()
        self._observed_transactions = set()

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            cuda_arenas=any(arena.is_cuda for arena in self._arenas.values())
        )

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def page_mirrors(self) -> dict[PageMirrorKey, BackendPageAddress]:
        with self._lock:
            return dict(self._page_mirrors)

    @property
    def token_mirrors(self) -> dict[TokenMirrorKey, BackendTokenAddress]:
        with self._lock:
            return dict(self._token_mirrors)

    def poison(self, reason: str) -> None:
        if not reason:
            raise ValueError("poison reason must not be empty")
        with self._lock:
            if self._poison_reason is None:
                self._poison_reason = reason

    def _ensure_healthy(self) -> None:
        if self._poison_reason is not None:
            raise AdapterPoisonedError(
                f"adapter is poisoned: {self._poison_reason}"
            )

    def _validate_page(
        self, address: BackendPageAddress
    ) -> tuple[_ArenaView, int]:
        try:
            arena = self._arenas[address.page.pool_id]
        except KeyError as error:
            raise AdapterPreflightError("page names an unregistered pool") from error
        registration = arena.registration
        page_index = address.page.page_id - registration.first_page_id
        backend_page_index = address.backend_index - registration.backend_base_index
        if (
            address.page.engine_epoch != self._engine_epoch
            or address.page.engine_epoch != registration.engine_epoch
            or address.page.pool_epoch != registration.pool_epoch
            or address.class_id != registration.class_id
            or address.backend_domain != registration.backend_domain
            or not 0 <= page_index < registration.page_count
            or page_index != backend_page_index
        ):
            raise AdapterPreflightError(
                "page address does not match its registered arena identity"
            )
        return arena, page_index

    def _plan_generations(
        self, addresses: Sequence[BackendPageAddress]
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        generations = dict(self._generations)
        reuse_allowed = dict(self._reuse_allowed_after)
        pending = {
            (certificate.page.pool_id, certificate.page.page_id)
            for evidence in self._pending_reuse.values()
            for certificate in evidence.certificates
        }
        batch_generations: dict[tuple[int, int], int] = {}
        for address in addresses:
            self._validate_page(address)
            key = (address.page.pool_id, address.page.page_id)
            generation = address.page.generation
            if (
                key in batch_generations
                and batch_generations[key] != generation
            ):
                raise AdapterPreflightError(
                    "one batch names multiple generations of a physical page"
                )
            batch_generations[key] = generation
            if key in pending:
                raise AdapterPreflightError(
                    "retiring page cannot be resolved before its manager ACK"
                )
            current = generations.get(key)
            if current is None:
                generations[key] = generation
            elif key in reuse_allowed:
                if generation <= current:
                    raise AdapterPreflightError(
                        "acknowledged page must reappear at a higher generation"
                    )
                generations[key] = generation
                del reuse_allowed[key]
            elif generation != current:
                raise AdapterPreflightError(
                    "page generation changed without an acknowledged reuse gate"
                )
        return generations, reuse_allowed

    def _commit_generations(
        self,
        plan: tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]],
    ) -> None:
        self._generations, self._reuse_allowed_after = plan

    def _validate_token(self, address: BackendTokenAddress) -> _ResolvedToken:
        arena, page_index = self._validate_page(address.page_address)
        if address.token_offset >= arena.registration.page_tokens:
            raise AdapterPreflightError("token offset is outside its page")
        return _ResolvedToken(
            ResolvedTokenAddress(
                address=address,
                arena=arena.storage,
                arena_page_index=page_index,
                token_index=page_index * arena.registration.page_tokens
                + address.token_offset,
                byte_length=arena.registration.token_bytes,
            ),
            arena,
        )

    @staticmethod
    def _reject_duplicates(name: str, values: Iterable[object]) -> None:
        values = tuple(values)
        if len(set(values)) != len(values):
            raise AdapterPreflightError(f"duplicate {name}")

    def resolve_pages(
        self, addresses: Sequence[BackendPageAddress]
    ) -> tuple[ResolvedPageAddress, ...]:
        with self._lock:
            self._ensure_healthy()
            addresses = tuple(addresses)
            generations = self._plan_generations(addresses)
            result = []
            for address in addresses:
                arena, page_index = self._validate_page(address)
                result.append(
                    ResolvedPageAddress(
                        address,
                        arena.storage,
                        page_index,
                        arena.registration.page_tokens
                        * arena.registration.token_bytes,
                    )
                )
            self._commit_generations(generations)
            return tuple(result)

    def resolve_tokens(
        self, addresses: Sequence[BackendTokenAddress]
    ) -> tuple[ResolvedTokenAddress, ...]:
        with self._lock:
            self._ensure_healthy()
            addresses = tuple(addresses)
            generations = self._plan_generations(
                tuple(address.page_address for address in addresses)
            )
            result = tuple(
                self._validate_token(address).public for address in addresses
            )
            self._commit_generations(generations)
            return result

    def read_tokens(
        self, addresses: Sequence[BackendTokenAddress]
    ) -> tuple[bytes, ...]:
        """Reference observation helper; not part of the portable SPI."""

        with self._lock:
            self._ensure_healthy()
            addresses = tuple(addresses)
            generations = self._plan_generations(
                tuple(address.page_address for address in addresses)
            )
            resolved = tuple(self._validate_token(item) for item in addresses)
            result = tuple(
                item.arena.read(
                    item.public.arena_page_index, item.public.address.token_offset
                )
                for item in resolved
            )
            self._commit_generations(generations)
            for address in addresses:
                # The observation completed synchronously, but it happened after
                # any earlier last-use declaration.  A fresh explicit last-use
                # fence is therefore required before this generation can retire.
                self._latest_page_fences[self._page_key(address.page_address)] = 0
            return result

    def _preflight_data(
        self,
        operation: DataPlaneOperation,
        copies: Sequence[TokenCopy],
        writes: Sequence[TokenWrite],
    ) -> tuple[
        tuple[tuple[TokenCopy, _ResolvedToken, _ResolvedToken], ...],
        tuple[tuple[TokenWrite, _ResolvedToken, bytes], ...],
        tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]],
    ]:
        copies = tuple(copies)
        writes = tuple(writes)
        contexts = tuple(dict.fromkeys(
            item.context for item in (*copies, *writes)
        ))
        transaction_owners = {}
        for context in contexts:
            self._validate_context(context, operation)
            previous = transaction_owners.setdefault(
                context.transaction, context.request
            )
            if previous != context.request:
                raise AdapterPreflightError(
                    "one transaction cannot authorize multiple requests"
                )
            if context.transaction in self._observed_transactions:
                raise AdapterPreflightError(
                    "manager transaction was already observed by the adapter"
                )
        generations = self._plan_generations(
            tuple(
                address.page_address
                for item in copies
                for address in (item.source, item.destination)
            )
            + tuple(item.destination.page_address for item in writes)
        )
        copy_plans = []
        write_plans = []
        destinations = []
        logical_tokens = []
        for item in copies:
            source = self._validate_token(item.source)
            destination = self._validate_token(item.destination)
            if (
                item.source.class_id != item.destination.class_id
                or item.source.backend_domain != item.destination.backend_domain
                or item.source.page.pool_id != item.destination.page.pool_id
            ):
                raise AdapterPreflightError(
                    "copy cannot cross class, backend domain, or pool"
                )
            if source.public.byte_length != destination.public.byte_length:
                raise AdapterPreflightError(
                    "copy source and destination token widths differ"
                )
            copy_plans.append((item, source, destination))
            destinations.append(item.destination)
            logical_tokens.append(
                (item.context, item.source.class_id, item.token_id)
            )
        for item in writes:
            destination = self._validate_token(item.destination)
            payload = bytes(item.payload)
            if len(payload) != destination.public.byte_length:
                raise AdapterPreflightError(
                    "write payload does not match destination token width"
                )
            write_plans.append((item, destination, payload))
            destinations.append(item.destination)
            logical_tokens.append(
                (item.context, item.destination.class_id, item.token_id)
            )
        self._reject_duplicates("data destination address", destinations)
        self._reject_duplicates("logical transaction token", logical_tokens)
        return tuple(copy_plans), tuple(write_plans), generations

    def _validate_context(
        self, context: object, operation: DataPlaneOperation
    ) -> None:
        try:
            engine_epoch = context.request.engine_epoch
            transaction_epoch = context.transaction.engine_epoch
            actual_operation = context.operation
        except AttributeError as error:
            raise AdapterPreflightError(
                "data operation lacks a valid manager context"
            ) from error
        if (
            engine_epoch != self._engine_epoch
            or transaction_epoch != self._engine_epoch
            or actual_operation is not operation
        ):
            raise AdapterPreflightError(
                "data operation context does not match adapter epoch or operation"
            )

    @staticmethod
    def _validate_completion_domain(
        completion_domain: int, touched: Sequence[_ArenaView]
    ) -> None:
        if isinstance(completion_domain, bool) or not isinstance(completion_domain, int):
            raise AdapterPreflightError("completion domain must be an integer")
        if completion_domain <= 0:
            raise AdapterPreflightError("completion domain must be positive")
        if completion_domain >= 1 << 64:
            raise AdapterPreflightError("completion domain exceeds uint64")
        devices = {
            str(arena.device) if arena.is_cuda else "cpu" for arena in touched
        }
        if len(devices) > 1:
            raise AdapterPreflightError(
                "one completion operation cannot span CPU/CUDA devices"
            )

    @staticmethod
    def _page_key(address: BackendPageAddress) -> tuple[int, int, int]:
        return address.page.pool_id, address.page.page_id, address.page.generation

    def _next_operation(self) -> int:
        if self._next_operation_id >= 1 << 64:
            raise AdapterPreflightError("operation sequence is exhausted")
        result = self._next_operation_id
        self._next_operation_id += 1
        return result

    def _prepare_timeline(
        self,
        completion_domain: int,
        touched: Sequence[_ArenaView],
        *,
        after_fence_ids: Sequence[int] = (),
    ) -> _Timeline:
        self._validate_completion_domain(completion_domain, touched)
        cuda = {str(arena.device): arena for arena in touched if arena.is_cuda}
        device_key = next(iter(cuda), "cpu")
        if (
            completion_domain in self._domain_devices
            and self._domain_devices[completion_domain] != device_key
        ):
            raise AdapterPreflightError(
                "completion domain cannot change its CPU/CUDA device timeline"
            )
        value = self._domain_values.get(completion_domain, 0) + 1
        if value >= 1 << 64:
            raise AdapterPreflightError("completion sequence is exhausted")
        if self._next_fence_id >= 1 << 64:
            raise AdapterPreflightError("completion fence sequence is exhausted")
        cuda_arena = next(iter(cuda.values()), None)
        stream = None
        if cuda_arena is not None:
            stream = cuda_arena._torch.cuda.current_stream(cuda_arena.device)
            dependencies = []
            predecessor = self._domain_last_events.get(completion_domain)
            if predecessor is not None:
                dependencies.append(predecessor)
            for fence_id in dict.fromkeys(after_fence_ids):
                state = self._fences.get(fence_id)
                if state is None:
                    raise AdapterPreflightError(
                        "completion dependency is unknown"
                    )
                if state.event is not None:
                    dependencies.append(state.event)
            for event in dict.fromkeys(dependencies):
                stream.wait_event(event)
        return _Timeline(
            completion_domain, value, device_key, cuda_arena, stream
        )

    def _record_completion(
        self,
        timeline: _Timeline,
        *,
        purpose: str,
        covered_pages: Sequence[BackendPageAddress] = (),
        ordered_after_fence_id: int | None = None,
    ) -> CompletionFence:
        covered_pages = tuple(covered_pages)
        self._reject_duplicates("completion covered page", covered_pages)
        fence_id = self._next_fence_id
        fence = CompletionFence(
            self._adapter_id,
            self._engine_epoch,
            timeline.completion_domain,
            timeline.completion_value,
            fence_id,
        )
        event = None
        if timeline.cuda_arena is not None:
            event = timeline.cuda_arena._torch.cuda.Event()
            event.record(timeline.stream)
            self._domain_last_events[timeline.completion_domain] = event
        self._domain_devices.setdefault(
            timeline.completion_domain, timeline.device_key
        )
        self._domain_values[timeline.completion_domain] = timeline.completion_value
        self._next_fence_id += 1
        self._fences[fence_id] = _FenceState(
            fence, event, purpose, covered_pages, ordered_after_fence_id
        )
        for page in covered_pages:
            self._latest_page_fences[self._page_key(page)] = fence_id
        return fence

    def record_completion(
        self,
        pages: Sequence[BackendPageAddress],
        *,
        completion_domain: int = 1,
    ) -> CompletionFence:
        with self._lock:
            self._ensure_healthy()
            pages = tuple(pages)
            if not pages:
                raise AdapterPreflightError(
                    "last-use fence must cover at least one exact page generation"
                )
            generations = self._plan_generations(pages)
            touched = tuple(self._validate_page(page)[0] for page in pages)
            timeline = self._prepare_timeline(
                completion_domain,
                touched,
                after_fence_ids=tuple(
                    self._latest_page_fences.get(self._page_key(page), 0)
                    for page in pages
                    if self._latest_page_fences.get(self._page_key(page), 0)
                ),
            )
            fence = self._record_completion(
                timeline,
                purpose="last_use",
                covered_pages=pages,
            )
            self._commit_generations(generations)
            return fence

    def _execute_data(
        self,
        operation: DataPlaneOperation,
        copies: Sequence[TokenCopy],
        writes: Sequence[TokenWrite],
        completion_domain: int,
    ) -> DataPlaneEvidence:
        self._ensure_healthy()
        copy_plans, write_plans, generations = self._preflight_data(
            operation, copies, writes
        )
        if not copy_plans and not write_plans:
            raise AdapterPreflightError(f"{operation.value} batch must not be empty")
        touched = tuple(
            arena
            for _item, _source, destination in copy_plans
            for arena in (_source.arena, destination.arena)
        ) + tuple(destination.arena for _item, destination, _payload in write_plans)
        covered_pages = tuple(
            address.page_address
            for item in copy_plans
            for address in (item[0].source, item[0].destination)
        ) + tuple(item.destination.page_address for item, _destination, _payload in write_plans)
        covered_pages = tuple(dict.fromkeys(covered_pages))
        timeline = self._prepare_timeline(
            completion_domain,
            touched,
            after_fence_ids=tuple(
                self._latest_page_fences.get(self._page_key(page), 0)
                for page in covered_pages
                if self._latest_page_fences.get(self._page_key(page), 0)
            ),
        )
        operation_id = self._next_operation()
        captured = tuple(
            source.arena.capture(
                source.public.arena_page_index, source.public.address.token_offset
            )
            for _item, source, _destination in copy_plans
        )
        mutated = False
        self._commit_generations(generations)
        try:
            for payload, (_item, _source, destination) in zip(
                captured, copy_plans, strict=True
            ):
                mutated = True
                destination.arena.write(
                    destination.public.arena_page_index,
                    destination.public.address.token_offset,
                    payload,
                )
            for _item, destination, payload in write_plans:
                mutated = True
                destination.arena.write(
                    destination.public.arena_page_index,
                    destination.public.address.token_offset,
                    payload,
                )
            fence = self._record_completion(
                timeline,
                purpose="data",
                covered_pages=covered_pages,
            )
        except Exception as error:
            if mutated:
                self.poison(
                    f"{operation.value} outcome became uncertain: {error}"
                )
                raise AdapterPoisonedError(
                    f"{operation.value} failed after external mutation"
                ) from error
            raise
        self._observed_transactions.update(
            item.context.transaction for item in (*copies, *writes)
        )
        return DataPlaneEvidence(
            operation_id,
            operation,
            tuple(item for item, _source, _destination in copy_plans),
            tuple(
                TokenWriteReceipt(
                    item.context,
                    item.token_id,
                    item.destination,
                    destination.public.byte_length,
                )
                for item, destination, _payload in write_plans
            ),
            fence,
        )

    def append(
        self,
        writes: Sequence[TokenWrite],
        *,
        cow_copies: Sequence[TokenCopy] = (),
        completion_domain: int = 1,
    ) -> DataPlaneEvidence:
        with self._lock:
            return self._execute_data(
                DataPlaneOperation.APPEND, cow_copies, writes, completion_domain
            )

    def relocate(
        self,
        moves: Sequence[TokenCopy],
        *,
        completion_domain: int = 1,
    ) -> DataPlaneEvidence:
        with self._lock:
            return self._execute_data(
                DataPlaneOperation.RELOCATE, moves, (), completion_domain
            )

    def _fence_state(self, fence: CompletionFence) -> _FenceState:
        if (
            fence.adapter_id != self._adapter_id
            or fence.engine_epoch != self._engine_epoch
        ):
            raise AdapterPreflightError("completion fence belongs to another adapter")
        state = self._fences.get(fence.fence_id)
        if state is None or state.fence != fence:
            raise AdapterPreflightError("completion fence is unknown or forged")
        return state

    def query_completion(
        self, fence: CompletionFence
    ) -> CompletionEvidence | None:
        with self._lock:
            self._ensure_healthy()
            state = self._fence_state(fence)
            try:
                ready = state.event is None or bool(state.event.query())
            except Exception as error:
                self.poison(f"completion query became uncertain: {error}")
                raise AdapterPoisonedError("completion query failed") from error
            if not ready:
                return None
            state.confirmed = True
            return CompletionEvidence(fence, state.covered_pages)

    def wait_completion(self, fence: CompletionFence) -> CompletionEvidence:
        with self._lock:
            self._ensure_healthy()
            state = self._fence_state(fence)
            try:
                if state.event is not None:
                    state.event.synchronize()
            except Exception as error:
                self.poison(f"completion wait became uncertain: {error}")
                raise AdapterPoisonedError("completion wait failed") from error
            state.confirmed = True
            return CompletionEvidence(fence, state.covered_pages)

    def _validate_evidence(self, evidence: CompletionEvidence) -> None:
        state = self._fence_state(evidence.fence)
        if not state.confirmed or evidence.covered_pages != state.covered_pages:
            raise AdapterPreflightError("completion evidence is not confirmed")

    def update_mirrors(
        self,
        page_updates: Sequence[PageMirrorUpdate],
        token_updates: Sequence[TokenMirrorUpdate],
        *,
        after: CompletionEvidence,
        completion_domain: int = 1,
    ) -> MirrorEvidence:
        with self._lock:
            self._ensure_healthy()
            self._validate_evidence(after)
            pages = tuple(page_updates)
            tokens = tuple(token_updates)
            self._reject_duplicates("page mirror key", (item.key for item in pages))
            self._reject_duplicates("token mirror key", (item.key for item in tokens))
            covered_pages = after.covered_pages
            for page in covered_pages:
                if (
                    self._latest_page_fences.get(self._page_key(page))
                    != after.fence.fence_id
                ):
                    raise AdapterPreflightError(
                        "mirror update depends on stale completion evidence"
                    )
            changed_pages = tuple(
                address
                for item in pages
                for address in (item.expected, item.replacement)
                if address is not None
            ) + tuple(
                address.page_address
                for item in tokens
                for address in (item.expected, item.replacement)
                if address is not None
            )
            if not set(changed_pages).issubset(set(covered_pages)):
                raise AdapterPreflightError(
                    "mirror update addresses are not covered by its dependency"
                )
            touched = tuple(self._validate_page(page)[0] for page in covered_pages)
            timeline = self._prepare_timeline(
                completion_domain,
                touched,
                after_fence_ids=(after.fence.fence_id,),
            )
            operation_id = self._next_operation()
            generations = self._plan_generations(
                tuple(
                    item.replacement
                    for item in pages
                    if item.replacement is not None
                )
                + tuple(
                    item.replacement.page_address
                    for item in tokens
                    if item.replacement is not None
                )
            )
            next_pages = dict(self._page_mirrors)
            next_tokens = dict(self._token_mirrors)
            for item in pages:
                if next_pages.get(item.key) != item.expected:
                    raise AdapterPreflightError("page mirror compare-and-swap failed")
                if item.replacement is not None:
                    if item.key.class_id != item.replacement.class_id:
                        raise AdapterPreflightError("page mirror class changed")
                    self._validate_page(item.replacement)
                    next_pages[item.key] = item.replacement
                else:
                    next_pages.pop(item.key, None)
            for item in tokens:
                if next_tokens.get(item.key) != item.expected:
                    raise AdapterPreflightError("token mirror compare-and-swap failed")
                if item.replacement is not None:
                    if item.key.class_id != item.replacement.class_id:
                        raise AdapterPreflightError("token mirror class changed")
                    self._validate_token(item.replacement)
                    next_tokens[item.key] = item.replacement
                else:
                    next_tokens.pop(item.key, None)
            self._page_mirrors = next_pages
            self._token_mirrors = next_tokens
            self._commit_generations(generations)
            try:
                fence = self._record_completion(
                    timeline,
                    purpose="mirror",
                    covered_pages=covered_pages,
                    ordered_after_fence_id=after.fence.fence_id,
                )
            except Exception as error:
                self.poison(f"mirror publication became uncertain: {error}")
                raise AdapterPoisonedError("mirror publication failed") from error
            return MirrorEvidence(
                operation_id, pages, tokens, after, fence
            )

    def prepare_reuse(
        self,
        certificates: Sequence[ReclamationCertificate],
        *,
        last_use: Sequence[CompletionEvidence],
        mirror_cleanup: CompletionEvidence,
    ) -> ReuseEvidence:
        with self._lock:
            self._ensure_healthy()
            certificates = tuple(certificates)
            last_use = tuple(last_use)
            if not certificates:
                raise AdapterPreflightError("reuse certificate batch must not be empty")
            if len(last_use) != 1:
                raise AdapterPreflightError(
                    "one reuse batch requires exactly one shared last-use fence"
                )
            self._reject_duplicates("reclamation certificate", certificates)
            self._reject_duplicates(
                "retiring physical page",
                ((item.page.pool_id, item.page.page_id) for item in certificates),
            )
            for evidence in last_use:
                self._validate_evidence(evidence)
                if self._fence_state(evidence.fence).purpose != "last_use":
                    raise AdapterPreflightError(
                        "reuse requires explicit last-use completion evidence"
                    )
            self._validate_evidence(mirror_cleanup)
            mirror_state = self._fence_state(mirror_cleanup.fence)
            if mirror_state.purpose != "mirror":
                raise AdapterPreflightError(
                    "mirror cleanup evidence was not emitted by a mirror operation"
                )
            completion_points = {
                (item.fence.completion_domain, item.fence.completion_value)
                for item in last_use
            }
            if mirror_state.ordered_after_fence_id not in {
                item.fence.fence_id for item in last_use
            }:
                raise AdapterPreflightError(
                    "mirror cleanup is not ordered after the supplied last-use evidence"
                )
            page_mirrors = set(self._page_mirrors.values())
            token_pages = {item.page_address for item in self._token_mirrors.values()}
            for certificate in certificates:
                self._validate_page(certificate.page_address)
                if certificate.page.engine_epoch != certificate.reclamation.engine_epoch:
                    raise AdapterPreflightError(
                        "reclamation and page engine epochs differ"
                    )
                key = (certificate.page.pool_id, certificate.page.page_id)
                if self._generations.get(key) != certificate.page.generation:
                    raise AdapterPreflightError(
                        "reclamation certificate names an unobserved or stale generation"
                    )
                point = (certificate.completion_domain, certificate.completion_value)
                if point not in completion_points:
                    raise AdapterPreflightError(
                        "certificate lacks exact completed last-use evidence"
                    )
                last_use_state = self._fence_state(last_use[0].fence)
                if certificate.page_address not in last_use_state.covered_pages:
                    raise AdapterPreflightError(
                        "last-use fence does not cover a retiring page generation"
                    )
                if (
                    self._latest_page_fences.get(
                        self._page_key(certificate.page_address)
                    )
                    != mirror_cleanup.fence.fence_id
                ):
                    raise AdapterPreflightError(
                        "retiring page was accessed after its mirror-cleanup fence"
                    )
                if (
                    certificate.page_address in page_mirrors
                    or certificate.page_address in token_pages
                ):
                    raise AdapterPreflightError(
                        "retiring page remains reachable from an adapter mirror"
                    )
                registration = self._arenas[certificate.page.pool_id].registration
                expected_begin = (
                    certificate.logical_ordinal * registration.page_tokens
                )
                if (
                    certificate.token_begin != expected_begin
                    or certificate.token_end_exclusive
                    > expected_begin + registration.page_tokens
                ):
                    raise AdapterPreflightError(
                        "reclamation certificate has a non-canonical token span"
                    )
                if key in self._reuse_allowed_after or any(
                    any(
                        (item.page.pool_id, item.page.page_id) == key
                        for item in pending.certificates
                    )
                    for pending in self._pending_reuse.values()
                ):
                    raise AdapterPreflightError(
                        "physical page already has pending or unused reuse evidence"
                    )
            evidence_id = self._next_evidence_id
            if evidence_id >= 1 << 64:
                raise AdapterPreflightError("reuse evidence sequence is exhausted")
            self._next_evidence_id += 1
            evidence = ReuseEvidence(
                self._adapter_id,
                evidence_id,
                certificates,
                last_use,
                mirror_cleanup,
            )
            self._pending_reuse[evidence_id] = evidence
            return evidence

    def note_reuse_acknowledged(self, evidence: ReuseEvidence) -> None:
        """Record a successful external manager ACK; this is not the ACK.

        Call exactly once and only after the manager accepted receipts derived
        from ``evidence.certificates``.  An uncertain manager return must poison
        the coordinator rather than invoke this method or retry the ACK.
        """

        with self._lock:
            self._ensure_healthy()
            if evidence.adapter_id != self._adapter_id:
                raise AdapterPreflightError("reuse evidence belongs to another adapter")
            pending = self._pending_reuse.get(evidence.evidence_id)
            if pending is not evidence:
                raise AdapterPreflightError(
                    "reuse evidence is unknown, forged, or already acknowledged"
                )
            for certificate in evidence.certificates:
                key = (certificate.page.pool_id, certificate.page.page_id)
                current = self._generations.get(key)
                if current != certificate.page.generation:
                    raise AdapterPreflightError(
                        "reuse evidence no longer names the current generation"
                    )
            del self._pending_reuse[evidence.evidence_id]
            self._acknowledged_evidence.add(evidence.evidence_id)
            for certificate in evidence.certificates:
                key = (certificate.page.pool_id, certificate.page.page_id)
                self._reuse_allowed_after[key] = certificate.page.generation


__all__ = [
    "AdapterError",
    "AdapterPoisonedError",
    "AdapterPreflightError",
    "ArenaBinding",
    "ReferencePagedAdapter",
]
