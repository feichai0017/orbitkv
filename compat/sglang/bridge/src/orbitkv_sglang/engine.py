from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any

from .config import RuntimeConfig, load_config
from .bridge import state as _state


_OWNER_ATTRIBUTE = "_orbitkv_lifecycle_owner"
_CONSTRUCTOR_KEY = object()
_OWNER_LOCK = RLock()
_OWNER: OrbitKvLifecycleOwner | None = None


def _register_radix_factory(factory: Callable[[Any], Any]) -> None:
    from sglang.srt.mem_cache.registry import (
        get_radix_cache_factory,
        register_radix_cache_backend,
    )

    if get_radix_cache_factory("orbitkv") is not None:
        raise RuntimeError(
            "the OrbitKV radix-cache name is already owned by another integration"
        )
    register_radix_cache_backend("orbitkv", factory)


def _build_prefix_cache(context: Any) -> Any:
    from .bridge.prefix_cache import _build_prefix_cache as implementation

    return implementation(context)


def _validate_configurator(
    native_configure: Callable[..., Any],
    configurator: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    from .bridge.validation import _validate_configurator as implementation

    return implementation(native_configure, configurator, *args, **kwargs)


def _validate_source_checkout() -> None:
    from .bridge.validation import _validate_sglang_revision

    _validate_sglang_revision()


def _build_token_to_kv_pool_allocator(
    configurator: Any,
    *,
    sizes: Any,
    token_to_kv_pool: Any,
    is_dsv4_model: bool,
    req_to_token_pool: Any,
    token_to_kv_pool_allocator: Any,
) -> Any:
    from .bridge.facade import (
        _build_token_to_kv_pool_allocator as implementation,
    )

    return implementation(
        configurator,
        sizes=sizes,
        token_to_kv_pool=token_to_kv_pool,
        is_dsv4_model=is_dsv4_model,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
    )


def _alloc_for_extend(batch: Any) -> tuple[Any, Any, Any]:
    from .bridge.lowering import _alloc_for_extend as implementation

    return implementation(batch)


def _alloc_for_decode(batch: Any, token_per_req: int) -> Any:
    from .bridge.lowering import _alloc_for_decode as implementation

    return implementation(batch, token_per_req)


def _manager_maybe_evict_swa(batch: Any) -> None:
    from .bridge.lowering import _manager_maybe_evict_swa as implementation

    implementation(batch)


def _get_next_batch_to_run(
    native_get_next_batch: Callable[..., Any],
    scheduler: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    from .bridge.lowering import _get_next_batch_to_run as implementation

    return implementation(native_get_next_batch, scheduler, *args, **kwargs)


def _run_scheduled_batch(
    native_run_batch: Callable[..., Any],
    scheduler: Any,
    batch: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    from .bridge.session_lifecycle import run_scheduled_batch

    return run_scheduled_batch(
        native_run_batch, scheduler, batch, *args, **kwargs
    )


def _get_internal_state(
    native_get_internal_state: Callable[..., Any],
    scheduler: Any,
    *args: Any,
    owner: Any | None = None,
    **kwargs: Any,
) -> Any:
    from .observability import augment_internal_state

    return augment_internal_state(
        native_get_internal_state,
        scheduler,
        *args,
        owner=owner,
        **kwargs,
    )


def _release_kv_cache(
    req: Any, tree_cache: Any, is_insert: bool = True
) -> None:
    from .bridge.lowering import _release_kv_cache as implementation

    implementation(req, tree_cache, is_insert)


def _prepare_waiting_request_removal(req: Any, tree_cache: Any) -> bool:
    from .bridge.waiting_cleanup import prepare_waiting_request_removal

    return prepare_waiting_request_removal(req, tree_cache)


def _validate_direct_source_scope(config: RuntimeConfig) -> Any:
    from .runtime_admission import admit_product_takeover_config

    try:
        return admit_product_takeover_config(config)
    except ValueError as error:
        raise RuntimeError(
            f"OrbitKV product takeover admission failed: {error}"
        ) from error


class OrbitKvLifecycleOwner:
    """Process-wide OrbitKV authority for direct SGLang source wiring.

    ``create`` is the explicit, one-shot constructor. ``get_owner`` lazily
    creates the same singleton for source sites that cannot own startup. The
    existing validator remains authoritative for the eager, non-overlap,
    BF16/NHD runtime envelope.
    """

    __slots__ = ("_config", "_product_profile", "_radix_factory")

    def __init__(self, key: object, config: RuntimeConfig, product_profile: Any) -> None:
        if key is not _CONSTRUCTOR_KEY:
            raise TypeError(
                "OrbitKvLifecycleOwner must be created with "
                "OrbitKvLifecycleOwner.create()"
            )
        self._config = config
        self._product_profile = product_profile

        def radix_factory(context: Any) -> Any:
            self._assert_current()
            return self._adopt(_build_prefix_cache(context), "radix cache")

        self._radix_factory = radix_factory

    @classmethod
    def create(
        cls, environ: Mapping[str, str] | None = None
    ) -> OrbitKvLifecycleOwner:
        """Load and publish the sole direct-source lifecycle owner."""

        global _OWNER
        with _OWNER_LOCK:
            if _OWNER is not None:
                raise RuntimeError(
                    "the OrbitKV lifecycle owner was initialized more than once"
                )
            if _state._CONFIG is not None:
                raise RuntimeError(
                    "OrbitKV configuration is already owned by another integration"
                )
            _validate_source_checkout()
            config = load_config(environ)
            if not isinstance(config, RuntimeConfig):
                raise TypeError("load_config returned a non-canonical runtime config")
            product_profile = _validate_direct_source_scope(config)
            owner = cls(_CONSTRUCTOR_KEY, config, product_profile)

            # Publish the canonical config while registering so the registered
            # factory cannot ever observe an owner without its matching config.
            _state._CONFIG = config
            _state._PRODUCT_PROFILE = product_profile
            try:
                _register_radix_factory(owner._radix_factory)
            except Exception:
                if _state._CONFIG is config:
                    _state._CONFIG = None
                if _state._PRODUCT_PROFILE is product_profile:
                    _state._PRODUCT_PROFILE = None
                raise
            if (
                _state._CONFIG is not config
                or _state._PRODUCT_PROFILE is not product_profile
            ):
                raise RuntimeError(
                    "OrbitKV configuration or product-profile ownership changed "
                    "during initialization"
                )
            _OWNER = owner
            return owner

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def product_profile(self) -> Any:
        return self._product_profile

    @property
    def radix_factory(self) -> Callable[[Any], Any]:
        return self._radix_factory

    def _assert_current(self) -> None:
        if (
            _OWNER is not self
            or _state._CONFIG is not self._config
            or _state._PRODUCT_PROFILE is not self._product_profile
        ):
            raise RuntimeError(
                "OrbitKV lifecycle operation came from a foreign or stale owner"
            )

    def _adopt(self, value: Any, label: str) -> Any:
        current = getattr(value, _OWNER_ATTRIBUTE, None)
        if current is not None and current is not self:
            raise RuntimeError(f"OrbitKV received a foreign {label}")
        try:
            setattr(value, _OWNER_ATTRIBUTE, self)
        except Exception as error:
            raise RuntimeError(
                f"OrbitKV could not bind lifecycle ownership to the {label}"
            ) from error
        return value

    def configure(
        self,
        native_configure: Callable[..., Any],
        configurator: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Validate and run SGLang's eager, non-overlap configuration."""

        self._assert_current()
        return _validate_configurator(
            native_configure, configurator, *args, **kwargs
        )

    def build_allocator(
        self,
        configurator: Any,
        *,
        sizes: Any,
        token_to_kv_pool: Any,
        is_dsv4_model: bool,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
    ) -> Any:
        self._assert_current()
        allocator = _build_token_to_kv_pool_allocator(
            configurator,
            sizes=sizes,
            token_to_kv_pool=token_to_kv_pool,
            is_dsv4_model=is_dsv4_model,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        return self._adopt(allocator, "KV allocator")

    def prepare_extend(self, batch: Any) -> tuple[Any, Any, Any]:
        self._assert_current()
        return _alloc_for_extend(batch)

    def prepare_decode(self, batch: Any, token_per_req: int) -> Any:
        self._assert_current()
        return _alloc_for_decode(batch, token_per_req)

    def maybe_evict_swa(self, batch: Any) -> None:
        self._assert_current()
        _manager_maybe_evict_swa(batch)

    def next_batch(
        self,
        native_get_next_batch: Callable[..., Any],
        scheduler: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._assert_current()
        return _get_next_batch_to_run(
            native_get_next_batch, scheduler, *args, **kwargs
        )

    def run_batch(
        self,
        native_run_batch: Callable[..., Any],
        scheduler: Any,
        batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._assert_current()
        return _run_scheduled_batch(
            native_run_batch, scheduler, batch, *args, **kwargs
        )

    def internal_state(
        self,
        native_get_internal_state: Callable[..., Any],
        scheduler: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._assert_current()
        return _get_internal_state(
            native_get_internal_state,
            scheduler,
            *args,
            owner=self,
            **kwargs,
        )

    def prepare_waiting_request_removal(
        self, req: Any, tree_cache: Any
    ) -> bool:
        """Finish request ownership before SGLang removes a queued request."""

        self._assert_current()
        return _prepare_waiting_request_removal(req, tree_cache)

    def release_request(
        self, req: Any, tree_cache: Any, is_insert: bool = True
    ) -> None:
        self._assert_current()
        _release_kv_cache(req, tree_cache, is_insert)


def get_owner(
    environ: Mapping[str, str] | None = None,
) -> OrbitKvLifecycleOwner:
    """Return the process owner, creating it from canonical config if absent."""

    with _OWNER_LOCK:
        if _OWNER is None:
            return OrbitKvLifecycleOwner.create(environ)
        if environ is not None:
            raise RuntimeError(
                "the initialized OrbitKV lifecycle owner cannot be reconfigured"
            )
        _OWNER._assert_current()
        return _OWNER


__all__ = ("OrbitKvLifecycleOwner", "get_owner")
