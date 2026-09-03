"""Pure eviction planning over prefix-tree metadata."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Sequence, TYPE_CHECKING, TypeVar


class _PrefixNodeProtocol(Protocol):
    boundary: int
    prefix: Any | None
    resident_count: int
    swa_ref_count: int
    parent: _PrefixNodeProtocol | None
    children: dict[bytes, _PrefixNodeProtocol]
    lock_ref: int
    last_access: int


_PrefixId = TypeVar("_PrefixId")


@dataclass(frozen=True, slots=True)
class _EvictionPlanItem(Generic[_PrefixId]):
    prefix: _PrefixId
    full_tokens: int
    swa_tokens: int


class PrefixEvictionMixin(Generic[_PrefixId]):
    """Compute read-only eviction plans for a compatible prefix tree."""

    root_node: _PrefixNodeProtocol
    _nodes: dict[tuple[int, bytes], _PrefixNodeProtocol]
    page_size: int

    if TYPE_CHECKING:

        def _full_edge_tokens(self, node: _PrefixNodeProtocol) -> int: ...

    def _published_parent_map(
        self,
    ) -> dict[_PrefixNodeProtocol, _PrefixNodeProtocol | None]:
        result: dict[_PrefixNodeProtocol, _PrefixNodeProtocol | None] = {}
        stack = [(self.root_node, None)]
        while stack:
            node, published_parent = stack.pop()
            next_parent = published_parent
            if node is not self.root_node and node.prefix is not None:
                result[node] = published_parent
                next_parent = node
            stack.extend(
                (child, next_parent) for child in node.children.values()
            )
        return result

    def _eviction_plan(
        self, requested_full: int, requested_swa: int, *, evict_all: bool = False
    ) -> tuple[_PrefixId, ...]:
        plan = self._complete_eviction_plan()
        if evict_all:
            return tuple(item.prefix for item in plan)
        if requested_full <= 0 and requested_swa <= 0:
            return ()
        full_estimate = swa_estimate = 0
        selected = []
        for item in plan:
            selected.append(item.prefix)
            full_estimate += item.full_tokens
            swa_estimate += item.swa_tokens
            if full_estimate >= requested_full and swa_estimate >= requested_swa:
                break
        return tuple(selected)

    def _complete_eviction_plan(
        self, *, _work: dict[str, int] | None = None
    ) -> tuple[_EvictionPlanItem[_PrefixId], ...]:
        published_parents = self._published_parent_map()
        child_counts = {node: 0 for node in published_parents}
        for parent in published_parents.values():
            if parent is not None:
                child_counts[parent] += 1
        own_prefix_live = {
            node: node.prefix is not None for node in self._nodes.values()
        }
        remaining_children = {
            node: len(node.children) for node in self._nodes.values()
        }
        heap: list[tuple[int, int, _PrefixNodeProtocol]] = []
        for node, count in child_counts.items():
            if count == 0 and node.lock_ref == 0:
                heapq.heappush(heap, (node.last_access, id(node), node))
        selected_nodes: list[_PrefixNodeProtocol] = []
        selected_prefixes: list[_PrefixId] = []
        full_estimates: list[int] = []
        while heap:
            _clock, _identity, node = heapq.heappop(heap)
            assert node.prefix is not None
            full_estimate = 0
            parent = published_parents[node]
            if not own_prefix_live[node]:
                raise RuntimeError("semantic radix prefix was selected twice")
            own_prefix_live[node] = False
            current = node
            while (
                current is not self.root_node
                and not own_prefix_live[current]
                and remaining_children[current] == 0
            ):
                full_estimate += self._full_edge_tokens(current)
                structural_parent = current.parent
                assert structural_parent is not None
                if structural_parent is self.root_node:
                    break
                remaining_children[structural_parent] -= 1
                if remaining_children[structural_parent] < 0:
                    raise RuntimeError(
                        "semantic radix eviction census underflowed"
                    )
                current = structural_parent
            if parent is not None:
                child_counts[parent] -= 1
                if child_counts[parent] == 0 and parent.lock_ref == 0:
                    heapq.heappush(
                        heap, (parent.last_access, id(parent), parent)
                    )
            selected_nodes.append(node)
            selected_prefixes.append(node.prefix)
            full_estimates.append(full_estimate)
        swa_estimates = self._swa_eviction_estimates(selected_nodes, _work)
        return tuple(
            _EvictionPlanItem(prefix, full_tokens, swa_tokens)
            for prefix, full_tokens, swa_tokens in zip(
                selected_prefixes, full_estimates, swa_estimates, strict=True
            )
        )

    def _swa_eviction_estimates(
        self,
        selected: Sequence[_PrefixNodeProtocol],
        work: dict[str, int] | None,
    ) -> tuple[int, ...]:
        """Assign each last-selected SWA page to one eviction-plan item."""

        preorder, top_exclusive, coverage = self._swa_selected_coverage(
            selected, work
        )

        freeable = {
            node
            for node in preorder[1:]
            if coverage[node] > 0 and coverage[node] == node.swa_ref_count
        }
        next_free: dict[
            _PrefixNodeProtocol, _PrefixNodeProtocol | None
        ] = {self.root_node: None}

        def find(
            node: _PrefixNodeProtocol | None,
        ) -> _PrefixNodeProtocol | None:
            trail = []
            current = node
            while current is not None and next_free[current] is not current:
                trail.append(current)
                current = next_free[current]
                if work is not None:
                    work["swa_path_steps"] += 1
            for item in trail:
                next_free[item] = current
            return current

        for node in preorder[1:]:
            parent = node.parent
            assert parent is not None
            next_free[node] = node if node in freeable else find(parent)
            if work is not None:
                work["swa_path_steps"] += 1

        # Later plan items are the last decrements.  Removing each freeable
        # structural node from this parent-chain DSU assigns it exactly once;
        # locked or otherwise unselected references were excluded above.
        estimates = [0] * len(selected)
        assigned = 0
        for index in range(len(selected) - 1, -1, -1):
            node = selected[index]
            boundary = top_exclusive[node].boundary
            current = find(node)
            while current is not None and current.boundary > boundary:
                estimates[index] += self.page_size
                assigned += 1
                parent = current.parent
                assert parent is not None
                next_free[current] = find(parent)
                current = find(current)
                if work is not None:
                    work["swa_path_steps"] += 1
        if assigned != len(freeable):
            raise RuntimeError("semantic SWA eviction ownership was incomplete")
        return tuple(estimates)

    def _swa_selected_coverage(
        self,
        selected: Sequence[_PrefixNodeProtocol],
        work: dict[str, int] | None,
    ) -> tuple[
        list[_PrefixNodeProtocol],
        dict[_PrefixNodeProtocol, _PrefixNodeProtocol],
        dict[_PrefixNodeProtocol, int],
    ]:
        """Return exact selected-window coverage with one structural-tree pass."""

        if work is not None:
            work["swa_path_steps"] = 0

        selected_windows = {
            node: node.resident_count - node.boundary // self.page_size
            for node in selected
        }
        preorder: list[_PrefixNodeProtocol] = []
        top_exclusive: dict[_PrefixNodeProtocol, _PrefixNodeProtocol] = {}
        path: list[_PrefixNodeProtocol] = []
        stack = [(self.root_node, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                path.pop()
                continue
            path.append(node)
            preorder.append(node)
            if work is not None:
                work["swa_path_steps"] += 1
            if node in selected_windows:
                pages = selected_windows[node]
                depth = len(path) - 1
                if pages < 0 or pages > depth:
                    raise RuntimeError("semantic SWA eviction census underflowed")
                top_exclusive[node] = path[depth - pages]
            stack.append((node, True))
            stack.extend(
                (child, False)
                for child in reversed(tuple(node.children.values()))
            )
        if len(preorder) != len(self._nodes) + 1 or len(top_exclusive) != len(
            selected
        ):
            raise RuntimeError(
                "semantic radix topology changed during eviction"
            )

        # Tree-path difference computes how many selected prefix windows cover
        # each structural page without walking any individual window.
        coverage = {node: 0 for node in preorder}
        for node in selected:
            coverage[node] += 1
            coverage[top_exclusive[node]] -= 1
            if work is not None:
                work["swa_path_steps"] += 1
        for node in reversed(preorder[1:]):
            parent = node.parent
            assert parent is not None
            coverage[parent] += coverage[node]
            if work is not None:
                work["swa_path_steps"] += 1
        if any(
            count < 0 or count > node.swa_ref_count
            for node, count in coverage.items()
            if node is not self.root_node
        ):
            raise RuntimeError("semantic SWA eviction census underflowed")
        return preorder, top_exclusive, coverage
