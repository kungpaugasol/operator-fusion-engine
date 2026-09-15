"""Graph node vocabulary. Zero logic lives here — only data."""

from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["SourceOp", "FilterOp", "MapOp", "ReduceOp"]


@dataclass
class SourceOp:
    data: Any  # the raw array.array — the graph's root


@dataclass
class FilterOp:
    parent: Any
    predicate: Callable[[float], bool]


@dataclass
class MapOp:
    parent: Any
    fn: Callable[[float], float]


@dataclass
class ReduceOp:
    parent: Any
    fn: Callable[[float, float], float]
    init: float
