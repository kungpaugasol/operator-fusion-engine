"""User-facing lazy column. Every method builds a node; nothing computes."""

from . import ops

__all__ = ["Col"]


class Col:
    def __init__(self, node):
        self._node = node

    @classmethod
    def from_array(cls, arr):
        return cls(ops.SourceOp(arr))

    def filter(self, predicate):
        return Col(ops.FilterOp(self._node, predicate))

    def map(self, fn):
        return Col(ops.MapOp(self._node, fn))

    def sum(self):
        from .compiler import compile_graph
        node = ops.ReduceOp(self._node, lambda a, b: a + b, 0.0)
        return compile_graph(node)()
