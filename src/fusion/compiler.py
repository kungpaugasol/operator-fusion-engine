"""Public compile seam. Two backends: interpreted walk, or generated code."""

from . import ops
from .codegen import compile_graph_codegen

__all__ = ["compile_graph"]

_SKIP = object()


def _collect_chain(terminal):
    chain = []
    cur = terminal
    while not isinstance(cur, ops.SourceOp):
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    return cur, chain


def _compile_walk(terminal):
    """Original walk-and-apply backend. Slower, but easier to debug."""
    source, chain = _collect_chain(terminal)

    if not chain or not isinstance(chain[-1], ops.ReduceOp):
        raise ValueError("graph must terminate in a ReduceOp")
    reduce_op = chain[-1]
    element_steps = chain[:-1]
    for step in element_steps:
        if isinstance(step, ops.ReduceOp):
            raise ValueError("ReduceOp may only appear as the terminal op")

    compiled_steps = []
    for step in element_steps:
        if isinstance(step, ops.FilterOp):
            compiled_steps.append(("filter", step.predicate))
        elif isinstance(step, ops.MapOp):
            compiled_steps.append(("map", step.fn))
        else:
            raise TypeError(f"unexpected step: {step!r}")

    data = source.data
    reduce_fn = reduce_op.fn

    def run():
        acc = None
        for x in data:
            val = x
            for kind, fn in compiled_steps:
                if kind == "filter":
                    if not fn(val):
                        val = _SKIP
                        break
                else:
                    val = fn(val)
            if val is _SKIP:
                continue
            if acc is None:
                acc = val
            else:
                acc = reduce_fn(acc, val)
        return acc if acc is not None else 0.0

    return run


def compile_graph(terminal, strategy="codegen"):
    if strategy == "codegen":
        return compile_graph_codegen(terminal)
    if strategy == "walk":
        return _compile_walk(terminal)
    raise ValueError(f"unknown strategy: {strategy!r}")
