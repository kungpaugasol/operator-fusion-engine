"""Turns a graph of ops into a single fused pass over the source."""

from . import ops

__all__ = ["compile_graph"]

_SKIP = object()  # sentinel: element was filtered out


def _collect_chain(terminal):
    """Walk from terminal back to SourceOp; return (source, [steps...])."""
    chain = []
    cur = terminal
    while not isinstance(cur, ops.SourceOp):
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    return cur, chain


def compile_graph(terminal):
    """Compile the graph ending at `terminal` into a zero-arg callable."""
    source, chain = _collect_chain(terminal)

    # Split the chain into (a) per-element filter/map steps and (b) the terminal reduce.
    # A ReduceOp is only valid as the last step; anything after it is a bug.
    if not chain or not isinstance(chain[-1], ops.ReduceOp):
        raise ValueError("graph must terminate in a ReduceOp")
    reduce_op = chain[-1]
    element_steps = chain[:-1]
    for step in element_steps:
        if isinstance(step, ops.ReduceOp):
            raise ValueError("ReduceOp may only appear as the terminal op")

    # Pre-bind step types so the hot loop doesn't do isinstance() per element.
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
    init = reduce_op.init

    def run():
        acc = init
        first = True
        for x in data:
            val = x
            for kind, fn in compiled_steps:
                if kind == "filter":
                    if not fn(val):
                        val = _SKIP
                        break
                else:  # map
                    val = fn(val)
            if val is _SKIP:
                continue
            if first:
                acc = val
                first = False
            else:
                acc = reduce_fn(acc, val)
        return acc

    return run
