"""Compile a graph into real Python source, exec'd once, then called.

This is the Phase 3b move: pay the interpretation cost at compile time, not
per element. The generated function has no per-element dispatch — each step
of the chain is emitted as straight-line Python.
"""

from . import ops

__all__ = ["compile_graph_codegen"]

_SKIP = object()


def _collect_chain(terminal):
    chain = []
    cur = terminal
    while not isinstance(cur, ops.SourceOp):
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    return cur, chain


def compile_graph_codegen(terminal):
    source, chain = _collect_chain(terminal)

    if not chain or not isinstance(chain[-1], ops.ReduceOp):
        raise ValueError("graph must terminate in a ReduceOp")
    reduce_op = chain[-1]
    element_steps = chain[:-1]
    for step in element_steps:
        if isinstance(step, ops.ReduceOp):
            raise ValueError("ReduceOp may only appear as the terminal op")

    # Bind the predicate/map/reduce callables as named locals in the generated
    # function's globals so they resolve via LOAD_GLOBAL (fast) rather than
    # being chased through a list of tuples per element.
    ns = {
        "data": source.data,
        "_SKIP": _SKIP,
        "_f0": None, "_m0": None,  # placeholders, filled below
    }

    lines = ["def _run():"]
    lines.append("    acc = None")
    lines.append("    for x in data:")
    lines.append("        val = x")

    fn_index = 0
    for step in element_steps:
        if isinstance(step, ops.FilterOp):
            name = f"_f{fn_index}"
            ns[name] = step.predicate
            lines.append(f"        if not {name}(val):")
            lines.append(f"            continue")
            fn_index += 1
        elif isinstance(step, ops.MapOp):
            name = f"_m{fn_index}"
            ns[name] = step.fn
            lines.append(f"        val = {name}(val)")
            fn_index += 1
        else:
            raise TypeError(f"unexpected step: {step!r}")

    ns["_reduce"] = reduce_op.fn
    lines.append("        if acc is None:")
    lines.append("            acc = val")
    lines.append("        else:")
    lines.append("            acc = _reduce(acc, val)")
    lines.append("    return acc if acc is not None else 0.0")

    src = "\n".join(lines)

    # exec the generated function once; cache it in this closure.
    code = compile(src, "<fusion.codegen>", "exec")
    exec(code, ns)
    return ns["_run"]
