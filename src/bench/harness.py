import time
from array import array
from fusion.column import Col
from fusion.baseline import row_wise, eager_vectorized
from .datagen import make_column

__all__ = ["time_it", "run"]


def time_it(fn, *args, repeats=5):
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best


def run(sizes):
    predicate = lambda x: x > 500
    mapfn = lambda x: x * 2

    print(f"{'n':>10}  {'row':>10}  {'vec':>10}  {'walk':>10}  {'codegen':>10}")
    for n in sizes:
        data = make_column(n)

        t_row = time_it(row_wise, data, predicate, mapfn)
        t_vec = time_it(eager_vectorized, data, predicate, mapfn)
        t_walk = time_it(_run_strategy, data, predicate, mapfn, "walk")
        t_gen = time_it(_run_strategy, data, predicate, mapfn, "codegen")

        print(f"{n:>10}  {t_row:>10.4f}  {t_vec:>10.4f}  {t_walk:>10.4f}  {t_gen:>10.4f}")


def _run_strategy(data, predicate, mapfn, strategy):
    from fusion import ops
    from fusion.compiler import compile_graph
    src = ops.SourceOp(data)
    filt = ops.FilterOp(src, predicate)
    mapped = ops.MapOp(filt, mapfn)
    red = ops.ReduceOp(mapped, lambda a, b: a + b, 0.0)
    return compile_graph(red, strategy=strategy)()
