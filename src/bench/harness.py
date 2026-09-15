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
    print(f"{'n':>10}  {'row':>10}  {'vec':>10}  {'fused':>10}")
    for n in sizes:
        data = make_column(n)
        t_row = time_it(row_wise, data, predicate, mapfn)
        t_vec = time_it(eager_vectorized, data, predicate, mapfn)
        t_fused = time_it(lambda: Col.from_array(data).filter(predicate).map(mapfn).sum())
        print(f"{n:>10}  {t_row:>10.4f}  {t_vec:>10.4f}  {t_fused:>10.4f}")
