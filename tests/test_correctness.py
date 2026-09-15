import random
from array import array
from fusion.column import Col
from fusion.baseline import row_wise, eager_vectorized


def test_all_three_agree():
    data = array('d', [random.random() * 100 for _ in range(10_000)])
    predicate = lambda x: x > 50
    mapfn = lambda x: x * 2

    expected = row_wise(data, predicate, mapfn)
    eager = eager_vectorized(data, predicate, mapfn)
    fused = Col.from_array(data).filter(predicate).map(mapfn).sum()

    # Relative tolerance: floating-point summation order differs between the
    # three implementations, so compare proportionally, not absolutely.
    rtol = 1e-12
    assert abs(eager - expected) <= rtol * abs(expected)
    assert abs(fused - expected) <= rtol * abs(expected)

def test_empty_after_filter_returns_zero():
    data = array('d', [1.0, 2.0, 3.0])
    fused = Col.from_array(data).filter(lambda x: x > 100).sum()
    assert fused == 0.0


def test_map_before_filter():
    data = array('d', [1.0, 2.0, 3.0, 4.0])
    # map first, then filter: (x*10) > 25 → keep 3,4 → 30+40 = 70
    fused = Col.from_array(data).map(lambda x: x * 10).filter(lambda x: x > 25).sum()
    assert fused == 70.0

def test_walk_and_codegen_agree():
    from array import array
    from fusion import ops
    from fusion.compiler import compile_graph

    data = array('d', [1.0, 2.0, 3.0, 4.0])
    pred = lambda x: x > 2
    mp = lambda x: x * 10

    src = ops.SourceOp(data)
    node = ops.ReduceOp(
        ops.MapOp(ops.FilterOp(src, pred), mp),
        lambda a, b: a + b,
        0.0,
    )

    walk = compile_graph(node, strategy="walk")()
    gen = compile_graph(node, strategy="codegen")()
    assert walk == gen == 70.0
