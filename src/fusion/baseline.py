"""Two obviously-correct implementations. Nothing here imports ops/compiler."""

__all__ = ["row_wise", "eager_vectorized"]


def row_wise(data, predicate, mapfn):
    total = 0.0
    for x in data:
        if predicate(x):
            total += mapfn(x)
    return total


def eager_vectorized(data, predicate, mapfn):
    mask = [predicate(x) for x in data]
    filtered = [x for x, m in zip(data, mask) if m]
    mapped = [mapfn(x) for x in filtered]
    return sum(mapped)
