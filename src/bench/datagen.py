import random
from array import array

__all__ = ["make_column"]


def make_column(n, seed=0):
    random.seed(seed)
    return array('d', [random.random() * 1000 for _ in range(n)])
