\# operator-fusion-engine



A lazy column engine that collapses `filter` / `map` / `reduce` pipelines into a

single fused pass over data, benchmarked against two obviously-correct baselines.



The point of the project is to demonstrate, in \~200 lines of stdlib-only Python,

why query engines (Catalyst, Polars, tinygrad) build a graph and then \*compile\*

it rather than executing each step eagerly. The benchmark shows exactly where

that pays off — and where it doesn't.



\## What it does



```python

from array import array

from fusion.column import Col



data = array('d', \[1.0, 2.0, 3.0, 4.0, 5.0])

result = Col.from\_array(data).filter(lambda x: x > 2).map(lambda x: x \* 10).sum()

\# 120.0 — computed in one pass, no intermediate arrays

