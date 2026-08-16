"""Small numeric helpers; NumPy is an accelerator, not a hard requirement."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

try:  # pragma: no cover - both paths are intentionally supported
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def array(values, dtype=None):
    return np.asarray(values, dtype=dtype) if np is not None else values


def to_list(values) -> list[float]:
    return values.tolist() if np is not None and hasattr(values, "tolist") else list(values)


def length(values) -> int:
    return int(values.shape[0]) if np is not None and hasattr(values, "shape") else len(values)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def dot(a, b) -> float:
    if np is not None and hasattr(a, "dot"):
        return float(np.dot(a, b))
    return sum(float(x) * float(y) for x, y in zip(a, b))


def norm(a) -> float:
    if np is not None and hasattr(a, "dot"):
        return float(np.linalg.norm(a))
    return math.sqrt(dot(a, a))


def rms(a) -> float:
    n = length(a)
    return math.sqrt(dot(a, a) / n) if n else 0.0


def mean(a) -> float:
    n = length(a)
    return sum(float(x) for x in a) / n if n else 0.0


def subtract(a, b):
    if np is not None and hasattr(a, "shape"):
        return a - b
    return [x - y for x, y in zip(a, b)]


def multiply(a, value: float):
    if np is not None and hasattr(a, "shape"):
        return a * value
    return [x * value for x in a]


def zeros(n: int):
    return np.zeros(n, dtype=float) if np is not None else [0.0] * n


def pad_or_trim(values, n: int):
    if np is not None and hasattr(values, "shape"):
        if len(values) >= n:
            return values[:n]
        return np.pad(values, (0, n - len(values)))
    values = list(values[:n])
    return values + [0.0] * (n - len(values))


def max_abs(values) -> float:
    if np is not None and hasattr(values, "shape"):
        return float(np.max(np.abs(values))) if len(values) else 0.0
    return max((abs(float(x)) for x in values), default=0.0)


def abs_values(values):
    if np is not None and hasattr(values, "shape"):
        return np.abs(values)
    return [abs(float(x)) for x in values]


def slice_values(values, start: int, end: int):
    return values[start:end]


def as_float_list(values) -> list[float]:
    if np is not None and hasattr(values, "tolist"):
        return [float(x) for x in values.tolist()]
    return [float(x) for x in values]
