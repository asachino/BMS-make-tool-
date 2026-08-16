"""Time-domain similarity and gain estimation."""

from __future__ import annotations

import math

from .._numeric import clip, dot, norm


def gain_estimate(reference, candidate) -> float:
    denominator = dot(reference, reference)
    return dot(reference, candidate) / denominator if denominator > 1e-15 else 0.0


def _similarity_from_error(error: float, scale: float) -> float:
    return clip(1.0 - error / max(scale, 1e-12), 0.0, 1.0)


def waveform_similarity(reference, candidate) -> tuple[float, float, float]:
    """Return raw similarity, gain-normalized similarity and gain in dB."""
    if not len(reference) or not len(candidate):
        return 0.0, 0.0, 0.0
    scale = max(norm(reference), norm(candidate))
    raw = _similarity_from_error(norm(candidate - reference), scale) if hasattr(candidate, "shape") else _similarity_from_error(norm([y - x for x, y in zip(reference, candidate)]), scale)
    gain = gain_estimate(reference, candidate)
    if gain <= 0.0:
        # Keep project JSON strict; polarity-inverted or silent candidates are
        # deliberately not eligible for SAME/GAIN_VARIANT.
        return raw, 0.0, -120.0
    if hasattr(reference, "shape"):
        residual = candidate - gain * reference
    else:
        residual = [y - gain * x for x, y in zip(reference, candidate)]
    normalized = _similarity_from_error(norm(residual), norm(candidate))
    return raw, normalized, 20.0 * math.log10(gain)
