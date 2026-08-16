"""Spectral features implemented with NumPy when present, DFT fallback otherwise."""

from __future__ import annotations

import math

from .._numeric import clip, dot, norm, np


def _spectrum(signal):
    n = len(signal)
    if not n:
        return []
    if np is not None:
        window = np.hanning(n)
        return np.abs(np.fft.rfft(signal * window))
    # Fallback keeps only a small, useful spectrum; it is for portability, not speed.
    bins = min(256, n // 2 + 1)
    return [
        math.sqrt(
            sum(float(signal[t]) * math.cos(2 * math.pi * k * t / n) for t in range(n)) ** 2
            + sum(float(signal[t]) * math.sin(2 * math.pi * k * t / n) for t in range(n)) ** 2
        )
        for k in range(bins)
    ]


def spectral_similarity(reference, candidate) -> float:
    """Cosine similarity of normalized log magnitude spectra."""
    left, right = _spectrum(reference), _spectrum(candidate)
    n = min(len(left), len(right))
    if not n:
        return 0.0
    if np is not None and hasattr(left, "shape"):
        left = np.log1p(left[:n])
        right = np.log1p(right[:n])
    else:
        left = [math.log1p(x) for x in left[:n]]
        right = [math.log1p(x) for x in right[:n]]
    left_norm = norm(left)
    right_norm = norm(right)
    if not left_norm or not right_norm:
        return 1.0 if not left_norm and not right_norm else 0.0
    return clip(dot(left, right) / (left_norm * right_norm), 0.0, 1.0)


def spectral_features(signal, sample_rate: int) -> dict[str, float]:
    spectrum = _spectrum(signal)
    if len(spectrum) == 0:
        return {"centroid_hz": 0.0, "rolloff_hz": 0.0, "zcr": 0.0}
    if np is not None and hasattr(spectrum, "shape"):
        weights = spectrum
        frequencies = np.linspace(0.0, sample_rate / 2.0, len(spectrum))
        total = float(np.sum(weights))
        centroid = float(np.sum(frequencies * weights) / total) if total else 0.0
        cumulative = np.cumsum(weights)
        rolloff = float(frequencies[min(len(frequencies) - 1, int(np.searchsorted(cumulative, total * 0.85)))]) if total else 0.0
    else:
        total = sum(spectrum)
        centroid = sum(i * sample_rate / 2 / max(1, len(spectrum) - 1) * w for i, w in enumerate(spectrum)) / total if total else 0.0
        running = 0.0
        rolloff = 0.0
        for i, weight in enumerate(spectrum):
            running += weight
            if running >= total * 0.85:
                rolloff = i * sample_rate / 2 / max(1, len(spectrum) - 1)
                break
    if np is not None and hasattr(signal, "shape"):
        zcr = float(np.mean(np.abs(np.diff(np.signbit(signal))))) if len(signal) > 1 else 0.0
    else:
        zcr = sum(1 for a, b in zip(signal, signal[1:]) if (a < 0) != (b < 0)) / max(1, len(signal) - 1)
    return {"centroid_hz": centroid, "rolloff_hz": rolloff, "zcr": zcr}
