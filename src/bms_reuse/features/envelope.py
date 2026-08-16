"""Amplitude envelope summaries used in reports."""

from __future__ import annotations

import math

from .._numeric import np


def envelope_features(signal, sample_rate: int) -> dict[str, float]:
    if not len(signal):
        return {"peak_db": -120.0, "rms_db": -120.0, "attack_ms": 0.0, "tail_energy": 0.0}
    if np is not None and hasattr(signal, "shape"):
        absolute = np.abs(signal)
        peak = float(np.max(absolute))
        rms = float(np.sqrt(np.mean(signal * signal)))
        attack_index = int(np.argmax(absolute >= peak * 0.9)) if peak else 0
        tail_start = int(len(signal) * 0.25)
        tail_energy = float(np.sqrt(np.mean(signal[tail_start:] ** 2))) if tail_start < len(signal) else 0.0
    else:
        absolute = [abs(float(x)) for x in signal]
        peak = max(absolute)
        rms = math.sqrt(sum(float(x) ** 2 for x in signal) / len(signal))
        attack_index = next((i for i, x in enumerate(absolute) if x >= peak * 0.9), 0) if peak else 0
        tail = signal[len(signal) // 4 :]
        tail_energy = math.sqrt(sum(float(x) ** 2 for x in tail) / len(tail)) if tail else 0.0
    db = lambda value: 20.0 * math.log10(max(value, 1e-6))
    return {
        "peak_db": db(peak),
        "rms_db": db(rms),
        "attack_ms": attack_index * 1000.0 / sample_rate,
        "tail_energy": tail_energy,
    }
