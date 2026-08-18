"""Small, deterministic feature profiles for common BMS percussion stems."""

from __future__ import annotations

import math

from .._numeric import np
from .spectral import _spectrum


SUPPORTED_INSTRUMENTS = ("kick", "snare", "hihat", "other")
INSTRUMENT_ALIASES = {
    "kick": "kick",
    "bd": "kick",
    "bassdrum": "kick",
    "snare": "snare",
    "sd": "snare",
    "clap": "snare",
    "hihat": "hihat",
    "hi-hat": "hihat",
    "hi_hat": "hihat",
    "hat": "hihat",
    "hh": "hihat",
    "other": "other",
    "misc": "other",
}

# The bands are intentionally broad.  They are shape descriptors, not a
# classifier trained on a particular sample pack.
INSTRUMENT_BANDS = {
    "kick": (20.0, 180.0, 1200.0),
    "snare": (120.0, 900.0, 4200.0),
    "hihat": (600.0, 4200.0, 9000.0),
    "other": (80.0, 1000.0, 6000.0),
}


def normalize_instrument(value: str | None) -> str:
    """Return a stable instrument identifier used by JSON and clustering."""
    text = str(value or "kick").strip().casefold().replace(" ", "")
    try:
        return INSTRUMENT_ALIASES[text]
    except KeyError as exc:
        choices = ", ".join(SUPPORTED_INSTRUMENTS)
        raise ValueError(f"instrument must be one of: {choices}") from exc


def _band_sum(spectrum, sample_rate: int, low: float, high: float) -> float:
    if spectrum is None or len(spectrum) == 0:
        return 0.0
    if np is not None and hasattr(spectrum, "shape"):
        frequencies = np.linspace(0.0, sample_rate / 2.0, len(spectrum))
        selected = spectrum[(frequencies >= low) & (frequencies < high)]
        return float(np.sum(selected)) if len(selected) else 0.0
    if len(spectrum) == 1:
        return float(spectrum[0]) if low <= 0.0 < high else 0.0
    total = 0.0
    for index, value in enumerate(spectrum):
        frequency = index * sample_rate / 2.0 / (len(spectrum) - 1)
        if low <= frequency < high:
            total += float(value)
    return total


def _zero_crossing_rate(signal) -> float:
    if np is not None and hasattr(signal, "shape"):
        return float(np.mean(np.abs(np.diff(np.signbit(signal))))) if len(signal) > 1 else 0.0
    return sum(1 for left, right in zip(signal, signal[1:]) if (left < 0) != (right < 0)) / max(1, len(signal) - 1)


def percussion_features(signal, sample_rate: int, instrument: str = "kick") -> dict[str, float | str]:
    """Extract broad instrument-aware descriptors from the existing FFT window.

    The returned keys are intentionally generic so saved analysis JSON can be
    reclustered by older code that simply ignores the additional values.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    canonical = normalize_instrument(instrument)
    shape = getattr(signal, "shape", None)
    if shape is not None and len(shape) > 1:
        signal = signal.mean(axis=1)
    elif shape is None and len(signal) and isinstance(signal[0], (list, tuple)):
        signal = [sum(float(value) for value in row) / max(1, len(row)) for row in signal]
    spectrum = _spectrum(signal)
    low_edge, mid_edge, high_edge = INSTRUMENT_BANDS[canonical]
    low = _band_sum(spectrum, sample_rate, 0.0, low_edge)
    mid = _band_sum(spectrum, sample_rate, low_edge, mid_edge)
    high = _band_sum(spectrum, sample_rate, mid_edge, high_edge)
    total = max(1e-12, low + mid + high + _band_sum(spectrum, sample_rate, high_edge, sample_rate / 2.0 + 1.0))

    n = len(signal)
    if n:
        split = max(1, n // 8)
        if np is not None and hasattr(signal, "shape"):
            absolute = np.abs(signal)
            early = float(np.sqrt(np.mean(signal[:split] * signal[:split]))) if split else 0.0
            whole = float(np.sqrt(np.mean(signal * signal)))
            late = float(np.sqrt(np.mean(signal[-split:] * signal[-split:]))) if split else 0.0
        else:
            values = [float(value) for value in signal]
            absolute = [abs(value) for value in values]
            early = math.sqrt(sum(value * value for value in values[:split]) / max(1, len(values[:split])))
            whole = math.sqrt(sum(value * value for value in values) / n)
            late = math.sqrt(sum(value * value for value in values[-split:]) / max(1, len(values[-split:])))
        peak = max((float(value) for value in absolute), default=0.0)
        transient_ratio = early / max(1e-6, whole)
        decay_ratio = late / max(1e-6, whole)
        peak_ratio = peak / max(1e-6, whole)
    else:
        transient_ratio = decay_ratio = peak_ratio = 0.0
    return {
        "instrument": canonical,
        "band_low_ratio": round(low / total, 8),
        "band_mid_ratio": round(mid / total, 8),
        "band_high_ratio": round(high / total, 8),
        "transient_ratio": round(transient_ratio, 8),
        "decay_ratio": round(decay_ratio, 8),
        "peak_to_rms": round(peak_ratio, 8),
        "percussion_zcr": round(_zero_crossing_rate(signal), 8) if n else 0.0,
    }


def instruments_compatible(left: str | None, right: str | None) -> bool:
    """Return whether two hits are allowed to share a reuse cluster."""
    try:
        return normalize_instrument(left) == normalize_instrument(right)
    except ValueError:
        return False


__all__ = [
    "INSTRUMENT_BANDS",
    "SUPPORTED_INSTRUMENTS",
    "instruments_compatible",
    "normalize_instrument",
    "percussion_features",
]
