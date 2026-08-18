from __future__ import annotations

from .envelope import envelope_features
from .percussion import percussion_features
from .spectral import spectral_features


def extract_features(signal, sample_rate: int, instrument: str = "kick") -> dict:
    result = envelope_features(signal, sample_rate)
    result.update(spectral_features(signal, sample_rate))
    result.update(percussion_features(signal, sample_rate, instrument))
    return result
