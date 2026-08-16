from __future__ import annotations

from .envelope import envelope_features
from .spectral import spectral_features


def extract_features(signal, sample_rate: int) -> dict[str, float]:
    result = envelope_features(signal, sample_rate)
    result.update(spectral_features(signal, sample_rate))
    return result
