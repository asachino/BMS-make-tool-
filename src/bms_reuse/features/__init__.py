from .waveform import gain_estimate, waveform_similarity
from .spectral import spectral_similarity
from .automation import detect_automation, detect_automation_changes
from .percussion import SUPPORTED_INSTRUMENTS, normalize_instrument, percussion_features

__all__ = [
    "gain_estimate", "waveform_similarity", "spectral_similarity",
    "detect_automation", "detect_automation_changes", "SUPPORTED_INSTRUMENTS", "normalize_instrument", "percussion_features",
]
