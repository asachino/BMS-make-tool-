from __future__ import annotations

from ..similarity.score import SimilarityReport


SIMILARITY_PROFILE_NAME = "waveform_spectral_v2"
DEFAULT_WAVEFORM_THRESHOLD = 0.95
DEFAULT_SPECTRAL_THRESHOLD = 0.94
DEFAULT_GAIN_TOLERANCE_DB = 0.25


def classify_report(
    report: SimilarityReport,
    *,
    threshold: float = DEFAULT_WAVEFORM_THRESHOLD,
    spectral_threshold: float = DEFAULT_SPECTRAL_THRESHOLD,
    gain_tolerance_db: float = DEFAULT_GAIN_TOLERANCE_DB,
) -> SimilarityReport:
    """Classify by aligned shape and spectrum, retaining overlap as a warning.

    ``raw_similarity`` remains part of every report for diagnostics, but the
    gain-normalized waveform and log-spectrum cosine are the primary decision
    signals.  An overlap warning no longer vetoes a clearly matching sound.
    """
    waveform = float(report.gain_normalized_similarity)
    spectral = float(report.spectral_similarity)
    similar = waveform >= threshold and spectral >= spectral_threshold
    if similar and abs(report.gain_db) < gain_tolerance_db:
        report.classification = "SAME"
        report.confidence = round(min(waveform, spectral) * 100.0, 2)
    elif similar:
        report.classification = "GAIN_VARIANT"
        report.confidence = round(min(waveform, spectral) * 100.0, 2)
    elif report.overlap_warning:
        # Keep the existing review state for an overlapped, non-matching
        # window; a matching window was handled above.
        report.classification = "OVERLAP"
        report.confidence = 0.0
    elif waveform < threshold and spectral < spectral_threshold:
        report.classification = "DIFFERENT"
        report.confidence = round((1.0 - max(waveform, spectral)) * 100.0, 2)
    else:
        report.classification = "UNSURE"
        report.confidence = round((waveform + spectral) * 50.0, 2)
    return report
