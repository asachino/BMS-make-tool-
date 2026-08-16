from __future__ import annotations

from ..similarity.score import SimilarityReport


def classify_report(
    report: SimilarityReport,
    *,
    threshold: float = 0.995,
    spectral_threshold: float = 0.92,
    gain_tolerance_db: float = 0.2,
) -> SimilarityReport:
    """Apply conservative MVP rules and attach a confidence estimate."""
    if report.overlap_warning:
        # Do not deduplicate a hit whose window contains neighbouring audio.
        report.classification = "OVERLAP"
        report.confidence = 0.0
        return report
    if report.raw_similarity >= threshold and abs(report.gain_db) < gain_tolerance_db:
        report.classification = "SAME"
        report.confidence = round(min(report.raw_similarity, report.spectral_similarity) * 100.0, 2)
    elif report.gain_normalized_similarity >= threshold and report.spectral_similarity >= spectral_threshold:
        report.classification = "GAIN_VARIANT"
        report.confidence = round(min(report.gain_normalized_similarity, report.spectral_similarity) * 100.0, 2)
    elif report.gain_normalized_similarity < threshold and report.spectral_similarity < spectral_threshold:
        report.classification = "DIFFERENT"
        report.confidence = round((1.0 - max(report.gain_normalized_similarity, report.spectral_similarity)) * 100.0, 2)
    else:
        report.classification = "UNSURE"
        report.confidence = round((report.gain_normalized_similarity + report.spectral_similarity) * 50.0, 2)
    return report
