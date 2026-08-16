"""Combined time and frequency similarity for two extracted hits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..extraction.alignment import align_pair
from ..features.spectral import spectral_similarity
from ..features.waveform import waveform_similarity


@dataclass
class SimilarityReport:
    reference_id: int
    candidate_id: int
    raw_similarity: float
    gain_normalized_similarity: float
    gain_db: float
    spectral_similarity: float
    attack_similarity: float
    body_similarity: float
    tail_similarity: float
    alignment_samples: int
    overlap_warning: bool = False
    classification: str = "UNSURE"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _window_similarity(reference, candidate, start: int, end: int) -> float:
    raw, normalized, _ = waveform_similarity(reference[start:end], candidate[start:end])
    return normalized


def compare_hits(reference, candidate, sample_rate: int, *, max_alignment_ms: float = 5.0) -> SimilarityReport:
    max_shift = max(0, round(sample_rate * max_alignment_ms / 1000.0))
    ref_samples, candidate_samples, shift = align_pair(reference.samples, candidate.samples, max_shift)
    raw, normalized, gain_db = waveform_similarity(ref_samples, candidate_samples)
    n = len(ref_samples)
    attack_end = min(n, max(1, round(sample_rate * 0.03)))
    body_end = min(n, max(attack_end, round(sample_rate * 0.2)))
    attack = _window_similarity(ref_samples, candidate_samples, 0, attack_end)
    body = _window_similarity(ref_samples, candidate_samples, attack_end, body_end)
    tail = _window_similarity(ref_samples, candidate_samples, body_end, n)
    spectral = spectral_similarity(ref_samples, candidate_samples)
    overlap = bool(getattr(reference, "overlap_warning", False) or getattr(candidate, "overlap_warning", False))
    return SimilarityReport(
        reference.id,
        candidate.id,
        raw,
        normalized,
        gain_db,
        spectral,
        attack,
        body,
        tail,
        shift,
        overlap,
    )
