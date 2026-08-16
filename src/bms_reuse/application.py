"""Application-level MVP pipeline."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .classification.classifier import classify_report
from .clustering.reuse_plan import ReusePlan, build_reuse_plan
from .detection.onset import detect_onsets
from .extraction.hit_extractor import extract_hits
from .features.feature_extractor import extract_features
from .project.model import Project
from .similarity.score import SimilarityReport, compare_hits
from .audio.loader import load_audio, mono_signal


@dataclass
class AnalysisResult:
    source: str
    sample_rate: int
    duration: float
    hits: list
    comparisons: list[SimilarityReport]
    plan: ReusePlan
    settings: dict
    source_hash: str

    @property
    def summary(self) -> dict:
        counts = Counter(report.classification for report in self.comparisons)
        return {
            "duration_seconds": round(self.duration, 6),
            "detected_hits": len(self.hits),
            "same": counts.get("SAME", 0),
            "gain_variants": counts.get("GAIN_VARIANT", 0),
            "different": counts.get("DIFFERENT", 0),
            "unsure": counts.get("UNSURE", 0),
            "overlap": counts.get("OVERLAP", 0),
            "required_samples": self.plan.required_samples,
            "reuse_ratio": round((1.0 - self.plan.required_samples / len(self.hits)) * 100.0, 2) if self.hits else 0.0,
        }

    def to_dict(self) -> dict:
        project = Project(
            source=self.source,
            settings=self.settings,
            hits=[hit.to_dict() for hit in self.hits],
            comparisons=[report.to_dict() for report in self.comparisons],
            clusters=[cluster.to_dict() for cluster in self.plan.clusters],
            reuse_plan=self.plan.to_dict(),
        )
        data = project.to_dict()
        data.update({"source_hash": self.source_hash, "sample_rate": self.sample_rate, "duration": self.duration, "summary": self.summary})
        return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_file(
    path: str | Path,
    *,
    instrument: str = "kick",
    threshold: float = 0.995,
    spectral_threshold: float = 0.92,
    onset_threshold: float = 0.35,
    min_separation_ms: float = 50.0,
    pre_roll_ms: float = 5.0,
    window_ms: float = 800.0,
    max_alignment_ms: float = 5.0,
    bpm: float | None = None,
    offset: float = 0.0,
    subdivision: int = 16,
) -> AnalysisResult:
    path = Path(path)
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= spectral_threshold <= 1.0:
        raise ValueError("threshold and spectral_threshold must be between 0 and 1")
    if onset_threshold < 0 or min_separation_ms <= 0 or pre_roll_ms < 0 or window_ms <= 0 or max_alignment_ms < 0:
        raise ValueError("analysis timing values are out of range")
    if bpm is not None and bpm <= 0:
        raise ValueError("bpm must be positive")
    if subdivision <= 0:
        raise ValueError("subdivision must be positive")
    audio = load_audio(path)
    mono = mono_signal(audio)
    onsets = detect_onsets(
        mono,
        audio.sample_rate,
        threshold=onset_threshold,
        min_separation_ms=min_separation_ms,
        bpm=bpm,
        offset=offset,
        subdivision=subdivision,
    )
    hits = extract_hits(audio, onsets, pre_roll_ms=pre_roll_ms, window_ms=window_ms)
    for hit in hits:
        hit.features = extract_features(hit.samples, audio.sample_rate)

    def compare(reference, candidate):
        return compare_hits(reference, candidate, audio.sample_rate, max_alignment_ms=max_alignment_ms)

    plan, comparisons = build_reuse_plan(
        hits,
        compare,
        threshold=threshold,
        spectral_threshold=spectral_threshold,
    )
    settings = {
        "instrument": instrument,
        "threshold": threshold,
        "spectral_threshold": spectral_threshold,
        "onset_threshold": onset_threshold,
        "min_separation_ms": min_separation_ms,
        "pre_roll_ms": pre_roll_ms,
        "window_ms": window_ms,
        "max_alignment_ms": max_alignment_ms,
        "bpm": bpm,
        "offset": offset,
        "subdivision": subdivision,
    }
    return AnalysisResult(str(path), audio.sample_rate, audio.duration, hits, comparisons, plan, settings, _sha256(path))
