"""Application-level MVP pipeline."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
            "timings": dict(self.settings.get("timings", {})),
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


class AnalysisCancelled(Exception):
    """Raised when an optional GUI cancellation request is observed."""


def record_output_timing(result: AnalysisResult, output_seconds: float) -> None:
    """Attach export timing after the caller finishes writing artifacts."""
    timings = result.settings.setdefault("timings", {})
    output_seconds = max(0.0, float(output_seconds))
    timings["output_seconds"] = round(output_seconds, 6)
    timings["total_seconds"] = round(float(timings.get("analysis_seconds", 0.0)) + output_seconds, 6)


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
    progress: Callable[[int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    min_interval_sec: float | None = None,
    beat_division: int | None = None,
    margin_percent: float | None = None,
    margin: float | None = None,
    fade_in_ms: float = 0.0,
    fade_out_ms: float = 0.0,
) -> AnalysisResult:
    analysis_started = time.perf_counter()

    def report(percent: int, message: str) -> None:
        if is_cancelled and is_cancelled():
            raise AnalysisCancelled()
        if progress:
            progress(percent, message)

    path = Path(path)
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= spectral_threshold <= 1.0:
        raise ValueError("threshold and spectral_threshold must be between 0 and 1")
    if min_interval_sec is not None:
        if min_interval_sec <= 0:
            raise ValueError("min_interval_sec must be positive")
        min_separation_ms = min_interval_sec * 1000.0
    if margin is not None:
        if margin_percent is not None and margin_percent != margin:
            raise ValueError("margin and margin_percent must match")
        margin_percent = margin
    if beat_division is not None and beat_division <= 0:
        raise ValueError("beat_division must be positive")
    if margin_percent is not None and not 0.0 < margin_percent <= 100.0:
        raise ValueError("margin_percent must be between 0 and 100")
    if fade_in_ms < 0 or fade_out_ms < 0:
        raise ValueError("fade durations must be non-negative")
    if onset_threshold < 0 or min_separation_ms <= 0 or pre_roll_ms < 0 or window_ms <= 0 or max_alignment_ms < 0:
        raise ValueError("analysis timing values are out of range")
    if bpm is not None and bpm <= 0:
        raise ValueError("bpm must be positive")
    if subdivision <= 0:
        raise ValueError("subdivision must be positive")
    timings = {
        "load_seconds": 0.0,
        "onset_seconds": 0.0,
        "hit_seconds": 0.0,
        "feature_seconds": 0.0,
        "compare_seconds": 0.0,
        "output_seconds": 0.0,
        "analysis_seconds": 0.0,
        "total_seconds": 0.0,
    }
    stage_started = time.perf_counter()
    report(5, "Loading audio")
    audio = load_audio(path)
    mono = mono_signal(audio)
    timings["load_seconds"] = round(time.perf_counter() - stage_started, 6)
    stage_started = time.perf_counter()
    report(18, "Detecting onsets")
    onsets = detect_onsets(
        mono,
        audio.sample_rate,
        threshold=onset_threshold,
        min_separation_ms=min_separation_ms,
        bpm=bpm,
        offset=offset,
        subdivision=subdivision,
    )
    timings["onset_seconds"] = round(time.perf_counter() - stage_started, 6)
    stage_started = time.perf_counter()
    report(28, f"Extracting {len(onsets)} hits")
    report(28, f"Extracting hits 0/{len(onsets)}")
    hits = extract_hits(
        audio,
        onsets,
        pre_roll_ms=pre_roll_ms,
        window_ms=window_ms,
        progress=lambda done, total: report(28 + round(done / max(1, total) * 2), f"Extracting hits {done}/{total}"),
        is_cancelled=is_cancelled,
    )
    timings["hit_seconds"] = round(time.perf_counter() - stage_started, 6)
    stage_started = time.perf_counter()
    if hits:
        report(30, f"Extracting features 0/{len(hits)}")
    for index, hit in enumerate(hits):
        if is_cancelled and is_cancelled():
            raise AnalysisCancelled()
        hit.features = extract_features(hit.samples, audio.sample_rate)
        if hits:
            report(30 + round((index + 1) / len(hits) * 20), f"Extracting features {index + 1}/{len(hits)}")
    timings["feature_seconds"] = round(time.perf_counter() - stage_started, 6)

    def compare(reference, candidate):
        return compare_hits(reference, candidate, audio.sample_rate, max_alignment_ms=max_alignment_ms)

    report(52, "Comparing and clustering hits")
    stage_started = time.perf_counter()
    last_compare_report = [0.0]

    def report_compare_detail(current: int, total: int, compared: int) -> None:
        now = time.perf_counter()
        if compared == 1 or current == total or now - last_compare_report[0] >= 0.15:
            last_compare_report[0] = now
            report(
                52 + round(current / max(1, total) * 43),
                f"Comparing and clustering hits {current}/{total} ({compared} comparisons)",
            )

    plan, comparisons = build_reuse_plan(
        hits,
        compare,
        threshold=threshold,
        spectral_threshold=spectral_threshold,
        progress=lambda done, total: report(52 + round(done / max(1, total) * 43), f"Comparing and clustering hits {done}/{total}"),
        progress_detail=report_compare_detail,
        is_cancelled=is_cancelled,
    )
    timings["compare_seconds"] = round(time.perf_counter() - stage_started, 6)
    timings["analysis_seconds"] = round(time.perf_counter() - analysis_started, 6)
    timings["total_seconds"] = timings["analysis_seconds"]
    if not audio.frame_count:
        for key in timings:
            timings[key] = 0.0
    settings = {
        "instrument": instrument,
        "threshold": threshold,
        "spectral_threshold": spectral_threshold,
        "onset_threshold": onset_threshold,
        "min_separation_ms": min_separation_ms,
        "min_interval_sec": min_separation_ms / 1000.0,
        "beat_division": beat_division,
        "margin_percent": margin_percent,
        "margin": margin_percent,
        "pre_roll_ms": pre_roll_ms,
        "window_ms": window_ms,
        "max_alignment_ms": max_alignment_ms,
        "bpm": bpm,
        "offset": offset,
        "subdivision": subdivision,
        "fade_in_ms": fade_in_ms,
        "fade_out_ms": fade_out_ms,
        "timings": timings,
    }
    report(100, "Analysis complete")
    return AnalysisResult(str(path), audio.sample_rate, audio.duration, hits, comparisons, plan, settings, _sha256(path))
