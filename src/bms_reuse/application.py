"""Application-level MVP pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .classification.classifier import (
    DEFAULT_GAIN_TOLERANCE_DB,
    DEFAULT_SPECTRAL_THRESHOLD,
    DEFAULT_WAVEFORM_THRESHOLD,
    SIMILARITY_PROFILE_NAME,
)
from .clustering.reuse_plan import ReusePlan, build_reuse_plan
from .detection.onset import BPM_SNAP_TOLERANCE_MS, detect_onsets
from .extraction.hit_extractor import extract_hits
from .features.feature_extractor import extract_features
from .project.model import Project
from .similarity.score import SimilarityReport, compare_hits
from .audio.loader import load_audio, mono_signal


ANALYSIS_VERSION = "0.3.0"


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
            "overlap_warnings": sum(bool(report.overlap_warning) for report in self.comparisons),
            "required_samples": self.plan.required_samples,
            "reuse_ratio": round((1.0 - self.plan.required_samples / len(self.hits)) * 100.0, 2) if self.hits else 0.0,
            "comparisons": len(self.comparisons),
            "comparison_cache_hits": int(self.settings.get("comparison_cache_hits", 0)),
            "compare_mode": self.settings.get("compare_mode", "normal"),
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
        settings = self.settings
        repro = settings.get("reproducibility", {}) if isinstance(settings, dict) else {}
        grid = {
            "bpm": settings.get("bpm"),
            "offset": settings.get("offset", 0.0),
            "subdivision": settings.get("subdivision", 16),
            "beat_division": settings.get("beat_division"),
            "margin_percent": settings.get("margin_percent"),
            "min_interval_sec": settings.get("min_interval_sec"),
        }
        data.update({
            "schema_version": 2,
            "source_hash": self.source_hash,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "analysis_version": ANALYSIS_VERSION,
            "metadata": {
                "source_hash": self.source_hash,
                "analysis_version": ANALYSIS_VERSION,
                "reproducibility_hash": repro.get("reproducibility_hash", settings.get("settings_hash", "")),
                "settings_hash": repro.get("settings_hash", settings.get("settings_hash", "")),
                "grid": grid,
            },
            "review": {
                "overrides": settings.get("review_overrides", {}),
                "targets": settings.get("review_targets", {}),
                "excluded_hits": settings.get("excluded_hits", []),
            },
            "validation": settings.get("validation", {}),
            "exports": settings.get("exports", {}),
            "summary": self.summary,
        })
        return data


class AnalysisCancelled(Exception):
    """Raised when an optional GUI cancellation request is observed."""


def _sample_fingerprint(samples) -> tuple[tuple[int, ...], str, bytes]:
    """Build an internal shape/dtype/bytes fingerprint without changing JSON."""
    shape_value = getattr(samples, "shape", None)
    shape = tuple(int(value) for value in shape_value) if shape_value is not None else (len(samples),)
    dtype = str(getattr(samples, "dtype", "python"))
    if hasattr(samples, "tobytes"):
        raw = samples.tobytes()
    else:
        raw = repr(tuple(float(value) for value in samples)).encode("utf-8")
    return shape, dtype, hashlib.sha256(raw).digest()


def _samples_equal(left, right) -> bool:
    """Verify a digest match using the exact sample representation."""
    left_shape_value = getattr(left, "shape", None)
    right_shape_value = getattr(right, "shape", None)
    left_shape = tuple(int(value) for value in left_shape_value) if left_shape_value is not None else (len(left),)
    right_shape = tuple(int(value) for value in right_shape_value) if right_shape_value is not None else (len(right),)
    if left_shape != right_shape or str(getattr(left, "dtype", "python")) != str(getattr(right, "dtype", "python")):
        return False
    if hasattr(left, "tobytes") and hasattr(right, "tobytes"):
        return left.tobytes() == right.tobytes()
    return list(left) == list(right)


def record_output_timing(result: AnalysisResult, output_seconds: float) -> None:
    """Attach export timing after the caller finishes writing artifacts."""
    timings = result.settings.setdefault("timings", {})
    output_seconds = max(0.0, float(output_seconds))
    timings["output_seconds"] = round(output_seconds, 6)
    timings["total_seconds"] = round(float(timings.get("analysis_seconds", 0.0)) + output_seconds, 6)


def refresh_reproducibility(result: AnalysisResult) -> None:
    """Recompute review-aware hashes after a human changes the plan."""
    settings = result.settings
    stable_settings = {
        key: value
        for key, value in settings.items()
        if key not in {
            "timings", "comparison_count", "comparison_cache_hits", "comparison_cache_entries",
            "settings_hash", "reproducibility_hash", "reproducibility", "validation", "exports",
        }
    }
    canonical = json.dumps(stable_settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    settings_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    reproducibility_hash = hashlib.sha256(f"{result.source_hash}:{settings_hash}:{ANALYSIS_VERSION}".encode("utf-8")).hexdigest()
    settings["settings_hash"] = settings_hash
    settings["reproducibility_hash"] = reproducibility_hash
    settings["reproducibility"] = {
        "source_hash": result.source_hash,
        "settings_hash": settings_hash,
        "reproducibility_hash": reproducibility_hash,
        "analysis_version": ANALYSIS_VERSION,
    }


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def exclude_hit(result: AnalysisResult, hit_id: int) -> None:
    """Remove one reviewed hit from the reusable plan and all exports."""
    hit_id = int(hit_id)
    result.hits[:] = [hit for hit in result.hits if hit.id != hit_id]
    result.comparisons[:] = [
        report for report in result.comparisons
        if report.reference_id != hit_id and report.candidate_id != hit_id
    ]
    retained_clusters = []
    old_cluster_ids: dict[int, int] = {}
    for cluster in result.plan.clusters:
        cluster.hit_ids[:] = [value for value in cluster.hit_ids if value != hit_id]
        if cluster.hit_ids:
            if cluster.representative_hit == hit_id:
                cluster.representative_hit = cluster.hit_ids[0]
            retained_clusters.append(cluster)
    result.plan.clusters[:] = retained_clusters
    for index, cluster in enumerate(result.plan.clusters, 1):
        old_cluster_ids[int(cluster.id)] = index
        cluster.id = index
    targets = result.settings.get("review_targets", {})
    if isinstance(targets, dict):
        result.settings["review_targets"] = {
            str(target_hit): old_cluster_ids[int(target_cluster)]
            for target_hit, target_cluster in targets.items()
            if str(target_hit) != str(hit_id)
            and _safe_int(target_cluster) in old_cluster_ids
        }
    cluster_by_hit = {
        hit: cluster.id
        for cluster in result.plan.clusters
        for hit in cluster.hit_ids
    }
    result.plan.events[:] = [
        dict(event, sample_id=f"sample_{cluster_by_hit[event['hit']]:03d}")
        for event in result.plan.events
        if int(event.get("hit", -1)) != hit_id and int(event.get("hit", -1)) in cluster_by_hit
    ]
    excluded = result.settings.setdefault("excluded_hits", [])
    if hit_id not in excluded:
        excluded.append(hit_id)
    overrides = result.settings.setdefault("review_overrides", {})
    overrides[str(hit_id)] = "I"
    refresh_reproducibility(result)


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
    threshold: float = DEFAULT_WAVEFORM_THRESHOLD,
    spectral_threshold: float = DEFAULT_SPECTRAL_THRESHOLD,
    onset_threshold: float = 0.35,
    min_separation_ms: float = 50.0,
    pre_roll_ms: float = 5.0,
    window_ms: float = 800.0,
    max_alignment_ms: float = 20.0,
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
    fast_compare: bool = False,
    bms_channel: str = "01",
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
    if len(str(bms_channel)) != 2 or not str(bms_channel).isdigit():
        raise ValueError("bms_channel must be a two-digit decimal channel")
    bms_channel = str(bms_channel).upper()
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

    fingerprints = {hit.id: _sample_fingerprint(hit.samples) for hit in hits}
    exact_report_cache: dict[tuple[tuple[int, ...], str, bytes, bool], list[tuple[object, SimilarityReport]]] = {}
    cache_stats = {"hits": 0, "entries": 0}

    def compare(reference, candidate):
        reference_fingerprint = fingerprints[reference.id]
        candidate_fingerprint = fingerprints[candidate.id]
        overlap = bool(getattr(reference, "overlap_warning", False) or getattr(candidate, "overlap_warning", False))
        if reference_fingerprint == candidate_fingerprint:
            key = reference_fingerprint + (overlap,)
            for cached_samples, cached_report in exact_report_cache.get(key, []):
                if _samples_equal(cached_samples, candidate.samples) and _samples_equal(cached_samples, reference.samples):
                    cache_stats["hits"] += 1
                    return replace(cached_report, reference_id=reference.id, candidate_id=candidate.id)
        report = compare_hits(reference, candidate, audio.sample_rate, max_alignment_ms=max_alignment_ms)
        if reference_fingerprint == candidate_fingerprint:
            key = reference_fingerprint + (overlap,)
            exact_report_cache.setdefault(key, []).append((reference.samples, replace(report)))
            cache_stats["entries"] += 1
        return report

    report(52, "Comparing and clustering hits")
    stage_started = time.perf_counter()
    last_compare_report = [0.0]

    def report_compare_detail(current: int, total: int, compared: int) -> None:
        now = time.perf_counter()
        if compared == 1 or current == total or now - last_compare_report[0] >= 0.15:
            last_compare_report[0] = now
            report(
                52 + round(current / max(1, total) * 43),
                f"Comparing and clustering hits {current}/{total} ({compared} comparisons, {cache_stats['hits']} cache hits)",
            )

    plan, comparisons = build_reuse_plan(
        hits,
        compare,
        threshold=threshold,
        spectral_threshold=spectral_threshold,
        progress=lambda done, total: report(52 + round(done / max(1, total) * 43), f"Comparing and clustering hits {done}/{total}"),
        progress_detail=report_compare_detail,
        is_cancelled=is_cancelled,
        fast_compare=fast_compare,
        reuse_key=lambda hit: fingerprints[hit.id] + (bool(getattr(hit, "overlap_warning", False)),),
        reuse_equal=lambda left, right: _samples_equal(left.samples, right.samples) and bool(getattr(left, "overlap_warning", False)) == bool(getattr(right, "overlap_warning", False)),
        cache_hit=lambda: cache_stats.__setitem__("hits", cache_stats["hits"] + 1),
    )
    timings["compare_seconds"] = round(time.perf_counter() - stage_started, 6)
    timings["analysis_seconds"] = round(time.perf_counter() - analysis_started, 6)
    timings["total_seconds"] = timings["analysis_seconds"]
    if not audio.frame_count:
        for key in timings:
            timings[key] = 0.0
    source_hash = _sha256(path)
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
        "bpm_snap_tolerance_ms": BPM_SNAP_TOLERANCE_MS,
        "bpm": bpm,
        "offset": offset,
        "subdivision": subdivision,
        "fade_in_ms": fade_in_ms,
        "fade_out_ms": fade_out_ms,
        "bms_channel": bms_channel,
        "similarity_profile": {
            "name": SIMILARITY_PROFILE_NAME,
            "waveform_threshold": threshold,
            "spectral_threshold": spectral_threshold,
            "gain_tolerance_db": DEFAULT_GAIN_TOLERANCE_DB,
            "alignment_ms": max_alignment_ms,
            "raw_similarity_role": "auxiliary",
            "overlap_policy": "similarity_first_warning",
        },
        "compare_mode": "fast" if fast_compare else "normal",
        "fast_compare": bool(fast_compare),
        "comparison_count": len(comparisons),
        "comparison_cache_hits": cache_stats["hits"],
        "comparison_cache_entries": cache_stats["entries"],
        "timings": timings,
        "analysis_version": ANALYSIS_VERSION,
    }
    stable_settings = {
        key: value
        for key, value in settings.items()
        if key not in {"timings", "comparison_count", "comparison_cache_hits", "comparison_cache_entries"}
    }
    canonical_settings = json.dumps(stable_settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    settings["settings_hash"] = hashlib.sha256(canonical_settings.encode("utf-8")).hexdigest()
    settings["reproducibility_hash"] = hashlib.sha256(
        f"{source_hash}:{settings['settings_hash']}:{ANALYSIS_VERSION}".encode("utf-8")
    ).hexdigest()
    settings["reproducibility"] = {
        "source_hash": source_hash,
        "settings_hash": settings["settings_hash"],
        "reproducibility_hash": settings["reproducibility_hash"],
        "analysis_version": ANALYSIS_VERSION,
    }
    report(100, "Analysis complete")
    return AnalysisResult(str(path), audio.sample_rate, audio.duration, hits, comparisons, plan, settings, source_hash)
