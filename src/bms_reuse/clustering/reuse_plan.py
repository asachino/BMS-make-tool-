"""Conservative sequential clustering and the resulting reuse mapping."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

from ..classification.classifier import (
    DEFAULT_SPECTRAL_THRESHOLD,
    DEFAULT_WAVEFORM_THRESHOLD,
    classify_report,
)
from ..features.percussion import instruments_compatible
from ..similarity.score import SimilarityReport


_FEATURE_KEYS = ("centroid_hz", "rolloff_hz", "zcr", "attack_ms")
_OPTIONAL_FEATURE_KEYS = (
    "band_low_ratio", "band_mid_ratio", "band_high_ratio",
    "transient_ratio", "decay_ratio", "percussion_zcr",
)


def _gain_invariant_features(features: dict) -> tuple[float, ...] | None:
    """Return shape-oriented features without using absolute level."""
    if not features or any(key not in features for key in _FEATURE_KEYS):
        return None
    try:
        rms_db = float(features.get("rms_db", -120.0))
        rms_amplitude = max(1e-6, 10.0 ** (rms_db / 20.0))
        tail_ratio = float(features.get("tail_energy", 0.0)) / rms_amplitude
        values = tuple(float(features[key]) for key in _FEATURE_KEYS) + (tail_ratio,)
        values += tuple(float(features.get(key, 0.0)) for key in _OPTIONAL_FEATURE_KEYS)
    except (TypeError, ValueError, OverflowError):
        return None
    return values if all(value == value and abs(value) != float("inf") for value in values) else None


def _feature_distance(left, right) -> float:
    """Rank representatives by a scale-normalized, gain-invariant distance."""
    if not instruments_compatible(getattr(left, "instrument", "kick"), getattr(right, "instrument", "kick")):
        return float("inf")
    left_values = _gain_invariant_features(getattr(left, "features", {}))
    right_values = _gain_invariant_features(getattr(right, "features", {}))
    if left_values is None or right_values is None:
        return float("inf")
    return sum(
        ((left_value - right_value) / max(1.0, abs(left_value), abs(right_value))) ** 2
        for left_value, right_value in zip(left_values, right_values)
    )


@dataclass
class Cluster:
    id: int
    representative_hit: int
    hit_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "representative_hit": self.representative_hit,
            "hit_ids": self.hit_ids,
        }


@dataclass
class ReusePlan:
    clusters: list[Cluster]
    events: list[dict]

    @property
    def required_samples(self) -> int:
        return len(self.clusters)

    def to_dict(self) -> dict:
        return {
            "samples": [
                {"sample_id": f"sample_{cluster.id:03d}", "source_hit": cluster.representative_hit}
                for cluster in self.clusters
            ],
            "events": self.events,
        }


def _make_plan(hits, assignments: dict[int, tuple[int, float]], clusters: list[Cluster]) -> ReusePlan:
    events = [
        {
            "hit": hit.id,
            "time": hit.time,
            "sample_id": f"sample_{assignments[hit.id][0]:03d}",
            "gain_db": round(assignments[hit.id][1], 4),
        }
        for hit in hits
    ]
    return ReusePlan(clusters, events)


def build_reuse_plan(
    hits,
    compare: Callable | Iterable[SimilarityReport],
    *,
    threshold: float = DEFAULT_WAVEFORM_THRESHOLD,
    spectral_threshold: float = DEFAULT_SPECTRAL_THRESHOLD,
    progress: Callable[[int, int], None] | None = None,
    progress_detail: Callable[[int, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    fast_compare: bool = False,
    reuse_key: Callable[[object], object] | None = None,
    reuse_equal: Callable[[object, object], bool] | None = None,
    cache_hit: Callable[[], None] | None = None,
) -> tuple[ReusePlan, list[SimilarityReport]]:
    """Group hits against existing representatives; return plan and comparisons.

    A sequential representative pass is intentional: it keeps the MVP fast and
    deterministic for ordinary stems. A future large-batch mode can replace it
    with indexed clustering without changing the output model.
    """
    by_id = {hit.id: hit for hit in hits}
    clusters: list[Cluster] = []
    assignments: dict[int, tuple[int, float]] = {}
    reports: list[SimilarityReport] = []
    comparison_cache: dict[tuple[int, int], SimilarityReport] = {}
    reuse_cache: dict[object, list[dict]] = {}

    def find_reuse_entry(hit):
        if reuse_key is None:
            return None
        key = reuse_key(hit)
        if key is None:
            return None
        for entry in reuse_cache.get(key, []):
            if reuse_equal is None or reuse_equal(entry["hit"], hit):
                return entry
        return None

    def save_reuse_entry(hit, cluster: Cluster, hit_reports: list[SimilarityReport], gain_db: float) -> None:
        if reuse_key is None:
            return
        key = reuse_key(hit)
        if key is None:
            return
        entry = {
            "hit": hit,
            "cluster_id": cluster.id,
            "gain_db": gain_db,
            "reports": [replace(report) for report in hit_reports],
            # A newly-created representative was not compared with itself.
            # The first duplicate performs that one required comparison, then
            # subsequent duplicates can reuse the complete sequence.
            "needs_self": cluster.representative_hit == hit.id,
        }
        reuse_cache.setdefault(key, []).append(entry)
    # ponytail: sequential representative scan; add indexed clustering if
    # large stems make O(hits × clusters) measurable.
    if callable(compare):
        total = len(hits)
        for index, hit in enumerate(hits):
            if is_cancelled and is_cancelled():
                from ..application import AnalysisCancelled

                raise AnalysisCancelled()

            reuse_entry = find_reuse_entry(hit)
            if reuse_entry is not None:
                cluster = clusters[reuse_entry["cluster_id"] - 1]
                reused_reports = [replace(report, candidate_id=hit.id) for report in reuse_entry["reports"]]
                compared = len(reused_reports)
                if reuse_entry["needs_self"]:
                    if is_cancelled and is_cancelled():
                        from ..application import AnalysisCancelled

                        raise AnalysisCancelled()
                    self_report = classify_report(
                        compare(by_id[cluster.representative_hit], hit),
                        threshold=threshold,
                        spectral_threshold=spectral_threshold,
                    )
                    if self_report.classification not in {"SAME", "GAIN_VARIANT"}:
                        reuse_entry = None
                    else:
                        reused_reports.append(self_report)
                        compared += 1
                        reuse_entry["reports"] = [replace(report) for report in reused_reports]
                        reuse_entry["needs_self"] = False
                if reuse_entry is not None:
                    reports.extend(reused_reports)
                    cluster.hit_ids.append(hit.id)
                    assignments[hit.id] = (cluster.id, reuse_entry["gain_db"])
                    if cache_hit:
                        cache_hit()
                    if progress_detail:
                        for compared_index in range(1, compared + 1):
                            progress_detail(index + 1, total, compared_index)
                    if progress:
                        progress(index + 1, total)
                    continue

            assigned = False
            compared = 0
            hit_reports: list[SimilarityReport] = []
            candidate_clusters = clusters
            if fast_compare:
                candidate_clusters = sorted(
                    clusters,
                    key=lambda cluster: (_feature_distance(by_id[cluster.representative_hit], hit), cluster.id),
                )
            for cluster in candidate_clusters:
                if is_cancelled and is_cancelled():
                    from ..application import AnalysisCancelled

                    raise AnalysisCancelled()
                key = (cluster.representative_hit, hit.id)
                report = comparison_cache.get(key)
                if report is None:
                    report = compare(by_id[cluster.representative_hit], hit)
                    comparison_cache[key] = report
                report = classify_report(report, threshold=threshold, spectral_threshold=spectral_threshold)
                reports.append(report)
                hit_reports.append(report)
                compared += 1
                if progress_detail:
                    progress_detail(index + 1, total, compared)
                if report.classification in {"SAME", "GAIN_VARIANT"}:
                    cluster.hit_ids.append(hit.id)
                    assignments[hit.id] = (cluster.id, report.gain_db if report.classification == "GAIN_VARIANT" else 0.0)
                    assigned = True
                    break
            if not assigned:
                cluster = Cluster(len(clusters) + 1, hit.id, [hit.id])
                clusters.append(cluster)
                assignments[hit.id] = (cluster.id, 0.0)
            save_reuse_entry(hit, cluster, hit_reports, assignments[hit.id][1])
            if progress:
                progress(index + 1, total)
    else:
        reports = list(compare)
        indexed = {(report.reference_id, report.candidate_id): report for report in reports}
        for hit in hits:
            assigned = False
            for cluster in clusters:
                report = indexed.get((cluster.representative_hit, hit.id))
                if report and report.classification in {"SAME", "GAIN_VARIANT"}:
                    cluster.hit_ids.append(hit.id)
                    assignments[hit.id] = (cluster.id, report.gain_db if report.classification == "GAIN_VARIANT" else 0.0)
                    assigned = True
                    break
            if not assigned:
                cluster = Cluster(len(clusters) + 1, hit.id, [hit.id])
                clusters.append(cluster)
                assignments[hit.id] = (cluster.id, 0.0)
    return _make_plan(hits, assignments, clusters), reports
