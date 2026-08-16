"""Conservative sequential clustering and the resulting reuse mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..classification.classifier import classify_report
from ..similarity.score import SimilarityReport


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
    threshold: float = 0.995,
    spectral_threshold: float = 0.92,
    progress: Callable[[int, int], None] | None = None,
    progress_detail: Callable[[int, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
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
    # ponytail: sequential representative scan; add indexed clustering if
    # large stems make O(hits × clusters) measurable.
    if callable(compare):
        total = len(hits)
        for index, hit in enumerate(hits):
            if is_cancelled and is_cancelled():
                from ..application import AnalysisCancelled

                raise AnalysisCancelled()
            assigned = False
            compared = 0
            for cluster in clusters:
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
