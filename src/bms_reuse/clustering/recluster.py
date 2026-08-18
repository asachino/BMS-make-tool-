"""Threshold-only reclustering for an already analysed stem.

The API deliberately consumes only :class:`SimilarityReport` values and hit
features.  It never calls the audio loader, FFT, or waveform comparator.  A
GUI can therefore offer a reuse-level slider without re-running analysis.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ..classification.classifier import (
    DEFAULT_GAIN_TOLERANCE_DB,
    DEFAULT_SPECTRAL_THRESHOLD,
    DEFAULT_WAVEFORM_THRESHOLD,
    classify_report,
)
from ..similarity.score import SimilarityReport
from .reuse_plan import Cluster, ReusePlan


RECLUSTER_PROFILES = {"strict", "balanced", "aggressive"}


def _number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def resolve_recluster_profile(
    settings: dict | None = None,
    *,
    profile: str | float | int | None = None,
    reuse_level: str | float | int | None = None,
    threshold: float | None = None,
    spectral_threshold: float | None = None,
    gain_tolerance_db: float | None = None,
) -> tuple[str, dict[str, float]]:
    """Resolve a named or continuous reuse level into explicit thresholds.

    ``threshold`` is the continuous waveform threshold.  When no separate
    spectral threshold is supplied it is used for both signals, making a GUI
    slider predictable.  Named levels are relative to the thresholds used for
    the original analysis: strict adds ``0.03`` and aggressive subtracts
    ``0.05`` (both clamped to ``[0, 1]``).
    """
    base = settings or {}
    original = base.get("recluster_base_thresholds") or {}
    base_waveform = _number(original.get("waveform", base.get("threshold", DEFAULT_WAVEFORM_THRESHOLD)), "threshold")
    base_spectral = _number(original.get("spectral", base.get("spectral_threshold", DEFAULT_SPECTRAL_THRESHOLD)), "spectral_threshold")
    base_gain = float(
        base.get(
            "gain_tolerance_db",
            (base.get("recluster_thresholds") or {}).get("gain_tolerance_db", DEFAULT_GAIN_TOLERANCE_DB),
        )
    )
    selected_argument = reuse_level if reuse_level is not None else profile
    selected = selected_argument
    if selected is None:
        selected = base.get("recluster_profile", "balanced")
    if threshold is not None and selected_argument is None:
        selected = "custom"
    if isinstance(selected, (int, float)) and not isinstance(selected, bool):
        threshold = float(selected) if threshold is None else threshold
        selected = "custom"
    elif isinstance(selected, str):
        text = selected.strip().casefold()
        try:
            numeric = float(text)
        except ValueError:
            selected = text or "balanced"
        else:
            threshold = numeric if threshold is None else threshold
            selected = "custom"
    else:
        selected = "balanced"
    if selected not in RECLUSTER_PROFILES and selected != "custom":
        raise ValueError("profile must be strict, balanced, aggressive, or a numeric threshold")
    if threshold is None:
        if selected == "strict":
            waveform = min(1.0, base_waveform + 0.03)
            spectral = min(1.0, base_spectral + 0.03)
        elif selected == "aggressive":
            waveform = max(0.0, base_waveform - 0.05)
            spectral = max(0.0, base_spectral - 0.05)
        else:
            waveform, spectral = base_waveform, base_spectral
    else:
        waveform = _number(threshold, "threshold")
        spectral = waveform if spectral_threshold is None else _number(spectral_threshold, "spectral_threshold")
    if threshold is None and spectral_threshold is not None:
        spectral = _number(spectral_threshold, "spectral_threshold")
    gain = base_gain if gain_tolerance_db is None else float(gain_tolerance_db)
    if gain < 0:
        raise ValueError("gain_tolerance_db must be non-negative")
    return str(selected), {
        "waveform": round(waveform, 6),
        "spectral": round(spectral, 6),
        "waveform_threshold": round(waveform, 6),
        "spectral_threshold": round(spectral, 6),
        "gain_tolerance_db": round(gain, 6),
    }


def _report_key(report: SimilarityReport) -> tuple[int, int]:
    return int(report.reference_id), int(report.candidate_id)


def _reverse_report(report: SimilarityReport) -> SimilarityReport:
    """Return a report with IDs/gain oriented in the opposite direction."""
    return replace(
        report,
        reference_id=report.candidate_id,
        candidate_id=report.reference_id,
        gain_db=-float(report.gain_db),
    )


def _accepted(report: SimilarityReport, thresholds: dict[str, float]) -> bool:
    return (
        float(report.gain_normalized_similarity) >= thresholds["waveform"]
        and float(report.spectral_similarity) >= thresholds["spectral"]
    )


def _feature_report(reference, candidate) -> SimilarityReport | None:
    """Make a conservative report from serialized shape features only."""
    left = getattr(reference, "features", {}) or {}
    right = getattr(candidate, "features", {}) or {}
    keys = ("centroid_hz", "rolloff_hz", "zcr", "attack_ms", "tail_energy")
    if any(key not in left or key not in right for key in keys):
        return None
    try:
        differences = [
            abs(float(left[key]) - float(right[key])) / max(1e-6, abs(float(left[key])), abs(float(right[key])))
            for key in keys
        ]
    except (TypeError, ValueError, OverflowError):
        return None
    similarity = max(0.0, min(1.0, 1.0 - sum(differences) / len(differences)))
    return SimilarityReport(
        int(reference.id), int(candidate.id), similarity, similarity, 0.0,
        similarity, similarity, similarity, similarity, 0,
    )


def _copy_clusters(plan: ReusePlan | None) -> list[Cluster]:
    if not plan:
        return []
    return [Cluster(int(cluster.id), int(cluster.representative_hit), list(cluster.hit_ids)) for cluster in plan.clusters]


def recluster_plan(
    hits: Iterable,
    comparisons: Iterable[SimilarityReport],
    initial_plan: ReusePlan | None,
    *,
    settings: dict | None = None,
    profile: str | float | int | None = None,
    reuse_level: str | float | int | None = None,
    threshold: float | None = None,
    spectral_threshold: float | None = None,
    gain_tolerance_db: float | None = None,
) -> tuple[ReusePlan, list[SimilarityReport], str, dict[str, float]]:
    """Build a new plan using saved comparisons and preserve review intent.

    Data contract for GUI callers:

    * ``hits`` needs only ``id``, ``time`` and optional ``features``;
    * ``comparisons`` needs the serialized ``SimilarityReport`` fields;
    * ``initial_plan`` is used to resolve stored ``review_targets`` cluster
      IDs after the new cluster IDs are assigned.

    The returned plan contains one cluster per representative and one event
    per non-excluded hit.  No input object is mutated.
    """
    settings = settings or {}
    hits = sorted(list(hits), key=lambda hit: int(hit.id))
    reports = [replace(report) for report in comparisons]
    profile_name, thresholds = resolve_recluster_profile(
        settings,
        profile=profile,
        reuse_level=reuse_level,
        threshold=threshold,
        spectral_threshold=spectral_threshold,
        gain_tolerance_db=gain_tolerance_db,
    )
    report_map = {_report_key(report): report for report in reports}
    hit_by_id = {int(hit.id): hit for hit in hits}
    overrides = {
        int(hit_id): str(value).upper()
        for hit_id, value in (settings.get("review_overrides", {}) or {}).items()
        if str(value).upper() in {"S", "G", "D", "I"}
    }
    excluded = {
        int(hit_id)
        for hit_id in (settings.get("excluded_hits", []) or [])
    }
    excluded.update(hit_id for hit_id, value in overrides.items() if value == "I")
    forced_different = {hit_id for hit_id, value in overrides.items() if value == "D" and hit_id not in excluded}
    active_hits = [hit for hit in hits if int(hit.id) not in excluded]
    clusters: list[Cluster] = []
    assignments: dict[int, int] = {}
    gains: dict[int, float] = {}

    def find_report(cluster: Cluster, hit_id: int) -> SimilarityReport | None:
        # A saved analysis normally stores representative -> candidate.  The
        # member fallback also lets aggressive reclustering bridge two old
        # clusters when that pair was measured in the original pass.
        candidates = [cluster.representative_hit] + [value for value in cluster.hit_ids if value != cluster.representative_hit]
        for reference_id in candidates:
            report = report_map.get((reference_id, hit_id))
            if report is not None:
                return report
            reverse = report_map.get((hit_id, reference_id))
            if reverse is not None:
                return _reverse_report(reverse)
            feature_report = _feature_report(hit_by_id.get(reference_id), hit_by_id.get(hit_id))
            if feature_report is not None:
                report_map[(reference_id, hit_id)] = feature_report
                reports.append(feature_report)
                return feature_report
        return None

    for hit in active_hits:
        hit_id = int(hit.id)
        if hit_id in forced_different:
            cluster = Cluster(len(clusters) + 1, hit_id, [hit_id])
            clusters.append(cluster)
            assignments[hit_id] = cluster.id
            gains[hit_id] = 0.0
            continue
        assigned = False
        for cluster in clusters:
            if cluster.representative_hit in forced_different:
                continue
            report = find_report(cluster, hit_id)
            if report is None or not _accepted(report, thresholds):
                continue
            cluster.hit_ids.append(hit_id)
            assignments[hit_id] = cluster.id
            gains[hit_id] = float(report.gain_db) if abs(float(report.gain_db)) >= thresholds["gain_tolerance_db"] else 0.0
            assigned = True
            break
        if not assigned:
            cluster = Cluster(len(clusters) + 1, hit_id, [hit_id])
            clusters.append(cluster)
            assignments[hit_id] = cluster.id
            gains[hit_id] = 0.0

    # Resolve S/G targets against the original cluster IDs before renumbering.
    initial_clusters = _copy_clusters(initial_plan)
    initial_by_id = {cluster.id: cluster for cluster in initial_clusters}
    targets = settings.get("review_targets", {}) or {}

    def move_to_target(hit_id: int, target_cluster: Cluster | None, gain: float) -> None:
        if target_cluster is None:
            return
        current = next((cluster for cluster in clusters if hit_id in cluster.hit_ids), None)
        if current is target_cluster:
            gains[hit_id] = gain
            return
        if current is not None:
            current.hit_ids.remove(hit_id)
            if not current.hit_ids:
                clusters.remove(current)
        target_cluster.hit_ids.append(hit_id)
        assignments[hit_id] = target_cluster.id
        gains[hit_id] = gain

    for hit_id, override in sorted(overrides.items()):
        if override not in {"S", "G"} or hit_id in excluded:
            continue
        initial_target_id = None
        try:
            initial_target_id = int(targets.get(str(hit_id), targets.get(hit_id)))
        except (TypeError, ValueError):
            pass
        target_members = list(initial_by_id.get(initial_target_id, Cluster(0, hit_id, [hit_id])).hit_ids)
        target_members = [value for value in target_members if value not in excluded and value not in forced_different]
        if not target_members:
            target_members = [hit_id]
        target_hit = next((value for value in target_members if value != hit_id), target_members[0])
        target_cluster = next((cluster for cluster in clusters if target_hit in cluster.hit_ids), None)
        if target_cluster is None:
            target_cluster = next((cluster for cluster in clusters if hit_id in cluster.hit_ids), None)
        report = report_map.get((target_hit, hit_id)) or report_map.get((hit_id, target_hit))
        gain = float(report.gain_db) if override == "G" and report is not None else 0.0
        move_to_target(hit_id, target_cluster, gain)

    # Empty clusters can result from moving a forced review hit.  IDs are
    # intentionally compact so representative WAV filenames stay aligned.
    clusters = [cluster for cluster in clusters if cluster.hit_ids]
    cluster_by_hit: dict[int, int] = {}
    for cluster_id, cluster in enumerate(clusters, 1):
        cluster.id = cluster_id
        for hit_id in cluster.hit_ids:
            cluster_by_hit[int(hit_id)] = cluster_id
    events = [
        {
            "hit": int(hit.id),
            "time": float(hit.time),
            "sample_id": f"sample_{cluster_by_hit[int(hit.id)]:03d}",
            "gain_db": round(float(gains.get(int(hit.id), 0.0)), 4),
        }
        for hit in active_hits
        if int(hit.id) in cluster_by_hit
    ]
    plan = ReusePlan(clusters, events)

    # Reports remain useful diagnostics after a level change, but classifications
    # must describe the selected thresholds.  I removes the report entirely.
    updated_reports: list[SimilarityReport] = []
    for report in reports:
        if report.reference_id in excluded or report.candidate_id in excluded:
            continue
        updated = classify_report(
            report,
            threshold=thresholds["waveform"],
            spectral_threshold=thresholds["spectral"],
            gain_tolerance_db=thresholds["gain_tolerance_db"],
        )
        override = overrides.get(int(updated.candidate_id)) or overrides.get(int(updated.reference_id))
        if override == "S":
            updated.classification = "SAME"
        elif override == "G":
            updated.classification = "GAIN_VARIANT"
        elif override == "D":
            updated.classification = "DIFFERENT"
        updated_reports.append(updated)
    return plan, updated_reports, profile_name, thresholds
