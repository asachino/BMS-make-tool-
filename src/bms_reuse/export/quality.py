"""Cheap export sanity checks used by CLI and GUI."""

from __future__ import annotations

from pathlib import Path
import json
import re
import wave


def _referenced_files(path: Path) -> list[Path]:
    if path.suffix.casefold() == ".bms":
        references = []
        definitions: dict[str, Path] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^#WAV[0-9A-Z]{2}\s+(.+)$", line.strip(), re.IGNORECASE)
            if match:
                code = line.strip()[4:6].upper()
                definitions[code] = path.parent / match.group(1).strip()
                references.append(definitions[code])
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^#\d{3}\d{2}:([0-9A-Z]+)$", line.strip(), re.IGNORECASE)
            if not match or len(match.group(1)) % 2:
                continue
            for offset in range(0, len(match.group(1)), 2):
                code = match.group(1)[offset:offset + 2].upper()
                if code != "00" and code not in definitions:
                    references.append(path.parent / f"__missing_bms_wav_{code}.wav")
        return references
    if path.suffix.casefold() == ".bmson":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [path.parent / str(channel.get("name")) for channel in data.get("sound_channels", []) if channel.get("name")]
    return []


def _bms_event_count(path: Path) -> int:
    """Count non-empty event cells in a generated BMS channel map."""
    count = 0
    pattern = re.compile(r"^#\d{3}\d{2}:([0-9A-Z]+)$", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match or len(match.group(1)) % 2:
            continue
        count += sum(
            match.group(1)[offset:offset + 2].upper() != "00"
            for offset in range(0, len(match.group(1)), 2)
        )
    return count


def _wav_frame_count(path: Path) -> int:
    with wave.open(str(path), "rb") as stream:
        return int(stream.getnframes())


def check_export_quality(result, exported: dict | None = None) -> dict:
    exported = exported or {}
    sample_paths = exported.get("samples", []) if isinstance(exported, dict) else []
    sample_paths = list(sample_paths) if isinstance(sample_paths, (list, tuple)) else []
    expected = int(result.plan.required_samples)
    active_ids = result.settings.get("active_hit_ids") if isinstance(getattr(result, "settings", None), dict) else None
    active_hit_count = len(active_ids) if isinstance(active_ids, list) else len(result.hits)
    sample_dir = Path(exported["samples_dir"]) if exported.get("samples_dir") else None
    # An empty optional ``samples`` list is how CLI/GUI represent "not
    # requested".  A directory or at least one referenced file means the
    # caller requested representative WAV validation.
    sample_requested = sample_dir is not None or bool(sample_paths)
    sample_files = {Path(path).resolve() for path in sample_paths}
    actual_files = {path.resolve() for path in sample_dir.glob("*.wav")} if sample_dir and sample_dir.is_dir() else set()
    endpoint_mismatches: list[dict] = []
    if sample_requested:
        paths_by_name = {path.name.casefold(): path for path in sample_files}
        for cluster in result.plan.clusters:
            sample_path = paths_by_name.get(f"sample_{int(cluster.id):03d}.wav".casefold())
            if sample_path is None or not sample_path.is_file():
                continue
            representative = next(
                (hit for hit in result.hits if int(hit.id) == int(cluster.representative_hit)),
                None,
            )
            if representative is None:
                continue
            effective = getattr(representative, "effective_settings", {}) or {}
            smart_requested = bool(effective.get(
                "smart_end_requested",
                effective.get(
                    "smart_end_applied",
                    effective.get("enabled", False),
                ),
            ))
            expected_frames = (
                max(0, int(representative.source_end) - int(representative.source_start))
                if smart_requested
                else int(representative.sample_count)
            )
            try:
                actual_frames = _wav_frame_count(sample_path)
            except (OSError, wave.Error):
                endpoint_mismatches.append({
                    "sample": str(sample_path),
                    "cluster": int(cluster.id),
                    "expected_frames": expected_frames,
                    "actual_frames": None,
                    "smart_end": smart_requested,
                    "error": "WAVフレーム数を読み取れませんでした",
                })
                continue
            if actual_frames != expected_frames:
                endpoint_mismatches.append({
                    "sample": str(sample_path),
                    "cluster": int(cluster.id),
                    "expected_frames": expected_frames,
                    "actual_frames": actual_frames,
                    "smart_end": smart_requested,
                })
    checks = {
        "sample_folder_exists": sample_dir.is_dir() if sample_dir is not None else True,
        "sample_count_matches_clusters": len(sample_paths) == expected if sample_requested else True,
        "sample_files_exist": all(path.is_file() for path in sample_files) if sample_requested else True,
        "sample_folder_has_no_extra_wav": actual_files == sample_files if sample_dir else True,
        "sample_frames_match_endpoint": not endpoint_mismatches if sample_requested else True,
        "event_count_matches_hits": len(result.plan.events) == active_hit_count,
        "cluster_ids_unique": len({cluster.id for cluster in result.plan.clusters}) == expected,
        "source_hash_present": bool(result.source_hash),
    }
    for key in ("json", "csv", "bms", "bmson"):
        if exported.get(key):
            export_path = Path(exported[key])
            checks[f"{key}_written"] = export_path.is_file() and export_path.stat().st_size > 0
            if checks[f"{key}_written"] and key in {"bms", "bmson"}:
                try:
                    references = _referenced_files(export_path)
                    checks[f"{key}_references_exist"] = all(reference.is_file() for reference in references)
                    if key == "bms":
                        checks["bms_event_count_matches_hits"] = _bms_event_count(export_path) == active_hit_count
                        # Older versions emitted this marker after dropping a
                        # colliding event.  It is intentionally invalid now:
                        # collision handling must remain non-lossy.
                        text = export_path.read_text(encoding="utf-8")
                        checks["bms_grid_collisions_preserved"] = "後続を省略" not in text
                except (OSError, ValueError, json.JSONDecodeError):
                    checks[f"{key}_references_exist"] = False
                    if key == "bms":
                        checks["bms_event_count_matches_hits"] = False
                        checks["bms_grid_collisions_preserved"] = False
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "expected_samples": expected,
        "actual_samples": len(sample_paths),
        "active_hits": active_hit_count,
        "missing_samples": sorted(str(path) for path in sample_files if not path.is_file()),
        "extra_samples": sorted(str(path) for path in actual_files - sample_files),
        "endpoint_mismatches": endpoint_mismatches,
    }


def validate_exports(result, exported: dict | None = None) -> dict:
    """Compatibility name used by GUI/CLI integrations."""
    return check_export_quality(result, exported)
