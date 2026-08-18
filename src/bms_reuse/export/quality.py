"""Cheap export sanity checks used by CLI and GUI."""

from __future__ import annotations

from pathlib import Path
import json
import re


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


def check_export_quality(result, exported: dict | None = None) -> dict:
    exported = exported or {}
    sample_paths = exported.get("samples", []) if isinstance(exported, dict) else []
    sample_paths = list(sample_paths) if isinstance(sample_paths, (list, tuple)) else []
    expected = int(result.plan.required_samples)
    sample_dir = Path(exported["samples_dir"]) if exported.get("samples_dir") else None
    sample_requested = "samples" in exported or sample_dir is not None
    sample_files = {Path(path).resolve() for path in sample_paths}
    actual_files = {path.resolve() for path in sample_dir.glob("*.wav")} if sample_dir and sample_dir.is_dir() else set()
    checks = {
        "sample_count_matches_clusters": len(sample_paths) == expected if sample_requested else True,
        "sample_files_exist": all(path.is_file() for path in sample_files) if sample_requested else True,
        "sample_folder_has_no_extra_wav": actual_files == sample_files if sample_dir else True,
        "event_count_matches_hits": len(result.plan.events) == len(result.hits),
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
                except (OSError, ValueError, json.JSONDecodeError):
                    checks[f"{key}_references_exist"] = False
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "expected_samples": expected,
        "actual_samples": len(sample_paths),
        "missing_samples": sorted(str(path) for path in sample_files if not path.is_file()),
        "extra_samples": sorted(str(path) for path in actual_files - sample_files),
    }


def validate_exports(result, exported: dict | None = None) -> dict:
    """Compatibility name used by GUI/CLI integrations."""
    return check_export_quality(result, exported)
