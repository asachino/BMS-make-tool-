"""Small, standards-shaped BMSON exporter."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _sample_filename(prefix: str, sample_id: str) -> str:
    try:
        number = int(str(sample_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        number = 1
    prefix = str(prefix or "")
    if prefix.endswith("_") and "/" not in prefix and "\\" not in prefix:
        return f"{prefix}{number:03d}.wav"
    directory = Path(prefix.replace("\\", "/")).as_posix().rstrip("/")
    # Preserve a caller-supplied relative directory.  BMSON resolves sound
    # channel names relative to the chart, so dropping this part would make an
    # externally placed chart point at the wrong WAV folder.
    return f"{directory + '/' if directory and directory != '.' else ''}sample_{number:03d}.wav"


def write_bmson(
    path: str | Path,
    plan,
    *,
    bpm: float | None = None,
    offset: float = 0.0,
    resolution: int = 240,
    wav_prefix: str = "sample_",
    excluded_hits: set[int] | None = None,
) -> Path:
    """Write BMSON v1 notes and preserve gain metadata for review tools."""
    if bpm is None or bpm <= 0 or resolution <= 0:
        raise ValueError("BMSON出力にはBPMが必要です")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notes_by_sample: dict[str, list[dict]] = defaultdict(list)
    gain_events: list[dict] = []
    for event in plan.events:
        if excluded_hits and int(event.get("hit", -1)) in excluded_hits:
            continue
        tick = max(0, round((float(event.get("time", 0.0)) - float(offset)) / (60.0 / bpm) * resolution))
        sample_id = str(event.get("sample_id", "sample_001"))
        filename = _sample_filename(wav_prefix, sample_id)
        notes_by_sample[filename].append({
            "x": 0,
            "y": tick,
            "l": 0,
            "c": False,
        })
        gain_events.append({
            "hit": int(event.get("hit", -1)),
            "sample": filename,
            "gain_db": float(event.get("gain_db", 0.0)),
        })
    channels = [
        {"name": filename, "notes": sorted(notes, key=lambda note: (note["y"], note["x"]))}
        for filename, notes in sorted(notes_by_sample.items())
    ]
    sound_samples = [{"name": filename} for filename in sorted(notes_by_sample)]
    data = {
        "version": "1.0.0",
        "info": {
            "title": "BMS Stem Reuse",
            "artist": "",
            "genre": "",
            "mode_hint": "generic-nkeys",
            "chart_name": "",
            "level": 0,
            "init_bpm": float(bpm),
            "resolution": int(resolution),
        },
        "lines": [],
        "bpm_events": [{"y": 0, "bpm": float(bpm)}],
        "stop_events": [],
        "sound_channels": channels,
        # Extension for tools that want an explicit sample inventory; the
        # standard BMSON representation is the sound_channels name field.
        "sound_samples": sound_samples,
        "bga": {"bga_header": [], "bga_events": [], "layer_events": [], "poor_events": []},
        "bms_reuse": {"offset": float(offset), "gain_policy": "metadata", "gain_events": gain_events},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
