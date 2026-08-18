"""JSON presets for analysis settings."""

from __future__ import annotations

import json
from pathlib import Path


PRESET_VERSION = 1
PRESET_KEYS = {
    "instrument", "threshold", "spectral_threshold", "onset_threshold",
    "min_separation_ms", "pre_roll_ms", "window_ms", "max_alignment_ms",
    "bpm", "offset", "subdivision", "beat_division", "margin_percent",
    "margin", "fade_in_ms", "fade_out_ms", "fast_compare", "bms_channel",
}


def save_preset(path: str | Path, settings: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": PRESET_VERSION, "settings": {key: value for key, value in settings.items() if key in PRESET_KEYS}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_preset(path: str | Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("プリセットを読み込めませんでした") from exc
    if not isinstance(data, dict):
        raise ValueError("プリセット形式が不正です")
    values = data.get("settings", data)
    if not isinstance(values, dict):
        raise ValueError("プリセット形式が不正です")
    return {key: value for key, value in values.items() if key in PRESET_KEYS}
