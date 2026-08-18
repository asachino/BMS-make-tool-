"""Deterministic user-defined cut rules for loop/chop-oriented stems."""

from __future__ import annotations

import math

from .onset import Onset


LOOP_RULES = ("off", "seconds", "beats", "bars", "points")
_ALIASES = {
    "": "off",
    "none": "off",
    "detected": "off",
    "onset": "off",
    "fixed": "seconds",
    "fixed_seconds": "seconds",
    "second": "seconds",
    "beat": "beats",
    "bar": "bars",
    "custom": "points",
    "manual": "points",
    "grid": "beats",
}


def normalize_loop_rule(value: str | None) -> str:
    key = str(value or "off").strip().casefold().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in LOOP_RULES:
        raise ValueError("loop_rule must be off, seconds, beats, bars, or points")
    return key


def _as_points(points) -> list[float]:
    if points is None:
        return []
    if isinstance(points, str):
        points = [value for value in points.split(",") if value.strip()]
    try:
        values = sorted({round(float(value), 9) for value in points})
    except (TypeError, ValueError) as exc:
        raise ValueError("loop_points must contain seconds") from exc
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("loop_points must be finite and non-negative")
    return values


def _regular_points(start: float, duration: float, step: float) -> list[float]:
    if not math.isfinite(step) or step <= 0:
        raise ValueError("loop interval must be positive")
    points = []
    cursor = max(0.0, float(start))
    while cursor < duration:
        points.append(round(cursor, 9))
        cursor += step
    return points


def pattern_points(pattern, start: float, duration: float) -> list[float]:
    """Expand repeating positive second intervals into absolute cut points."""
    try:
        intervals = [float(value) for value in pattern]
    except (TypeError, ValueError) as exc:
        raise ValueError("loop_pattern must contain positive seconds") from exc
    if not intervals or any(not math.isfinite(value) or value <= 0 for value in intervals):
        raise ValueError("loop_pattern must contain finite positive seconds")
    points = []
    cursor = max(0.0, float(start))
    index = 0
    while cursor < duration:
        points.append(round(cursor, 9))
        cursor += intervals[index % len(intervals)]
        index += 1
    return points


def build_cut_onsets(
    onsets: list[Onset],
    frame_count: int,
    sample_rate: int,
    *,
    rule: str = "off",
    seconds: float | None = None,
    beats: float | None = None,
    bars: float | None = None,
    bpm: float | None = None,
    start_sec: float = 0.0,
    points=None,
) -> tuple[list[Onset], list[float]]:
    """Return extraction boundaries and the exact seconds used in settings.

    `off` preserves spectral onsets byte-for-byte.  Other modes intentionally
    generate regular boundaries, making a loop/chop operation reproducible.
    """
    canonical = normalize_loop_rule(rule)
    if sample_rate <= 0 or frame_count < 0:
        raise ValueError("sample_rate must be positive and frame_count must be non-negative")
    duration = frame_count / sample_rate if sample_rate else 0.0
    if canonical == "off":
        return list(onsets), [round(float(onset.time), 9) for onset in onsets]
    if not math.isfinite(float(start_sec)) or start_sec < 0:
        raise ValueError("loop_start_sec must be finite and non-negative")
    if canonical == "points":
        values = _as_points(points)
    elif canonical == "seconds":
        if seconds is None or not math.isfinite(float(seconds)) or float(seconds) <= 0:
            raise ValueError("loop_seconds must be positive for the seconds rule")
        values = _regular_points(start_sec, duration, float(seconds))
    elif canonical in {"beats", "bars"}:
        if bpm is None or not math.isfinite(float(bpm)) or float(bpm) <= 0:
            raise ValueError("bpm is required for beat/bar loop cuts")
        amount = beats if canonical == "beats" else (float(bars) * 4.0 if bars is not None else None)
        if amount is None or not math.isfinite(float(amount)) or float(amount) <= 0:
            name = "loop_beats" if canonical == "beats" else "loop_bars"
            raise ValueError(f"{name} must be positive for the {canonical} rule")
        values = _regular_points(start_sec, duration, 60.0 / float(bpm) * float(amount))
    else:  # pragma: no cover - normalize_loop_rule keeps this unreachable.
        values = []
    values = [value for value in values if value < duration or (frame_count and value == duration)]
    positions = [min(max(0, round(value * sample_rate)), max(0, frame_count - 1)) for value in values]
    unique_positions = []
    for position in positions:
        if not unique_positions or position > unique_positions[-1]:
            unique_positions.append(position)
    cut_points = [round(position / sample_rate, 9) for position in unique_positions]
    return [Onset(index, position, position / sample_rate) for index, position in enumerate(unique_positions)], cut_points


__all__ = ["LOOP_RULES", "build_cut_onsets", "build_loop_segments", "normalize_loop_rule", "pattern_points"]

build_loop_segments = build_cut_onsets
