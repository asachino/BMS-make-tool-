"""Heuristics for flagging changing gain, timbre, pan, or chopped tails."""

from __future__ import annotations

import math

from .._numeric import rms
from .spectral import spectral_features


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-6))


def _chunks(signal, count: int = 4):
    n = len(signal)
    if not n:
        return []
    count = max(2, min(int(count), n))
    return [signal[(index * n) // count : ((index + 1) * n) // count] for index in range(count)]


def _rms_value(signal) -> float:
    return rms(signal) if len(signal) else 0.0


def _pan_db(frame) -> float | None:
    if frame is None:
        return None
    shape = getattr(frame, "shape", None)
    if shape is not None:
        if len(shape) < 2 or shape[1] < 2:
            return None
        left = frame[:, 0]
        right = frame[:, 1]
        return _db(_rms_value(left)) - _db(_rms_value(right))
    if not frame or not isinstance(frame[0], (list, tuple)) or len(frame[0]) < 2:
        return None
    left = [row[0] for row in frame]
    right = [row[1] for row in frame]
    return _db(_rms_value(left)) - _db(_rms_value(right))


def _slice_channels(channels, start: int, end: int):
    if channels is None:
        return None
    return channels[start:end]


def detect_automation(
    signal,
    sample_rate: int,
    *,
    channels=None,
    volume_threshold_db: float = 3.0,
    timbre_threshold: float = 0.18,
    pan_threshold_db: float = 3.0,
    chop_floor: float = 0.08,
    segments: int = 4,
) -> dict:
    """Return JSON-safe automation diagnostics for one extracted hit.

    This is a review hint, not a hard clustering veto.  It deliberately uses
    a few coarse windows so normal decay is not mistaken for a DAW envelope.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    shape = getattr(signal, "shape", None)
    if shape is not None and len(shape) > 1:
        if channels is None:
            channels = signal
        signal = signal.mean(axis=1)
    elif shape is None and len(signal) and isinstance(signal[0], (list, tuple)):
        if channels is None:
            channels = signal
        signal = [sum(float(value) for value in row) / max(1, len(row)) for row in signal]
    chunks = _chunks(signal, segments)
    if not chunks:
        return {"flags": [], "variations": [], "volume_range_db": 0.0, "timbre_range": 0.0, "pan_range_db": 0.0, "chop_count": 0, "segments": []}
    n = len(signal)
    levels = [_rms_value(chunk) for chunk in chunks]
    level_db = [_db(value) for value in levels]
    spectral = [spectral_features(chunk, sample_rate) for chunk in chunks if len(chunk)]
    centroids = [float(item.get("centroid_hz", 0.0)) for item in spectral]
    audible_floor = max(levels, default=0.0) * 0.05
    audible_centroids = [centroid for centroid, level in zip(centroids, levels) if level > audible_floor]
    mean_centroid = sum(audible_centroids) / max(1, len(audible_centroids))
    timbre_range = (max(audible_centroids) - min(audible_centroids)) / max(100.0, mean_centroid) if audible_centroids else 0.0

    pan_values = []
    if channels is not None:
        channel_count = len(channels)
        for index in range(len(chunks)):
            start = (index * channel_count) // len(chunks)
            end = ((index + 1) * channel_count) // len(chunks)
            value = _pan_db(_slice_channels(channels, start, end))
            if value is not None:
                pan_values.append(value)
    pan_range = max(pan_values) - min(pan_values) if pan_values else 0.0

    max_level = max(levels, default=0.0)
    chop_count = 0
    if max_level > 1e-5 and len(levels) > 2:
        # Count low-energy gaps between two audible chunks.  A falling tail
        # alone has no rise after the gap, so it does not trigger this flag.
        for index in range(1, len(levels) - 1):
            if levels[index] <= max_level * float(chop_floor) and levels[index - 1] > max_level * 0.2 and levels[index + 1] > max_level * 0.2:
                chop_count += 1

    volume_range = max(level_db) - min(level_db) if level_db else 0.0
    level_deltas = [right - left for left, right in zip(level_db, level_db[1:])]
    monotonic_decay = bool(level_deltas) and all(delta <= 1.0 for delta in level_deltas)
    volume_changed = volume_range >= float(volume_threshold_db) and not monotonic_decay
    flags = []
    if volume_changed:
        flags.append("volume_change")
    if timbre_range >= float(timbre_threshold):
        flags.append("timbre_change")
    if pan_range >= float(pan_threshold_db):
        flags.append("pan_change")
    if chop_count:
        flags.append("chopped")
    variations = []
    if "volume_change" in flags:
        variations.append("GAIN")
    if "timbre_change" in flags:
        variations.append("TIMBRE")
    if chop_count:
        variations.append("DENSITY")
    if "pan_change" in flags:
        variations.append("STEREO")
    if monotonic_decay and volume_range >= float(volume_threshold_db) and not chop_count:
        variations.append("TAIL")

    segment_rows = []
    for index, level in enumerate(level_db):
        row = {
            "index": index,
            "rms_db": round(level, 4),
            "centroid_hz": round(float(centroids[index]), 4) if index < len(centroids) else 0.0,
        }
        if index < len(pan_values):
            row["pan_db"] = round(float(pan_values[index]), 4)
        segment_rows.append(row)
    return {
        "flags": flags,
        "variations": variations,
        "volume_range_db": round(volume_range, 4),
        "volume_slope_db": round((level_db[-1] - level_db[0]) if len(level_db) > 1 else 0.0, 4),
        "timbre_range": round(timbre_range, 6),
        "centroid_range_hz": round(max(audible_centroids) - min(audible_centroids), 4) if audible_centroids else 0.0,
        "pan_range_db": round(pan_range, 4),
        "chop_count": chop_count,
        "segments": segment_rows,
    }


detect_automation_changes = detect_automation

__all__ = ["detect_automation", "detect_automation_changes"]
