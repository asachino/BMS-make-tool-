"""Lightweight spectral-flux onset detection."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .._numeric import np


@dataclass(frozen=True)
class Onset:
    id: int
    sample: int
    time: float

    def to_dict(self) -> dict:
        return asdict(self)


def _energy_novelty(signal: list[float], frame: int, hop: int) -> list[float]:
    values: list[float] = []
    previous = 0.0
    for start in range(0, max(1, len(signal)), hop):
        block = signal[start : start + frame]
        energy = math.sqrt(sum(x * x for x in block) / len(block)) if block else 0.0
        values.append(max(0.0, energy - previous))
        previous = energy
    return values


def _spectral_flux(signal, frame: int, hop: int) -> list[float]:
    if np is None:  # pragma: no cover - exercised only without NumPy
        return _energy_novelty([float(x) for x in signal], frame, hop)
    if not len(signal):
        return []
    window = np.hanning(frame)
    previous = np.zeros(frame // 2 + 1)
    result: list[float] = []
    for start in range(0, len(signal), hop):
        block = np.zeros(frame)
        part = signal[start : start + frame]
        block[: len(part)] = part
        magnitude = np.abs(np.fft.rfft(block * window))
        result.append(float(np.sum(np.maximum(magnitude - previous, 0.0))))
        previous = magnitude
    return result


def _pick_peaks(novelty: list[float], threshold: float, min_frames: int) -> list[int]:
    if len(novelty) < 1:
        return []
    ordered = sorted(novelty)
    median = ordered[len(ordered) // 2]
    mean = sum(novelty) / len(novelty)
    variance = sum((x - mean) ** 2 for x in novelty) / len(novelty)
    spread = math.sqrt(variance)
    maximum = max(novelty)
    # A robust dynamic threshold handles both sparse kicks and dense stems.
    cutoff = median + max(spread * threshold, maximum * 0.015)
    candidates = [
        i
        for i in range(1, len(novelty) - 1)
        if novelty[i] >= cutoff and novelty[i] >= novelty[i - 1] and novelty[i] > novelty[i + 1]
    ]
    if len(novelty) > 1 and novelty[0] >= cutoff and novelty[0] > novelty[1]:
        candidates.insert(0, 0)
    selected: list[int] = []
    for index in candidates:
        if not selected or index - selected[-1] >= min_frames:
            selected.append(index)
        elif novelty[index] > novelty[selected[-1]]:
            selected[-1] = index
    return selected


def detect_onsets(
    signal,
    sample_rate: int,
    *,
    threshold: float = 0.35,
    min_separation_ms: float = 50.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    bpm: float | None = None,
    offset: float = 0.0,
    subdivision: int = 16,
) -> list[Onset]:
    """Detect transient positions, optionally nudging nearby hits to a BPM grid."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if frame_ms <= 0 or hop_ms <= 0 or min_separation_ms <= 0:
        raise ValueError("frame_ms, hop_ms and min_separation_ms must be positive")
    frame = max(8, round(sample_rate * frame_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    min_frames = max(1, round(min_separation_ms / hop_ms))
    novelty = _spectral_flux(signal, frame, hop)
    peaks = _pick_peaks(novelty, threshold, min_frames)
    positions = [min(len(signal) - 1, max(0, index * hop)) for index in peaks] if len(signal) else []
    # Flux is frame-based, so refine each candidate to the local attack start
    # before the small alignment search in the comparator.
    if positions:
        radius = max(1, frame)
        refined = []
        for position in positions:
            start = max(0, position - radius)
            end = min(len(signal), position + radius + 1)
            if np is not None:
                local = np.abs(signal[start:end])
                if len(local):
                    peak_index = int(np.argmax(local))
                    peak = float(local[peak_index])
                    onset_index = peak_index
                    trigger = peak * 0.1
                    while onset_index > 0 and local[onset_index - 1] >= trigger:
                        onset_index -= 1
                    refined.append(start + onset_index)
                else:
                    refined.append(position)
            else:  # pragma: no cover
                if end > start:
                    local = [abs(signal[start + i]) for i in range(end - start)]
                    peak_index = max(range(len(local)), key=local.__getitem__)
                    trigger = local[peak_index] * 0.1
                    onset_index = peak_index
                    while onset_index > 0 and local[onset_index - 1] >= trigger:
                        onset_index -= 1
                    refined.append(start + onset_index)
                else:
                    refined.append(position)
        positions = refined
        minimum_samples = max(1, round(sample_rate * min_separation_ms / 1000.0))
        deduplicated: list[int] = []
        for position in positions:
            if not deduplicated or position - deduplicated[-1] >= minimum_samples:
                deduplicated.append(position)
        positions = deduplicated
    times = [sample / sample_rate for sample in positions]
    if bpm and bpm > 0 and subdivision > 0:
        grid_step = 60.0 / bpm / subdivision
        snapped: list[float] = []
        for time in times:
            grid_time = offset + round((time - offset) / grid_step) * grid_step
            if abs(grid_time - time) <= max(0.025, grid_step * 0.25):
                snapped.append(max(0.0, grid_time))
            else:
                snapped.append(time)
        times = snapped
        positions = [min(len(signal) - 1, max(0, round(time * sample_rate))) for time in times]
    return [Onset(index, sample, sample / sample_rate) for index, sample in enumerate(positions)]
