"""Fixed comparison windows around detected onsets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .._numeric import as_float_list, max_abs, np, pad_or_trim, rms
from ..audio.loader import AudioData, mono_signal
from ..detection.onset import Onset


@dataclass
class Hit:
    id: int
    onset_sample: int
    time: float
    samples: object
    source_start: int
    source_end: int
    overlap_warning: bool = False
    features: dict = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        shape = getattr(self.samples, "shape", None)
        return int(shape[0]) if shape is not None else len(self.samples)

    def to_dict(self, include_samples: bool = False) -> dict:
        data = {
            "id": self.id,
            "time": self.time,
            "sample": self.onset_sample,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "overlap_warning": self.overlap_warning,
            "features": self.features,
        }
        if include_samples:
            data["samples"] = as_float_list(self.samples)
        return data


def extract_hits(
    audio: AudioData,
    onsets: list[Onset],
    *,
    pre_roll_ms: float = 5.0,
    window_ms: float = 800.0,
    overlap_threshold: float = 0.01,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Hit]:
    """Extract equal-length mono windows while retaining source coordinates."""
    if pre_roll_ms < 0 or window_ms <= 0:
        raise ValueError("pre_roll_ms must be non-negative and window_ms must be positive")
    signal = mono_signal(audio)
    total = audio.frame_count
    pre = round(audio.sample_rate * pre_roll_ms / 1000.0)
    window = max(1, round(audio.sample_rate * window_ms / 1000.0))
    hits: list[Hit] = []
    peak = max_abs(signal)
    for position, onset in enumerate(onsets):
        if is_cancelled and is_cancelled():
            from ..application import AnalysisCancelled

            raise AnalysisCancelled()
        start = max(0, onset.sample - pre)
        requested_end = min(total, start + window)
        # Stop before the next detected attack; otherwise a fast kick pattern
        # would compare each hit together with its successor.
        next_boundary = total
        if position + 1 < len(onsets):
            next_boundary = max(start, onsets[position + 1].sample - pre)
        end = min(requested_end, next_boundary)
        data = signal[start:end]
        samples = pad_or_trim(data, window)
        # The pre-roll itself contains the current attack.  Inspect a context
        # immediately before pre-roll so a normal ramp is not mistaken for a
        # preceding hit's tail.
        context = max(pre, round(audio.sample_rate * 20.0 / 1000.0))
        before = signal[max(0, start - context) : start]
        overlap = len(before) > 0 and rms(before) > max(1e-5, peak * overlap_threshold)
        hits.append(Hit(onset.id, onset.sample, onset.time, samples, start, end, overlap))
        if progress:
            progress(position + 1, len(onsets))
    return hits
