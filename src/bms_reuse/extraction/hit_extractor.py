"""Fixed comparison windows around detected onsets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .._numeric import as_float_list, max_abs, np, pad_or_trim, rms
from ..audio.loader import AudioData, mono_signal
from ..detection.onset import Onset
from ..features.percussion import normalize_instrument


# These profiles deliberately describe the *tail search* rather than the
# instrument itself.  A closed hi-hat and an open hi-hat use the same profile;
# the former reaches the silence gate early while the latter naturally keeps
# the detected end later.  A caller can override any value through
# ``smart_end_settings`` and the effective values are saved on every hit.
SMART_END_PROFILES = {
    "kick": {"min_tail_ms": 18.0, "max_tail_ms": 620.0, "silence_ms": 28.0},
    "snare": {"min_tail_ms": 35.0, "max_tail_ms": 900.0, "silence_ms": 48.0},
    "hihat": {"min_tail_ms": 8.0, "max_tail_ms": 720.0, "silence_ms": 22.0},
    "other": {"min_tail_ms": 20.0, "max_tail_ms": 760.0, "silence_ms": 36.0},
}

SMART_END_DEFAULTS = {
    "enabled": True,
    "apply_to_explicit": False,
    "silence_rms_db": -42.0,
    "silence_peak_db": -34.0,
    "frame_ms": 2.0,
    "zero_crossing_ms": 2.0,
    "safety_margin_ms": 1.0,
    "next_attack_margin_ms": 1.0,
    "attack_window_ms": 16.0,
}


@dataclass(frozen=True)
class SmartEndDecision:
    """A deterministic endpoint decision kept small enough for JSON/CSV."""

    source_end: int
    reason: str
    confidence: float
    warnings: tuple[str, ...] = ()


def resolve_smart_end_settings(
    instrument: str = "kick",
    settings: Mapping[str, object] | None = None,
) -> dict:
    """Return JSON-safe, instrument-aware endpoint settings.

    The function is intentionally public so GUI/CLI callers can display the
    exact settings that were used.  Unknown keys are ignored so a preset can
    safely carry settings from a newer build without changing this detector.
    """
    canonical = normalize_instrument(instrument)
    values = dict(SMART_END_DEFAULTS)
    values.update(SMART_END_PROFILES[canonical])
    if isinstance(settings, Mapping):
        aliases = {
            "tail_min_ms": "min_tail_ms",
            "tail_max_ms": "max_tail_ms",
            "max_duration_ms": "max_tail_ms",
        }
        for key, value in settings.items():
            key = aliases.get(key, key)
            if key in {
                "enabled", "apply_to_explicit", "min_tail_ms", "max_tail_ms",
                "silence_ms", "silence_rms_db", "silence_peak_db", "frame_ms",
                "zero_crossing_ms", "safety_margin_ms", "next_attack_margin_ms",
                "attack_window_ms",
            }:
                values[key] = value
    numeric_keys = {
        "min_tail_ms", "max_tail_ms", "silence_ms", "silence_rms_db",
        "silence_peak_db", "frame_ms", "zero_crossing_ms", "safety_margin_ms",
        "next_attack_margin_ms", "attack_window_ms",
    }
    for key in numeric_keys:
        try:
            value = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"smart end setting {key} must be a number") from exc
        if not math.isfinite(value):
            raise ValueError(f"smart end setting {key} must be finite")
        if key not in {"silence_rms_db", "silence_peak_db"} and value < 0:
            raise ValueError(f"smart end setting {key} must be non-negative")
        if key in {"silence_rms_db", "silence_peak_db"} and value > 0:
            raise ValueError(f"smart end setting {key} must be non-positive")
        values[key] = value
    if values["max_tail_ms"] < values["min_tail_ms"]:
        raise ValueError("smart end max_tail_ms must be at least min_tail_ms")
    values["enabled"] = bool(values["enabled"])
    values["apply_to_explicit"] = bool(values["apply_to_explicit"])
    values["instrument"] = canonical
    values["profile"] = canonical
    values["tail_min_ms"] = values["min_tail_ms"]
    values["tail_max_ms"] = values["max_tail_ms"]
    values["max_duration_ms"] = values["max_tail_ms"]
    # Stable insertion and numeric values make settings hashes reproducible.
    return values


def _segment_rms_peak(values) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if np is not None and hasattr(values, "shape"):
        absolute = np.abs(values)
        return float(np.sqrt(np.mean(values * values))), float(np.max(absolute))
    numeric = [float(value) for value in values]
    return rms(numeric), max_abs(numeric)


def _zero_crossing_near(values, center: int, radius: int) -> int | None:
    """Return the closest sign transition, preferring the post-center side."""
    if radius <= 0 or len(values) == 0:
        return None
    low = max(1, center - radius)
    high = min(len(values) - 1, center + radius)
    candidates = []
    for index in range(low, high + 1):
        left = float(values[index - 1])
        right = float(values[index])
        if (left < 0.0 <= right) or (left > 0.0 >= right):
            candidates.append(index)
    if not candidates:
        return None
    return min(candidates, key=lambda index: (abs(index - center), index < center))


def detect_smart_end(
    signal,
    *,
    start: int,
    hard_end: int,
    sample_rate: int,
    instrument: str = "kick",
    onset_sample: int | None = None,
    next_attack_sample: int | None = None,
    settings: Mapping[str, object] | None = None,
) -> SmartEndDecision:
    """Find a safe endpoint inside ``hard_end`` using decay and attack bounds.

    ``hard_end`` is the existing extractor boundary (window or next onset),
    so this function can only shorten a segment.  The detector combines a
    minimum/maximum tail, frame RMS and peak silence gates, a consecutive
    silence duration, a small safety margin, and a nearby zero crossing.  It
    never scans beyond the next attack boundary supplied by the caller.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    effective = resolve_smart_end_settings(instrument, settings)
    low = max(0, int(start))
    upper = max(low, int(hard_end))
    if next_attack_sample is not None:
        upper = min(upper, max(low, int(next_attack_sample)))
        next_margin = round(sample_rate * effective["next_attack_margin_ms"] / 1000.0)
        upper = min(upper, max(low, int(next_attack_sample) - next_margin))
    attack = max(low, min(upper, int(onset_sample if onset_sample is not None else low)))
    warnings: list[str] = []
    if upper <= attack:
        warnings.append("hard_limit_before_attack")
        if next_attack_sample is not None:
            warnings.append("NEXT_ATTACK_LIMIT")
        reason = "next_attack" if next_attack_sample is not None else "window"
        return SmartEndDecision(upper, reason, 0.2, tuple(warnings))

    min_tail = round(sample_rate * effective["min_tail_ms"] / 1000.0)
    max_tail = round(sample_rate * effective["max_tail_ms"] / 1000.0)
    minimum = min(upper, attack + max(0, min_tail))
    maximum = min(upper, attack + max(1, max_tail))
    if maximum < minimum:
        maximum = minimum
    values = signal[attack:maximum]
    if len(values) == 0:
        warnings.append("empty_tail")
        if next_attack_sample is not None:
            warnings.append("NEXT_ATTACK_LIMIT")
        reason = "next_attack" if next_attack_sample is not None else "window"
        return SmartEndDecision(maximum, reason, 0.2, tuple(warnings))

    attack_window = min(len(values), max(1, round(sample_rate * effective["attack_window_ms"] / 1000.0)))
    _, attack_peak = _segment_rms_peak(values[:attack_window])
    if attack_peak <= 1e-8:
        warnings.append("no_attack_energy")
        return SmartEndDecision(minimum, "no_decay", 0.15, tuple(warnings))

    # The absolute floor keeps a very quiet source from treating floating
    # point noise as a meaningful tail while the relative gates follow gain.
    rms_gate = max(1e-6, attack_peak * (10.0 ** (effective["silence_rms_db"] / 20.0)))
    peak_gate = max(1e-5, attack_peak * (10.0 ** (effective["silence_peak_db"] / 20.0)))
    frame = max(1, round(sample_rate * effective["frame_ms"] / 1000.0))
    silence_frames = max(1, math.ceil(sample_rate * effective["silence_ms"] / 1000.0 / frame))
    first_silent: int | None = None
    run = 0
    # Use absolute source coordinates.  Frames that overlap the minimum tail
    # are still measured, but cannot become a cut point before ``minimum``.
    for offset in range(0, len(values), frame):
        frame_values = values[offset : offset + frame]
        frame_rms, frame_peak = _segment_rms_peak(frame_values)
        silent = frame_rms <= rms_gate and frame_peak <= peak_gate
        run = run + 1 if silent else 0
        if run >= silence_frames:
            first_silent = attack + max(minimum - attack, offset - (run - 1) * frame)
            break

    candidate = maximum
    max_duration_end = attack + max(1, max_tail)
    max_duration_reached = max_duration_end <= upper
    next_attack_limited = next_attack_sample is not None and upper < max_duration_end
    reason = "max_duration" if max_duration_reached else "window"
    confidence = 0.62
    if first_silent is not None:
        candidate = min(maximum, max(minimum, first_silent))
        reason = "silence"
        confidence = 0.86
    else:
        warnings.append("no_continuous_silence")
        if next_attack_limited:
            reason = "next_attack"
            warnings.extend(("NEXT_ATTACK_LIMIT", "TAIL_CUT"))
            confidence = 0.78
        elif max_duration_reached:
            warnings.append("TOO_LONG")

    if upper <= attack + max(0, min_tail):
        warnings.append("minimum_tail_clipped")
        confidence = min(confidence, 0.55)

    margin = round(sample_rate * effective["safety_margin_ms"] / 1000.0)
    crossing = _zero_crossing_near(signal, min(upper, candidate + margin), round(sample_rate * effective["zero_crossing_ms"] / 1000.0))
    if crossing is not None and minimum <= crossing <= upper:
        if crossing != candidate:
            candidate = crossing
            warnings.append("zero_crossing_adjusted")
    else:
        warnings.append("zero_crossing_not_found")

    candidate = max(low, min(upper, int(candidate)))
    return SmartEndDecision(candidate, reason, round(max(0.0, min(1.0, confidence)), 3), tuple(dict.fromkeys(warnings)))


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
    instrument: str = "kick"
    automation: dict = field(default_factory=dict)
    segment_index: int | None = None
    segment_rule: str = "off"
    end_reason: str = "window"
    end_confidence: float = 0.0
    end_warnings: list[str] = field(default_factory=list)
    effective_settings: dict = field(default_factory=dict)
    # Saved analysis JSON intentionally omits the comparison samples.  Keep
    # their original length separately so fixed-window exports can be
    # reproduced after deserialization without allocating placeholder audio.
    serialized_sample_count: int | None = None

    @property
    def sample_count(self) -> int:
        shape = getattr(self.samples, "shape", None)
        if shape is not None:
            count = int(shape[0])
        else:
            try:
                count = len(self.samples)
            except TypeError:
                count = 0
        if count > 0 or self.serialized_sample_count is None:
            return count
        try:
            return max(0, int(self.serialized_sample_count))
        except (TypeError, ValueError):
            return count

    @property
    def variations(self) -> list[str]:
        values = self.automation.get("variations", []) if isinstance(self.automation, dict) else []
        return [str(value) for value in values] if isinstance(values, (list, tuple)) else []

    def to_dict(self, include_samples: bool = False) -> dict:
        data = {
            "id": self.id,
            "time": self.time,
            "sample": self.onset_sample,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "sample_count": self.sample_count,
            "overlap_warning": self.overlap_warning,
            "features": self.features,
            "instrument": self.instrument,
            "automation": self.automation,
            "variations": self.variations,
            "segment_index": self.segment_index,
            "segment_rule": self.segment_rule,
            "end_reason": self.end_reason,
            "end_confidence": self.end_confidence,
            "end_warnings": list(self.end_warnings),
            "warnings": list(self.end_warnings),
            "effective_settings": dict(self.effective_settings),
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
    smart_end: bool | Mapping[str, object] = False,
    smart_end_settings: Mapping[str, object] | None = None,
    instrument: str = "kick",
    cut_plan_mode: str = "auto",
    smart_end_apply_to_explicit: bool = False,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Hit]:
    """Extract mono comparison windows while retaining safe source endpoints.

    Comparison samples remain padded to the configured window for clustering;
    ``source_end`` is the real export boundary and is shortened only when
    ``smart_end`` is enabled.  Explicit grid/manual/pattern boundaries are
    preserved unless ``smart_end_apply_to_explicit`` is true.
    """
    if pre_roll_ms < 0 or window_ms <= 0:
        raise ValueError("pre_roll_ms must be non-negative and window_ms must be positive")
    signal = mono_signal(audio)
    total = audio.frame_count
    pre = round(audio.sample_rate * pre_roll_ms / 1000.0)
    window = max(1, round(audio.sample_rate * window_ms / 1000.0))
    canonical_instrument = normalize_instrument(instrument)
    if isinstance(smart_end, Mapping):
        merged_smart_settings = dict(smart_end)
        if isinstance(smart_end_settings, Mapping):
            merged_smart_settings.update(smart_end_settings)
        smart_end = bool(merged_smart_settings.pop("enabled", True))
        smart_end_settings = merged_smart_settings
    effective_smart_settings = resolve_smart_end_settings(canonical_instrument, smart_end_settings)
    effective_smart_settings["enabled"] = bool(smart_end)
    effective_smart_settings["apply_to_explicit"] = bool(
        smart_end_apply_to_explicit or effective_smart_settings.get("apply_to_explicit", False)
    )
    explicit_end = str(cut_plan_mode).strip().casefold() in {"grid", "manual", "pattern"}
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
        next_onset_sample = None
        if position + 1 < len(onsets):
            next_onset_sample = int(onsets[position + 1].sample)
            next_boundary = max(start, onsets[position + 1].sample - pre)
        end = min(requested_end, next_boundary)
        original_end = end
        has_next_boundary = position + 1 < len(onsets) and next_boundary <= requested_end
        end_reason = "next_attack" if has_next_boundary else "window"
        end_confidence = 0.0
        end_warnings: list[str] = []
        per_hit_settings = dict(effective_smart_settings)
        per_hit_settings["smart_end_requested"] = bool(smart_end)
        per_hit_settings["explicit_end"] = bool(explicit_end)
        per_hit_settings["hard_end_sample"] = int(original_end)
        per_hit_settings["next_attack_boundary_sample"] = int(next_boundary) if has_next_boundary else None
        per_hit_settings["next_attack_sample"] = next_onset_sample if has_next_boundary else None
        per_hit_settings["next_attack_upper_sample"] = (
            max(start, int(next_boundary) - round(audio.sample_rate * effective_smart_settings["next_attack_margin_ms"] / 1000.0))
            if has_next_boundary else None
        )
        applied = bool(smart_end) and (not explicit_end or effective_smart_settings["apply_to_explicit"])
        per_hit_settings["smart_end_applied"] = applied
        if explicit_end and not applied:
            end_reason = str(cut_plan_mode).strip().casefold()
            end_confidence = 1.0
            per_hit_settings["explicit_end_preserved"] = True
        elif applied:
            next_attack_boundary = next_boundary if has_next_boundary else None
            decision = detect_smart_end(
                signal,
                start=start,
                hard_end=end,
                sample_rate=audio.sample_rate,
                instrument=canonical_instrument,
                onset_sample=onset.sample,
                next_attack_sample=next_attack_boundary,
                settings=effective_smart_settings,
            )
            end = decision.source_end
            end_reason = decision.reason
            end_confidence = decision.confidence
            end_warnings = list(decision.warnings)
            per_hit_settings["explicit_end_preserved"] = False
        data = signal[start:end]
        samples = pad_or_trim(data, window)
        # The pre-roll itself contains the current attack.  Inspect a context
        # immediately before pre-roll so a normal ramp is not mistaken for a
        # preceding hit's tail.
        context = max(pre, round(audio.sample_rate * 20.0 / 1000.0))
        before = signal[max(0, start - context) : start]
        overlap = len(before) > 0 and rms(before) > max(1e-5, peak * overlap_threshold)
        hits.append(Hit(
            onset.id,
            onset.sample,
            onset.time,
            samples,
            start,
            end,
            overlap,
            end_reason=end_reason,
            end_confidence=end_confidence,
            end_warnings=end_warnings,
            effective_settings=per_hit_settings,
        ))
        if progress:
            progress(position + 1, len(onsets))
    return hits
