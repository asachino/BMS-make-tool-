"""Minimal PCM WAV exporters (16-bit output is broadly compatible with BMS tools)."""

from __future__ import annotations

import struct
import wave
import math
from pathlib import Path

from ..audio.loader import AudioData


def _pcm_bytes(sample: float, width: int) -> bytes:
    if not math.isfinite(sample):
        raise ValueError("WAV samples must be finite")
    if width == 1:
        value = round(max(-1.0, min(1.0, sample)) * 127.5 + 127.5)
        return struct.pack("<B", max(0, min(255, value)))
    if width == 2:
        value = round(max(-1.0, min(1.0, sample)) * 32768.0)
        return struct.pack("<h", max(-32768, min(32767, value)))
    if width == 3:
        value = round(max(-1.0, min(1.0, sample)) * 8388608.0)
        value = max(-8388608, min(8388607, value))
        return int(value).to_bytes(3, "little", signed=True)
    if width == 4:
        value = round(max(-1.0, min(1.0, sample)) * 2147483648.0)
        return struct.pack("<i", max(-2147483648, min(2147483647, value)))
    raise ValueError(f"Unsupported PCM sample width: {width} bytes")


def write_wav(path: str | Path, samples, sample_rate: int, channels: int = 1, *, sample_width: int = 2) -> Path:
    if sample_rate <= 0 or channels < 1:
        raise ValueError("sample_rate must be positive and channels must be at least one")
    if sample_width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    if channels == 1:
        for sample in samples:
            if isinstance(sample, (list, tuple)) or (hasattr(sample, "shape") and len(sample.shape) > 0):
                if len(sample) != 1:
                    raise ValueError("mono WAV samples must contain one value per frame")
                sample = sample[0]
            values.append((float(sample),))
    else:
        for row in samples:
            if len(row) != channels:
                raise ValueError("each WAV frame must contain exactly channels values")
            values.append(tuple(float(x) for x in row))
    raw = bytearray()
    for row in values:
        for sample in row:
            raw.extend(_pcm_bytes(sample, sample_width))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)
    return path


def write_hit_wavs(output_dir: str | Path, audio: AudioData, hits, plan) -> list[Path]:
    """Write one representative source window per reuse-plan cluster."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hit_by_id = {hit.id: hit for hit in hits}
    paths: list[Path] = []
    for cluster in plan.clusters:
        hit = hit_by_id[cluster.representative_hit]
        # Use the extracted source boundary.  Reading a full window from the
        # raw stem would re-introduce the next hit that extraction excluded.
        end = min(audio.frame_count, max(hit.source_start, hit.source_end))
        source = audio.samples[hit.source_start:end]
        missing = max(0, hit.sample_count - len(source))
        if missing:
            if hasattr(source, "shape"):
                import numpy as np

                source = np.concatenate((source, np.zeros((missing, audio.channels))), axis=0)
            else:
                source = list(source) + [[0.0] * audio.channels for _ in range(missing)]
        if audio.channels == 1:
            values = source[:, 0] if hasattr(source, "shape") and len(source.shape) > 1 else [row[0] for row in source]
        else:
            values = source
        path = output_dir / f"sample_{cluster.id:03d}.wav"
        write_wav(path, values, audio.sample_rate, audio.channels, sample_width=audio.sample_width)
        paths.append(path)
    return paths
