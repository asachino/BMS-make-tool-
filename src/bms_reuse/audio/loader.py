"""WAV loading and channel handling."""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from .._numeric import np


@dataclass(frozen=True)
class AudioData:
    sample_rate: int
    channels: int
    samples: object  # shape: (frames, channels), float in [-1, 1]
    sample_width: int = 2

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0]) if np is not None else len(self.samples)

    @property
    def duration(self) -> float:
        return self.frame_count / self.sample_rate if self.sample_rate else 0.0


def _decode_pcm(raw: bytes, width: int, channels: int) -> list[list[float]]:
    if channels < 1:
        raise ValueError("channels must be at least one")
    if width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported PCM sample width: {width} bytes")
    if len(raw) % (width * channels) != 0:
        raise ValueError("WAV data is not aligned to complete PCM frames")
    if width == 1:
        values = [b / 128.0 - 1.0 for b in raw]
    elif width == 2:
        count = len(raw) // 2
        values = [x / 32768.0 for x in struct.unpack(f"<{count}h", raw)]
    elif width == 3:
        values = []
        for i in range(0, len(raw) - 2, 3):
            value = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            values.append(value / 8388608.0)
    elif width == 4:
        count = len(raw) // 4
        values = [x / 2147483648.0 for x in struct.unpack(f"<{count}i", raw)]
    else:
        raise ValueError(f"Unsupported PCM sample width: {width} bytes")
    frames = [values[i : i + channels] for i in range(0, len(values), channels)]
    return frames


def load_audio(path: str | Path) -> AudioData:
    """Load an uncompressed PCM WAV into normalized floating point samples."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            width = wav.getsampwidth()
            if channels < 1 or rate < 1:
                raise ValueError("WAV must contain at least one channel and a positive sample rate")
            if wav.getcomptype() != "NONE":
                raise ValueError(f"Compressed WAV is not supported: {wav.getcomptype()}")
            frames = _decode_pcm(wav.readframes(wav.getnframes()), width, channels)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Could not read WAV {path}: {exc}") from exc
    if np is not None:
        # Keep empty WAVs two-dimensional, matching non-empty `(frames, channels)` data.
        samples = np.asarray(frames, dtype=float).reshape((-1, channels))
    else:
        samples = frames
    return AudioData(rate, channels, samples, width)


def mono_signal(audio: AudioData):
    """Return mono analysis audio; stereo is averaged without changing the source."""
    if audio.channels == 1:
        return audio.samples[:, 0] if np is not None else [row[0] for row in audio.samples]
    if np is not None:
        return audio.samples.mean(axis=1)
    return [sum(row) / len(row) for row in audio.samples]
