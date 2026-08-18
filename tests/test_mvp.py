import math
import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from bms_reuse.classification.classifier import classify_report
from bms_reuse.application import AnalysisCancelled, analyze_file
from bms_reuse.audio.loader import load_audio, mono_signal
from bms_reuse.cli import main as cli_main
from bms_reuse.detection.onset import Onset, detect_onsets
from bms_reuse.extraction.hit_extractor import Hit, extract_hits
from bms_reuse.export.json_exporter import write_json
from bms_reuse.export.wav_exporter import write_wav
from bms_reuse.similarity.score import compare_hits


class MvpSimilarityTest(unittest.TestCase):
    def test_gain_only_is_reusable(self):
        signal = [math.exp(-i / 35.0) * math.sin(i / 3.0) for i in range(400)]
        reference = Hit(0, 0, 0.0, signal, 0, len(signal))
        candidate = Hit(1, 0, 0.0, [sample * 0.5 for sample in signal], 0, len(signal))
        report = classify_report(compare_hits(reference, candidate, 48000))
        self.assertEqual(report.classification, "GAIN_VARIANT")
        self.assertAlmostEqual(report.gain_db, -6.0206, places=3)


class BoundaryTest(unittest.TestCase):
    def test_pcm_widths_and_stereo_round_trip(self):
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        with tempfile.TemporaryDirectory() as directory:
            for width in (1, 2, 3, 4):
                path = Path(directory) / f"mono-{width}.wav"
                write_wav(path, values, 48000, sample_width=width)
                audio = load_audio(path)
                decoded = mono_signal(audio)
                decoded = decoded.tolist() if hasattr(decoded, "tolist") else decoded
                tolerance = 2.0 / (2 ** (8 * width - 1)) if width > 1 else 2.0 / 255.0
                self.assertLess(max(abs(a - b) for a, b in zip(values, decoded)), tolerance)

            path = Path(directory) / "stereo.wav"
            source = [(-1.0, 1.0), (0.0, 0.25), (0.5, -0.5)]
            write_wav(path, source, 48000, channels=2, sample_width=3)
            audio = load_audio(path)
            decoded = audio.samples.tolist() if hasattr(audio.samples, "tolist") else audio.samples
            self.assertEqual(audio.channels, 2)
            self.assertEqual(audio.sample_width, 3)
            for expected, actual in zip(source, decoded):
                for left, right in zip(expected, actual):
                    self.assertAlmostEqual(left, right, places=5)

    def test_empty_audio_and_strict_json_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.wav"
            write_wav(path, [], 48000)
            first = analyze_file(path).to_dict()
            second = analyze_file(path).to_dict()
            self.assertEqual(first, second)
            self.assertEqual(first["summary"]["detected_hits"], 0)
            silent_path = Path(directory) / "silence.wav"
            write_wav(silent_path, [0.0] * 4800, 48000)
            self.assertEqual(analyze_file(silent_path).summary["detected_hits"], 0)
            output = Path(directory) / "empty.json"
            write_json(output, first)
            json.loads(output.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "invalid.json", {"nan": float("nan")})

    def test_truncated_wav_is_a_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.wav"
            path.write_bytes(b"not a wav")
            with self.assertRaises(ValueError):
                load_audio(path)

    def test_negative_gain_is_finite_and_not_reusable(self):
        signal = [math.exp(-i / 35.0) * math.sin(i / 3.0) for i in range(400)]
        report = classify_report(
            compare_hits(
                Hit(0, 0, 0.0, signal, 0, len(signal)),
                Hit(1, 0, 0.0, [-sample for sample in signal], 0, len(signal)),
                48000,
            )
        )
        self.assertTrue(math.isfinite(report.gain_db))
        self.assertNotIn(report.classification, {"SAME", "GAIN_VARIANT"})

    def test_onset_refinement_recovers_sample_positions(self):
        signal = [0.0] * 24000
        signal[1000] = 1.0
        signal[12000] = 1.0
        onsets = detect_onsets(signal, 48000, min_separation_ms=50.0)
        self.assertEqual([onset.sample for onset in onsets], [1000, 12000])

    def test_short_overlap_is_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.wav"
            source = [0.0] * 200
            source[20] = 1.0
            source[30] = 1.0
            audio = load_audio(write_wav(path, source, 1000))
            hits = extract_hits(audio, [Onset(0, 20, 0.02), Onset(1, 30, 0.03)], pre_roll_ms=0, window_ms=100)
            report = compare_hits(hits[0], hits[1], 1000, max_alignment_ms=5)
            self.assertTrue(report.overlap_warning)
            self.assertEqual(classify_report(report).classification, "OVERLAP")

    def test_hit_extraction_reports_progress_and_honors_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bms-progress.wav"
            audio = load_audio(write_wav(source, [0.0] * 120, 1000))
            onsets = [Onset(index, index * 20, index * 0.02) for index in range(3)]
            updates = []
            hits = extract_hits(audio, onsets, window_ms=20, progress=lambda done, total: updates.append((done, total)))
            self.assertEqual(len(hits), 3)
            self.assertEqual(updates[-1], (3, 3))
            with self.assertRaises(AnalysisCancelled):
                extract_hits(audio, onsets, window_ms=20, is_cancelled=lambda: True)

    def test_cli_fast_compare_flag_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cli-fast.wav"
            output = Path(directory) / "cli-fast.json"
            write_wav(source, [], 1000)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["analyze", str(source), "--output", str(output), "--fast-compare"]), 0)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["settings"]["compare_mode"], "fast")
            self.assertTrue(data["settings"]["fast_compare"])


if __name__ == "__main__":
    unittest.main()
