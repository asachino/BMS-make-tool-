import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from bms_reuse.audio.loader import AudioData, load_audio
from bms_reuse.application import AnalysisResult, analysis_result_from_dict, analyze_file
from bms_reuse.clustering.reuse_plan import Cluster, ReusePlan
from bms_reuse.detection.onset import Onset
from bms_reuse.extraction.hit_extractor import (
    Hit,
    detect_smart_end,
    extract_hits,
)
from bms_reuse.export.csv_exporter import write_hits_csv
from bms_reuse.export.quality import validate_exports
from bms_reuse.export.wav_exporter import write_hit_wavs, write_wav
from bms_reuse.cli import main as cli_main


def _tail_signal(duration_ms: int, *, length: int = 1000, sample_rate: int = 1000):
    values = [0.0] * length
    for offset in range(max(0, duration_ms)):
        index = 100 + offset
        if index >= length:
            break
        values[index] = 0.8 * math.sin(offset * 0.37) * math.exp(-offset / max(1.0, duration_ms / 3.0))
    return values


class SmartEndTest(unittest.TestCase):
    def test_instrument_profiles_distinguish_kick_snare_and_closed_open_hihat(self):
        kick = detect_smart_end(
            _tail_signal(120), start=100, hard_end=900, onset_sample=100,
            sample_rate=1000, instrument="kick",
        )
        snare = detect_smart_end(
            _tail_signal(220), start=100, hard_end=900, onset_sample=100,
            sample_rate=1000, instrument="snare",
        )
        closed_hihat = detect_smart_end(
            _tail_signal(35), start=100, hard_end=900, onset_sample=100,
            sample_rate=1000, instrument="hihat",
        )
        open_hihat = detect_smart_end(
            _tail_signal(300), start=100, hard_end=900, onset_sample=100,
            sample_rate=1000, instrument="hihat",
        )
        self.assertLess(kick.source_end, snare.source_end)
        self.assertLess(closed_hihat.source_end, open_hihat.source_end)
        self.assertNotIn("TAIL_CUT", closed_hihat.warnings)
        self.assertNotIn("TOO_LONG", closed_hihat.warnings)
        self.assertEqual(closed_hihat.reason, "silence")

    def test_next_attack_limit_and_zero_crossing_are_recorded(self):
        signal = [0.0] * 900
        for index in range(100, 700):
            signal[index] = 0.6 * math.sin(index * 0.17)
        decision = detect_smart_end(
            signal,
            start=100,
            hard_end=600,
            onset_sample=100,
            next_attack_sample=600,
            sample_rate=1000,
            settings={"next_attack_margin_ms": 12.0},
        )
        self.assertLessEqual(decision.source_end, 588)
        self.assertIn("NEXT_ATTACK_LIMIT", decision.warnings)
        self.assertIn("TAIL_CUT", decision.warnings)
        self.assertNotIn("TOO_LONG", decision.warnings)
        self.assertTrue(0.0 <= decision.confidence <= 1.0)
        long_tail = detect_smart_end(
            signal, start=100, hard_end=900, onset_sample=100,
            sample_rate=1000, instrument="kick",
        )
        self.assertEqual(long_tail.reason, "max_duration")
        self.assertIn("TOO_LONG", long_tail.warnings)

    def test_explicit_grid_manual_pattern_boundaries_are_preserved(self):
        sample_rate = 1000
        signal = _tail_signal(80)
        with tempfile.TemporaryDirectory() as directory:
            audio = load_audio(write_wav(Path(directory) / "explicit.wav", signal, sample_rate))
            self._assert_explicit_boundaries(audio, sample_rate)

    def _assert_explicit_boundaries(self, audio, sample_rate):
        onsets = [Onset(0, 100, 0.1), Onset(1, 300, 0.3)]
        for mode, expected in (("grid", 300), ("manual", 300), ("pattern", 300)):
            hits = extract_hits(
                audio, onsets, pre_roll_ms=0, window_ms=500,
                smart_end=True, instrument="kick", cut_plan_mode=mode,
            )
            self.assertEqual(hits[0].source_end, expected)
            self.assertEqual(hits[0].end_reason, mode)
            self.assertFalse(hits[0].effective_settings["smart_end_applied"])
        applied = extract_hits(
            audio, [Onset(0, 100, 0.1)], pre_roll_ms=0, window_ms=500,
            smart_end=True, instrument="kick", cut_plan_mode="manual",
            smart_end_apply_to_explicit=True,
        )[0]
        self.assertLess(applied.source_end, 500)
        self.assertTrue(applied.effective_settings["smart_end_applied"])

    def test_representative_wav_uses_actual_source_range_and_csv_persists_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_wav(root / "source.wav", [[0.5]] * 100, 1000)
            audio = load_audio(source)
            hit = Hit(
                0, 10, 0.01, [[0.0]] * 100, 0, 25,
                end_reason="next_attack", end_confidence=0.8,
                end_warnings=["TAIL_CUT"], effective_settings={"smart_end_requested": True, "min_tail_ms": 8.0},
            )
            plan = ReusePlan([Cluster(1, 0, [0])], [{"hit": 0, "sample_id": "sample_001", "gain_db": 0.0}])
            paths = write_hit_wavs(root / "keysounds", audio, [hit], plan, fade_in_ms=2.0, fade_out_ms=2.0)
            self.assertEqual(load_audio(paths[0]).frame_count, 25)
            csv_path = write_hits_csv(root / "events.csv", [hit], plan.events)
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["source_end"], "25")
            self.assertEqual(rows[0]["end_reason"], "next_attack")
            self.assertIn("TAIL_CUT", rows[0]["end_warnings"])
            self.assertIn("min_tail_ms", rows[0]["effective_settings"])

    def test_fixed_export_keeps_legacy_window_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_wav(root / "fixed.wav", [[0.5]] * 100, 1000)
            audio = load_audio(source)
            hit = Hit(0, 10, 0.01, [[0.0]] * 100, 0, 25, effective_settings={"smart_end_requested": False})
            plan = ReusePlan([Cluster(1, 0, [0])], [{"hit": 0, "sample_id": "sample_001", "gain_db": 0.0}])
            paths = write_hit_wavs(root / "keysounds", audio, [hit], plan)
            self.assertEqual(load_audio(paths[0]).frame_count, 100)

    def test_validate_exports_detects_smart_endpoint_frame_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_wav(root / "quality.wav", [[0.5]] * 100, 1000)
            audio = load_audio(source)
            hit = Hit(
                0, 10, 0.01, [[0.0]] * 100, 0, 25,
                effective_settings={"smart_end_requested": True},
            )
            plan = ReusePlan([Cluster(1, 0, [0])], [{"hit": 0, "sample_id": "sample_001", "gain_db": 0.0}])
            sample_dir = root / "keysounds"
            sample_dir.mkdir()
            sample = write_wav(sample_dir / "sample_001.wav", [[0.0]] * 100, 1000)
            result = AnalysisResult(
                str(source), audio.sample_rate, audio.duration, [hit], [], plan,
                {"active_hit_ids": [0]}, "source-hash",
            )
            exported = {"samples_dir": str(sample_dir), "samples": [str(sample)]}
            validation = validate_exports(result, exported)
            self.assertFalse(validation["ok"])
            self.assertFalse(validation["checks"]["sample_frames_match_endpoint"])
            self.assertEqual(validation["endpoint_mismatches"][0]["expected_frames"], 25)
            write_wav(sample, [[0.0]] * 25, 1000)
            self.assertTrue(validate_exports(result, exported)["checks"]["sample_frames_match_endpoint"])

    def test_cli_batch_smart_end_writes_json_csv_bms_bmson_and_wav_in_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            signal = [0.0] * 1200
            for offset in range(180):
                signal[100 + offset] = 0.8 * math.sin(offset * 0.31) * math.exp(-offset / 50.0)
            write_wav(input_dir / "kick.wav", signal, 1000)
            with redirect_stdout(io.StringIO()):
                code = cli_main([
                    "batch", str(input_dir), "--output-dir", str(output_dir),
                    "--bpm", "120", "--bms", "--bmson", "--smart-end",
                ])
            self.assertEqual(code, 0)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["success"], 1)
            item = manifest["items"][0]
            self.assertTrue(item["ok"])
            outputs = item["outputs"]
            data = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
            self.assertTrue(data["metadata"]["smart_end"]["enabled"])
            self.assertIn("endpoint_fields", data["exports"])
            self.assertTrue(Path(outputs["csv"]).is_file())
            self.assertTrue(Path(outputs["bms"]).is_file())
            self.assertTrue(Path(outputs["bmson"]).is_file())
            self.assertTrue(Path(outputs["samples"][0]).is_file())
            hit_by_id = {int(hit["id"]): hit for hit in data["hits"]}
            representative = hit_by_id[int(data["clusters"][0]["representative_hit"])]
            expected_frames = int(representative["source_end"]) - int(representative["source_start"])
            self.assertEqual(load_audio(outputs["samples"][0]).frame_count, expected_frames)
            self.assertIn("#WAV01", Path(outputs["bms"]).read_text(encoding="utf-8"))
            bmson = json.loads(Path(outputs["bmson"]).read_text(encoding="utf-8"))
            self.assertEqual(bmson["version"], "1.0.0")
            self.assertTrue(bmson["sound_channels"])

    def test_analysis_json_roundtrip_keeps_smart_endpoint_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = [0.0] * 800
            for offset in range(140):
                signal[100 + offset] = 0.8 * math.sin(offset * 0.31) * math.exp(-offset / 45.0)
            source = write_wav(root / "kick.wav", signal, 1000)
            result = analyze_file(source, onset_threshold=0.1, min_separation_ms=30.0, smart_end=True)
            data = result.to_dict()
            self.assertTrue(data["metadata"]["smart_end"]["enabled"])
            self.assertTrue(data["hits"])
            self.assertIn("end_reason", data["hits"][0])
            self.assertIn("effective_settings", data["hits"][0])
            restored = analysis_result_from_dict(data)
            self.assertEqual(restored.hits[0].source_end, result.hits[0].source_end)
            self.assertEqual(restored.hits[0].end_reason, result.hits[0].end_reason)
            self.assertEqual(restored.settings["smart_end_settings"], result.settings["smart_end_settings"])

    def test_fixed_analysis_json_roundtrip_restores_window_frames_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = [0.0] * 900
            for offset in range(80):
                signal[100 + offset] = 0.8 * math.sin(offset * 0.31)
            source = write_wav(root / "fixed_roundtrip.wav", signal, 1000)
            result = analyze_file(
                source,
                onset_threshold=0.1,
                min_separation_ms=30.0,
                window_ms=400.0,
                smart_end=False,
            )
            self.assertTrue(result.hits)
            data = result.to_dict()
            # Exercise the source/settings fallback used by older JSON files.
            data["hits"][0].pop("sample_count", None)
            restored = analysis_result_from_dict(data)
            self.assertEqual(restored.hits[0].sample_count, 400)
            sample_dir = root / "keysounds"
            audio = load_audio(source)
            paths = write_hit_wavs(sample_dir, audio, restored.hits, restored.plan)
            self.assertEqual(load_audio(paths[0]).frame_count, 400)
            validation = validate_exports(
                restored,
                {"samples_dir": str(sample_dir), "samples": [str(path) for path in paths]},
            )
            self.assertTrue(validation["ok"], validation)

    def test_cli_preset_filters_gui_metadata_and_merges_advanced_smart_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = [0.0] * 600
            for offset in range(80):
                signal[100 + offset] = 0.8 * math.sin(offset * 0.31)
            source = write_wav(root / "preset.wav", signal, 1000)
            preset = root / "gui.preset.json"
            preset.write_text(json.dumps({
                "version": 1,
                "settings": {
                    "smart_end": True,
                    "smart_end_settings": {"silence_ms": 44.0},
                    "smart_end_advanced": {
                        "settings": {"silence_ms": 13.0, "min_tail_ms": 7.0},
                        "unknown_gui_value": "ignore me",
                    },
                    "unknown_gui_metadata": {"display": "ignore me"},
                },
            }, ensure_ascii=False), encoding="utf-8")
            output = root / "preset.bra.json"
            with redirect_stdout(io.StringIO()):
                code = cli_main([
                    "analyze", str(source), "--preset-in", str(preset),
                    "--output", str(output), "--no-smart-end",
                ])
            self.assertEqual(code, 0)
            data = json.loads(output.read_text(encoding="utf-8"))
            # The explicit CLI switch still wins, while the nested GUI value
            # survives into the backend's resolved smart-end settings.
            self.assertFalse(data["settings"]["smart_end"])
            self.assertEqual(data["settings"]["smart_end_settings"]["silence_ms"], 13.0)
            self.assertNotIn("smart_end_advanced", data["settings"])
            self.assertNotIn("unknown_gui_metadata", data["settings"])

    def test_cli_preset_flat_settings_remain_backward_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_wav(root / "legacy.wav", [0.0] * 500, 1000)
            preset = root / "legacy.preset.json"
            preset.write_text(json.dumps({
                "settings": {"window_ms": 400.0, "smart_end": False, "legacy_unknown": True},
            }), encoding="utf-8")
            output = root / "legacy.bra.json"
            with redirect_stdout(io.StringIO()):
                code = cli_main([
                    "analyze", str(source), "--preset-in", str(preset),
                    "--output", str(output),
                ])
            self.assertEqual(code, 0)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["settings"]["window_ms"], 400.0)
            self.assertFalse(data["settings"]["smart_end"])


if __name__ == "__main__":
    unittest.main()
