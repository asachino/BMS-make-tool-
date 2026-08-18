import json
import tempfile
import unittest
from pathlib import Path

from bms_reuse.clustering.reuse_plan import Cluster, ReusePlan
from bms_reuse.export.bms_exporter import write_bms
from bms_reuse.export.bmson_exporter import write_bmson
from bms_reuse.export.quality import validate_exports
from bms_reuse.batch import run_batch
from bms_reuse.project.presets import load_preset, save_preset
from bms_reuse.application import AnalysisResult, exclude_hit, analyze_file
from bms_reuse.extraction.hit_extractor import Hit
from bms_reuse.export.wav_exporter import write_wav


class BmsFeatureTest(unittest.TestCase):
    def test_bms_bmson_and_preset_round_trip(self):
        plan = ReusePlan([Cluster(1, 0, [0, 1])], [{"hit": 0, "time": 0.5, "sample_id": "sample_001", "gain_db": -3.0}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bms = write_bms(root / "out.bms", plan, bpm=120)
            text = bms.read_text(encoding="utf-8")
            self.assertIn("#BPM 120", text)
            self.assertIn("#WAV01 sample_001.wav", text)
            self.assertIn("#00001:", text)
            self.assertIn("WARNING: channel 01", text)
            (root / "keysounds").mkdir()
            write_wav(root / "keysounds" / "sample_001.wav", [0.0], 1000)
            bms_with_prefix = write_bms(root / "keys.bms", plan, bpm=120, wav_prefix="keysounds/")
            self.assertIn("#WAV01 keysounds/sample_001.wav", bms_with_prefix.read_text(encoding="utf-8"))
            bmson = write_bmson(root / "out.bmson", plan, bpm=120)
            data = json.loads(bmson.read_text(encoding="utf-8"))
            self.assertEqual(data["info"]["init_bpm"], 120.0)
            self.assertEqual(data["info"]["resolution"], 240)
            self.assertEqual(data["version"], "1.0.0")
            self.assertIn("stop_events", data)
            self.assertEqual(data["sound_channels"][0]["name"], "sample_001.wav")
            self.assertEqual(set(data["sound_channels"][0]["notes"][0]), {"x", "y", "l", "c"})
            self.assertIs(data["sound_channels"][0]["notes"][0]["c"], False)
            self.assertNotIn("/", data["sound_channels"][0]["name"])
            preset = save_preset(root / "preset.json", {"bpm": 174, "subdivision": 16})
            self.assertEqual(load_preset(preset)["bpm"], 174)

    def test_preset_ignores_unknown_keys_and_batch_rejects_missing_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preset.json"
            path.write_text(json.dumps({"version": 1, "settings": {"threshold": 0.9, "unknown": "bad"}}), encoding="utf-8")
            self.assertNotIn("unknown", load_preset(path))
            with self.assertRaises(ValueError):
                run_batch(Path(directory) / "missing")

    def test_reproducibility_hash_ignores_runtime_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silent.wav"
            write_wav(source, [0.0] * 1000, 1000)
            first = analyze_file(source).to_dict()
            second = analyze_file(source).to_dict()
            self.assertEqual(first["metadata"]["reproducibility_hash"], second["metadata"]["reproducibility_hash"])

    def test_review_ignore_removes_plan_event_and_updates_hash(self):
        hits = [Hit(0, 0, 0.0, [1.0], 0, 1), Hit(1, 1, 0.1, [1.0], 1, 2)]
        plan = ReusePlan([Cluster(1, 0, [0, 1])], [
            {"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
            {"hit": 1, "time": 0.1, "sample_id": "sample_001", "gain_db": 0.0},
        ])
        result = AnalysisResult("stem.wav", 1000, 1.0, hits, [], plan, {"analysis_version": "0.3.0"}, "source")
        before = result.settings.get("reproducibility_hash")
        exclude_hit(result, 1)
        self.assertEqual([hit.id for hit in result.hits], [0])
        self.assertEqual([event["hit"] for event in result.plan.events], [0])
        self.assertEqual(result.to_dict()["review"]["excluded_hits"], [1])
        self.assertNotEqual(before, result.settings["reproducibility_hash"])

    def test_validate_exports_detects_extra_wav(self):
        hits = [Hit(0, 0, 0.0, [1.0], 0, 1)]
        plan = ReusePlan([Cluster(1, 0, [0])], [{"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0}])
        result = AnalysisResult("stem.wav", 1000, 1.0, hits, [], plan, {}, "source")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_wav(root / "sample_001.wav", [0.0], 1000)
            write_wav(root / "extra.wav", [0.0], 1000)
            validation = validate_exports(result, {"samples_dir": str(root), "samples": [str(root / "sample_001.wav")]})
            self.assertFalse(validation["ok"])
            self.assertFalse(validation["checks"]["sample_folder_has_no_extra_wav"])

    def test_validate_exports_treats_empty_optional_samples_as_unrequested(self):
        hits = [Hit(0, 0, 0.0, [1.0], 0, 1)]
        plan = ReusePlan([Cluster(1, 0, [0])], [{"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0}])
        result = AnalysisResult("stem.wav", 1000, 1.0, hits, [], plan, {}, "source")
        validation = validate_exports(result, {"samples": [], "samples_dir": None})
        self.assertTrue(validation["ok"])
        self.assertTrue(validation["checks"]["sample_folder_exists"])

    def test_validate_exports_detects_missing_bms_reference(self):
        hits = [Hit(0, 0, 0.0, [1.0], 0, 1)]
        plan = ReusePlan([Cluster(1, 0, [0])], [{"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0}])
        result = AnalysisResult("stem.wav", 1000, 1.0, hits, [], plan, {}, "source")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bms = root / "chart.bms"
            bms.write_text("#WAV01 missing.wav\n#00001:01\n", encoding="utf-8")
            validation = validate_exports(result, {"bms": str(bms)})
            self.assertFalse(validation["ok"])
            self.assertFalse(validation["checks"]["bms_references_exist"])


if __name__ == "__main__":
    unittest.main()
