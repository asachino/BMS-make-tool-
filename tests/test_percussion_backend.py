import math
import tempfile
import unittest
from pathlib import Path

from bms_reuse.application import analysis_result_from_dict, analyze_file
from bms_reuse.application import AnalysisResult, recluster_result
from bms_reuse.classification.classifier import classify_report
from bms_reuse.detection.loop_rules import build_cut_onsets
from bms_reuse.detection.onset import Onset
from bms_reuse.extraction.hit_extractor import Hit
from bms_reuse.export.wav_exporter import write_wav
from bms_reuse.export.quality import validate_exports
from bms_reuse.features.automation import detect_automation
from bms_reuse.features.feature_extractor import extract_features
from bms_reuse.similarity.score import compare_hits
from bms_reuse.clustering.reuse_plan import Cluster, ReusePlan
from bms_reuse.similarity.score import SimilarityReport


class PercussionBackendTest(unittest.TestCase):
    def test_instrument_features_and_cross_instrument_gate(self):
        signal = [math.sin(index / 3.0) * math.exp(-index / 500.0) for index in range(512)]
        for instrument in ("kick", "snare", "hihat", "other"):
            features = extract_features(signal, 48000, instrument=instrument)
            self.assertEqual(features["instrument"], instrument)
            self.assertIn("band_low_ratio", features)
            self.assertIn("transient_ratio", features)

        left = Hit(0, 0, 0.0, signal, 0, len(signal), instrument="kick")
        right = Hit(1, 0, 0.0, signal, 0, len(signal), instrument="snare")
        report = classify_report(compare_hits(left, right, 48000))
        self.assertFalse(report.instrument_compatible)
        self.assertEqual(report.classification, "DIFFERENT")

    def test_automation_flags_stereo_and_chopped_variation(self):
        stereo_signal = [math.sin(index / 4.0) * 0.7 for index in range(400)]
        stereo = [
            (value, value * 0.05) if index < 200 else (value * 0.05, value)
            for index, value in enumerate(stereo_signal)
        ]
        stereo_result = detect_automation(stereo_signal, 4000, channels=stereo)
        self.assertIn("STEREO", stereo_result["variations"])

        chopped = []
        for amplitude in (1.0, 0.01, 1.0, 0.01, 1.0, 0.01, 1.0, 0.01):
            chopped.extend(math.sin(index / 2.0) * amplitude for index in range(100))
        chopped_result = detect_automation(chopped, 4000, segments=8)
        self.assertIn("DENSITY", chopped_result["variations"])
        self.assertGreater(chopped_result["chop_count"], 0)

    def test_manual_cut_plan_is_deterministic(self):
        onsets, points = build_cut_onsets(
            [Onset(0, 20, 0.02)],
            4000,
            4000,
            rule="manual",
            points=[0.0, 0.25, 0.5, 0.75],
        )
        self.assertEqual([onset.sample for onset in onsets], [0, 1000, 2000, 3000])
        self.assertEqual(points, [0.0, 0.25, 0.5, 0.75])

    def test_analysis_roundtrip_keeps_profile_automation_and_cut_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "perc.wav"
            values = [0.0] * 4000
            for position in (0, 1000, 2000, 3000):
                for index in range(20):
                    values[position + index] = math.exp(-index / 5.0)
            write_wav(source, values, 4000)
            result = analyze_file(
                source,
                instrument="hihat",
                cut_plan={"mode": "manual", "points": [0.0, 0.25, 0.5, 0.75]},
                automation_detection=False,
            )
            self.assertEqual(len(result.hits), 4)
            self.assertEqual(result.settings["instrument_profile"]["name"], "hihat")
            self.assertEqual(result.settings["cut_plan"]["mode"], "manual")
            self.assertEqual(result.settings["loop_boundaries_sec"], [0.0, 0.25, 0.5, 0.75])
            restored = analysis_result_from_dict(result.to_dict())
            self.assertEqual(restored.hits[0].instrument, "hihat")
            self.assertEqual(restored.settings["cut_plan"]["mode"], "manual")
            self.assertEqual(restored.summary["detected_hits"], 4)

    def test_recluster_keeps_ignored_hit_reversible_but_excludes_exports(self):
        hits = [
            Hit(0, 0, 0.0, [1.0, 0.0], 0, 2),
            Hit(1, 1, 0.1, [1.0, 0.0], 1, 3),
        ]
        report = SimilarityReport(0, 1, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0)
        result = AnalysisResult(
            "stem.wav", 1000, 1.0, hits, [report],
            ReusePlan([Cluster(1, 0, [0, 1])], [
                {"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
                {"hit": 1, "time": 0.1, "sample_id": "sample_001", "gain_db": 0.0},
            ]),
            {"threshold": 0.95, "spectral_threshold": 0.94, "review_overrides": {"1": "I"}},
            "source",
        )
        recluster_result(result, reexport=False)
        self.assertEqual([hit.id for hit in result.hits], [0, 1])
        self.assertEqual(result.settings["active_hit_ids"], [0])
        self.assertEqual(result.summary["detected_hits"], 1)
        self.assertEqual([event["hit"] for event in result.plan.events], [0])
        validation = validate_exports(result, {"samples": [], "samples_dir": None})
        self.assertTrue(validation["checks"]["event_count_matches_hits"])


if __name__ == "__main__":
    unittest.main()
