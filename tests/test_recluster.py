import json
import tempfile
import unittest
from pathlib import Path

from bms_reuse.application import AnalysisResult, recluster_result
from bms_reuse.clustering.reuse_plan import Cluster, ReusePlan
from bms_reuse.cli import main as cli_main
from bms_reuse.extraction.hit_extractor import Hit
from bms_reuse.export.wav_exporter import write_wav
from bms_reuse.similarity.score import SimilarityReport


def _report(reference_id, candidate_id, waveform, spectral, gain_db=0.0):
    return SimilarityReport(
        reference_id, candidate_id, waveform, waveform, gain_db, spectral,
        waveform, waveform, waveform, 0,
    )


def _result(reports, settings=None, exports=None):
    hits = [Hit(index, index, float(index), [0.0], index, index + 1) for index in range(3)]
    plan = ReusePlan(
        [Cluster(1, 0, [0, 1]), Cluster(2, 2, [2])],
        [
            {"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
            {"hit": 1, "time": 1.0, "sample_id": "sample_001", "gain_db": 0.0},
            {"hit": 2, "time": 2.0, "sample_id": "sample_002", "gain_db": 0.0},
        ],
    )
    merged = {"threshold": 0.95, "spectral_threshold": 0.94, "review_overrides": {}, "review_targets": {}, "excluded_hits": []}
    merged.update(settings or {})
    if exports:
        merged["exports"] = exports
    return AnalysisResult("stem.wav", 1000, 3.0, hits, reports, plan, merged, "source-hash")


class ReclusterTest(unittest.TestCase):
    def test_named_and_continuous_levels_use_saved_reports_only(self):
        reports = [_report(0, 1, 0.99, 0.99), _report(0, 2, 0.91, 0.90), _report(1, 2, 0.91, 0.90)]
        for level, expected in (("strict", 2), ("balanced", 2), ("aggressive", 1), (0.92, 2)):
            result = _result(reports)
            recluster_result(result, reuse_level=level, reexport=False)
            self.assertEqual(result.summary["required_samples"], expected)
            self.assertEqual(len(result.plan.events), result.summary["detected_hits"])
            self.assertEqual(result.to_dict()["recluster"]["profile"], "custom" if isinstance(level, float) else level)
        result = _result(reports)
        recluster_result(result, threshold=0.92, reexport=False)
        self.assertEqual(result.to_dict()["recluster"]["profile"], "custom")

    def test_feature_fallback_handles_same_and_different_kicks_without_reports(self):
        result = _result([])
        feature_set = {"centroid_hz": 80, "rolloff_hz": 120, "zcr": 0.1, "attack_ms": 4, "tail_energy": 0.2}
        result.hits[0].features = dict(feature_set)
        result.hits[1].features = dict(feature_set)
        result.hits[2].features = {**feature_set, "centroid_hz": 800}
        recluster_result(result, profile="balanced", reexport=False)
        self.assertEqual([cluster.hit_ids for cluster in result.plan.clusters], [[0, 1], [2]])
        self.assertEqual(result.summary["comparisons"], 2)

    def test_review_overrides_win_and_ignore_is_removed(self):
        reports = [_report(0, 1, 0.80, 0.80), _report(0, 2, 0.80, 0.80), _report(0, 3, 0.80, 0.80)]
        result = _result(
            reports,
            {
                "review_overrides": {"1": "D", "2": "S", "3": "I"},
                "review_targets": {"2": 1},
            },
        )
        recluster_result(result, profile="aggressive", reexport=False)
        self.assertEqual([cluster.hit_ids for cluster in result.plan.clusters], [[0, 2], [1]])
        self.assertEqual([event["hit"] for event in result.plan.events], [0, 1, 2])
        self.assertEqual(result.summary["detected_hits"], 3)
        self.assertEqual(result.summary["required_samples"], 2)
        self.assertEqual(result.to_dict()["review"]["excluded_hits"], [3])

    def test_d_override_keeps_hit_in_an_independent_cluster(self):
        reports = [_report(0, 1, 0.99, 0.99), _report(0, 2, 0.99, 0.99), _report(1, 2, 0.99, 0.99)]
        result = _result(reports, {"review_overrides": {"1": "D"}})
        recluster_result(result, profile="aggressive", reexport=False)
        self.assertIn([1], [cluster.hit_ids for cluster in result.plan.clusters])

    def test_recluster_reexports_one_wav_per_cluster_and_removes_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stem.wav"
            write_wav(source, [0.0] * 100, 1000)
            exports_dir = root / "keysounds"
            exports_dir.mkdir()
            write_wav(exports_dir / "sample_001.wav", [0.0], 1000)
            write_wav(exports_dir / "sample_002.wav", [0.0], 1000)
            result = _result(
                [_report(0, 1, 0.99, 0.99), _report(0, 2, 0.99, 0.99)],
                exports={
                    "samples_dir": str(exports_dir),
                    "samples": [str(exports_dir / "sample_001.wav"), str(exports_dir / "sample_002.wav")],
                },
            )
            result.source = str(source)
            recluster_result(result, profile="aggressive")
            self.assertEqual(len(result.plan.clusters), 1)
            self.assertEqual(len(result.settings["exports"]["samples"]), 1)
            self.assertEqual(len(list(exports_dir.glob("sample_*.wav"))), 1)
            self.assertTrue(result.settings["validation"]["ok"])

    def test_cli_recluster_reads_saved_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_json = root / "source.bra.json"
            output_json = root / "reclustered.bra.json"
            source_json.write_text(json.dumps(_result([_report(0, 1, 0.99, 0.99), _report(0, 2, 0.91, 0.90)]).to_dict()), encoding="utf-8")
            self.assertEqual(
                cli_main(["recluster", str(source_json), "--output", str(output_json), "--reuse-level", "aggressive", "--no-reexport"]),
                0,
            )
            data = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(data["recluster"]["profile"], "aggressive")
            self.assertEqual(data["summary"]["required_samples"], 1)


if __name__ == "__main__":
    unittest.main()
