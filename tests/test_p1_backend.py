import json
import math
import tempfile
import unittest
from pathlib import Path

from bms_reuse.application import (
    AnalysisResult,
    analyze_file,
    relative_sample_prefix_for_export,
    recluster_result,
    set_review_state,
)
from bms_reuse.clustering.reuse_plan import Cluster, ReusePlan
from bms_reuse.extraction.hit_extractor import Hit
from bms_reuse.export.bms_exporter import write_bms
from bms_reuse.export.bmson_exporter import write_bmson
from bms_reuse.export.quality import validate_exports
from bms_reuse.export.wav_exporter import write_wav
from bms_reuse.features.automation import detect_automation
from bms_reuse.similarity.score import SimilarityReport


def _report(reference_id, candidate_id, *, compatible=True, gain_db=0.0):
    return SimilarityReport(
        reference_id,
        candidate_id,
        1.0,
        1.0,
        gain_db,
        1.0,
        1.0,
        1.0,
        1.0,
        0,
        instrument_compatible=compatible,
    )


def _result(hits, reports, clusters, events=None, settings=None):
    if events is None:
        events = [
            {
                "hit": hit.id,
                "time": hit.time,
                "sample_id": f"sample_{next(cluster.id for cluster in clusters if hit.id in cluster.hit_ids):03d}",
                "gain_db": 0.0,
            }
            for hit in hits
        ]
    base = {
        "threshold": 0.95,
        "spectral_threshold": 0.94,
        "review_overrides": {},
        "review_targets": {},
        "excluded_hits": [],
    }
    base.update(settings or {})
    return AnalysisResult(
        "stem.wav",
        1000,
        3.0,
        hits,
        reports,
        ReusePlan(clusters, events),
        base,
        "source-hash",
    )


class P1BackendTest(unittest.TestCase):
    def test_saved_instrument_gate_blocks_merge_but_explicit_s_can_override(self):
        hits = [
            Hit(0, 0, 0.0, [1.0], 0, 1, instrument="kick"),
            Hit(1, 1, 1.0, [1.0], 1, 2, instrument="snare"),
        ]
        result = _result(hits, [_report(0, 1, compatible=False)], [
            Cluster(1, 0, [0]),
            Cluster(2, 1, [1]),
        ])
        recluster_result(result, profile="aggressive", reexport=False)
        self.assertEqual([cluster.hit_ids for cluster in result.plan.clusters], [[0], [1]])

        stale = _result(hits, [_report(0, 1, compatible=True)], [
            Cluster(1, 0, [0]),
            Cluster(2, 1, [1]),
        ])
        recluster_result(stale, profile="aggressive", reexport=False)
        self.assertEqual([cluster.hit_ids for cluster in stale.plan.clusters], [[0], [1]])
        self.assertFalse(stale.comparisons[0].instrument_compatible)

        result.settings["review_overrides"] = {"1": "S"}
        result.settings["review_targets"] = {"1": 1}
        recluster_result(result, profile="aggressive", reexport=False)
        self.assertEqual([cluster.hit_ids for cluster in result.plan.clusters], [[0, 1]])

    def test_review_api_rebuilds_representatives_and_restores_ignored_hit(self):
        hits = [
            Hit(0, 0, 0.0, [1.0], 0, 1),
            Hit(1, 1, 1.0, [1.0], 1, 2),
        ]
        result = _result(hits, [_report(0, 1)], [Cluster(1, 0, [0, 1])])

        set_review_state(result, 0, "D", reexport=False)
        self.assertEqual([cluster.hit_ids for cluster in result.plan.clusters], [[0], [1]])
        self.assertEqual([cluster.representative_hit for cluster in result.plan.clusters], [0, 1])

        set_review_state(result, 0, "I", reexport=False)
        self.assertEqual(result.settings["active_hit_ids"], [1])
        self.assertEqual([event["hit"] for event in result.plan.events], [1])
        self.assertEqual([cluster.representative_hit for cluster in result.plan.clusters], [1])

        set_review_state(result, 1, "I", reexport=False)
        self.assertEqual(result.settings["active_hit_ids"], [])
        set_review_state(result, 0, None, reexport=False)
        self.assertEqual(result.settings["active_hit_ids"], [0])
        self.assertEqual([event["hit"] for event in result.plan.events], [0])

        set_review_state(result, 1, None, reexport=False)
        set_review_state(result, 0, None, reexport=False)
        self.assertEqual(result.settings["active_hit_ids"], [0, 1])
        self.assertEqual([event["hit"] for event in result.plan.events], [0, 1])
        self.assertEqual(result.plan.clusters[0].representative_hit, 0)

    def test_bmson_external_sample_folder_and_bms_collision_are_validated(self):
        hits = [
            Hit(0, 0, 0.0, [1.0], 0, 1),
            Hit(1, 1, 0.0, [1.0], 1, 2),
        ]
        plan = ReusePlan(
            [Cluster(1, 0, [0]), Cluster(2, 1, [1])],
            [
                {"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
                {"hit": 1, "time": 0.0, "sample_id": "sample_002", "gain_db": 0.0},
            ],
        )
        result = AnalysisResult("stem.wav", 1000, 1.0, hits, [], plan, {}, "source")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = root / "keysounds"
            sample_dir.mkdir()
            samples = [sample_dir / "sample_001.wav", sample_dir / "sample_002.wav"]
            for path in samples:
                write_wav(path, [0.0], 1000)

            bms_path = root / "chart" / "chart.bms"
            bms = write_bms(
                bms_path,
                plan,
                bpm=120,
                wav_prefix=relative_sample_prefix_for_export(bms_path, sample_dir),
            )
            bms_text = bms.read_text(encoding="utf-8")
            self.assertIn("衝突", bms_text)
            self.assertNotIn("後続を省略", bms_text)
            bms_validation = validate_exports(result, {
                "bms": str(bms),
                "samples": [str(path) for path in samples],
                "samples_dir": str(sample_dir),
            })
            self.assertTrue(bms_validation["checks"]["bms_event_count_matches_hits"])
            self.assertTrue(bms_validation["checks"]["bms_grid_collisions_preserved"])
            self.assertTrue(bms_validation["ok"])

            bmson_path = root / "chart" / "chart.bmson"
            bmson = write_bmson(
                bmson_path,
                plan,
                bpm=120,
                wav_prefix=relative_sample_prefix_for_export(bmson_path, sample_dir),
            )
            data = json.loads(bmson.read_text(encoding="utf-8"))
            self.assertEqual(data["sound_channels"][0]["name"], "../keysounds/sample_001.wav")
            bmson_validation = validate_exports(result, {
                "bmson": str(bmson),
                "samples": [str(path) for path in samples],
                "samples_dir": str(sample_dir),
            })
            self.assertTrue(bmson_validation["checks"]["bmson_references_exist"])
            self.assertTrue(bmson_validation["ok"])

    def test_all_cut_plan_modes_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cut.wav"
            values = [0.0] * 2200
            for start in (100, 600, 1100, 1600):
                for index in range(20):
                    values[start + index] = math.exp(-index / 5.0)
            write_wav(source, values, 1000)
            common = {
                "window_ms": 100,
                "min_separation_ms": 20,
                "onset_threshold": 0.05,
                "automation_detection": False,
            }
            cases = {
                "auto": {},
                "grid": {"bpm": 120, "cut_plan": {"mode": "grid", "rule": "beats", "beats": 1}},
                "manual": {"cut_plan": {"mode": "manual", "points": [0.1, 0.6, 1.1]}},
                "pattern": {"cut_plan": {"mode": "pattern", "intervals": [0.25, 0.5]}},
            }
            for mode, options in cases.items():
                result = analyze_file(source, **common, **options)
                self.assertEqual(result.settings["cut_plan"]["mode"], mode)
                self.assertGreater(len(result.hits), 0)

    def test_variation_flags_cover_gain_timbre_tail_density_stereo_and_off_grid(self):
        sample_rate = 1000
        gain = []
        for amplitude in (1.0, 0.1, 1.0, 0.2):
            gain.extend(amplitude * math.sin(2 * math.pi * 40 * index / sample_rate) for index in range(100))
        self.assertIn("GAIN", detect_automation(gain, sample_rate)["variations"])

        timbre = []
        for frequency in (30, 30, 300, 300):
            timbre.extend(math.sin(2 * math.pi * frequency * index / sample_rate) for index in range(100))
        self.assertIn("TIMBRE", detect_automation(timbre, sample_rate, timbre_threshold=0.05)["variations"])

        tail = []
        for amplitude in (1.0, 0.5, 0.2, 0.05):
            tail.extend(amplitude * math.sin(2 * math.pi * 40 * index / sample_rate) for index in range(100))
        self.assertIn("TAIL", detect_automation(tail, sample_rate)["variations"])

        chopped = []
        for amplitude in (1.0, 0.01, 1.0, 0.01, 1.0, 0.01, 1.0, 0.01):
            chopped.extend(amplitude * math.sin(2 * math.pi * 40 * index / sample_rate) for index in range(50))
        self.assertIn("DENSITY", detect_automation(chopped, sample_rate, segments=8)["variations"])

        base = [math.sin(2 * math.pi * 40 * index / sample_rate) for index in range(400)]
        stereo = [(value, value * 0.05) if index < 200 else (value * 0.05, value) for index, value in enumerate(base)]
        self.assertIn("STEREO", detect_automation(base, sample_rate, channels=stereo)["variations"])

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "off-grid.wav"
            values = [0.0] * 1200
            for start in (140, 640):
                for index in range(20):
                    values[start + index] = math.exp(-index / 4.0)
            write_wav(source, values, sample_rate)
            result = analyze_file(
                source,
                bpm=120,
                subdivision=4,
                cut_plan={"mode": "manual", "points": [0.14, 0.64]},
                window_ms=100,
                min_separation_ms=20,
                onset_threshold=0.05,
            )
            self.assertTrue(any("OFF_GRID" in (hit.automation or {}).get("variations", []) for hit in result.hits))

    def test_review_reexport_keeps_json_csv_bms_bmson_and_wav_in_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stem.wav"
            write_wav(source, [0.0] * 1000, 1000)
            sample_dir = root / "keysounds"
            sample_dir.mkdir()
            json_path = root / "chart.json"
            csv_path = root / "chart.csv"
            bms_path = root / "chart.bms"
            bmson_path = root / "charts" / "chart.bmson"
            initial_samples = sample_dir / "sample_001.wav"
            write_wav(initial_samples, [0.0], 1000)
            hits = [
                Hit(0, 0, 0.0, [1.0], 0, 1),
                Hit(1, 1, 0.1, [1.0], 1, 2),
            ]
            plan = ReusePlan(
                [Cluster(1, 0, [0, 1])],
                [
                    {"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
                    {"hit": 1, "time": 0.1, "sample_id": "sample_001", "gain_db": 0.0},
                ],
            )
            result = AnalysisResult(
                str(source),
                1000,
                1.0,
                hits,
                [_report(0, 1)],
                plan,
                {
                    "bpm": 120,
                    "subdivision": 16,
                    "review_overrides": {"1": "I"},
                    "exports": {
                        "json": str(json_path),
                        "csv": str(csv_path),
                        "bms": str(bms_path),
                        "bmson": str(bmson_path),
                        "samples_dir": str(sample_dir),
                        "samples": [str(initial_samples)],
                    },
                },
                "source-hash",
            )
            recluster_result(result, reexport=True)
            exports = result.settings["exports"]
            self.assertEqual(result.settings["active_hit_ids"], [0])
            self.assertEqual(len(result.plan.events), 1)
            self.assertEqual(len(exports["samples"]), result.plan.required_samples)
            self.assertEqual(len(list(sample_dir.glob("sample_*.wav"))), result.plan.required_samples)
            bms_text = Path(exports["bms"]).read_text(encoding="utf-8")
            self.assertIn("#WAV01", bms_text)
            self.assertTrue(result.settings["validation"]["checks"]["bms_event_count_matches_hits"])
            csv_lines = Path(exports["csv"]).read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(csv_lines), 2)
            bmson_data = json.loads(Path(exports["bmson"]).read_text(encoding="utf-8"))
            self.assertEqual(len(bmson_data["sound_channels"][0]["notes"]), 1)
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["review"]["active_hit_ids"], [0])
            self.assertEqual(saved["summary"]["required_samples"], 1)
            self.assertTrue(result.settings["validation"]["ok"])


if __name__ == "__main__":
    unittest.main()
