import time
import unittest

from bms_reuse.clustering.reuse_plan import build_reuse_plan
from bms_reuse.extraction.hit_extractor import Hit
from bms_reuse.similarity.score import SimilarityReport


def _report(reference, candidate, classification: str) -> SimilarityReport:
    if classification == "SAME":
        return SimilarityReport(reference.id, candidate.id, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0)
    if classification == "GAIN_VARIANT":
        return SimilarityReport(reference.id, candidate.id, 0.9, 1.0, 6.0, 1.0, 1.0, 1.0, 1.0, 0)
    return SimilarityReport(reference.id, candidate.id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def _feature_hit(index: int, centroid: float) -> Hit:
    return Hit(
        index,
        index,
        float(index),
        [float(index)],
        index,
        index + 1,
        features={
            "centroid_hz": centroid,
            "rolloff_hz": centroid,
            "zcr": centroid / 1000.0,
            "attack_ms": centroid / 10.0,
            "tail_energy": 0.1,
            "rms_db": -20.0,
        },
    )


class PerformanceSmokeTest(unittest.TestCase):
    def test_526_hit_clustering_reports_progress(self):
        hits = [Hit(index, index, float(index), [0.0], index, index + 1) for index in range(526)]
        updates = []
        detail_count = [0]

        def compare(reference, candidate):
            return SimilarityReport(reference.id, candidate.id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        started = time.perf_counter()
        plan, reports = build_reuse_plan(
            hits,
            compare,
            progress=lambda done, total: updates.append((done, total)),
            progress_detail=lambda _current, _total, _compared: detail_count.__setitem__(0, detail_count[0] + 1),
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(len(plan.clusters), 526)
        self.assertEqual(len(reports), 526 * 525 // 2)
        self.assertEqual(updates[-1], (526, 526))
        self.assertEqual(detail_count[0], len(reports))
        print(f"526-hit clustering smoke: {elapsed:.3f}s, comparisons={len(reports)}")

    def test_normal_exact_hit_reuses_assignment_and_reports(self):
        hits = [Hit(index, index, float(index), [1.0, 2.0], index, index + 2) for index in range(3)]
        calls = [0]
        cache_hits = [0]

        def compare(reference, candidate):
            calls[0] += 1
            return _report(reference, candidate, "SAME")

        plan, reports = build_reuse_plan(
            hits,
            compare,
            reuse_key=lambda hit: tuple(hit.samples),
            reuse_equal=lambda left, right: left.samples == right.samples,
            cache_hit=lambda: cache_hits.__setitem__(0, cache_hits[0] + 1),
        )

        self.assertEqual(calls[0], 1)
        self.assertEqual(cache_hits[0], 2)
        self.assertEqual(len(reports), 2)
        self.assertEqual([report.candidate_id for report in reports], [1, 2])
        self.assertTrue(all(report.classification == "SAME" for report in reports))
        self.assertEqual(plan.to_dict()["events"], [
            {"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
            {"hit": 1, "time": 1.0, "sample_id": "sample_001", "gain_db": 0.0},
            {"hit": 2, "time": 2.0, "sample_id": "sample_001", "gain_db": 0.0},
        ])

    def test_normal_mode_matches_explicit_baseline_order(self):
        hits = [_feature_hit(0, 0.0), _feature_hit(1, 100.0)]
        hits.extend(
            [
                Hit(2, 2, 2.0, [1.0], 2, 3, features=dict(hits[1].features)),
                Hit(3, 3, 3.0, [0.0], 3, 4, features=dict(hits[0].features)),
            ]
        )

        def compare(reference, candidate):
            return _report(reference, candidate, "SAME" if reference.samples == candidate.samples else "DIFFERENT")

        baseline_plan, baseline_reports = build_reuse_plan(hits, compare)
        cache_hits = [0]
        cached_plan, cached_reports = build_reuse_plan(
            hits,
            compare,
            fast_compare=False,
            reuse_key=lambda hit: tuple(hit.samples),
            reuse_equal=lambda left, right: left.samples == right.samples,
            cache_hit=lambda: cache_hits.__setitem__(0, cache_hits[0] + 1),
        )
        self.assertEqual(baseline_plan.to_dict(), cached_plan.to_dict())
        self.assertEqual([report.to_dict() for report in baseline_reports], [report.to_dict() for report in cached_reports])
        self.assertGreater(cache_hits[0], 0)

    def test_fast_mode_reduces_comparisons_and_keeps_fallback(self):
        hits = [_feature_hit(index, float(index % 3) * 100.0) for index in range(30)]

        def compare(reference, candidate):
            # Similarity uses FFT/alignment in production; retain a small
            # deterministic workload so this fixture measures wall time as
            # well as the number of representative candidates.
            sum(index * index for index in range(5000))
            return _report(reference, candidate, "SAME" if reference.id % 3 == candidate.id % 3 else "DIFFERENT")

        normal_started = time.perf_counter()
        normal_plan, normal_reports = build_reuse_plan(hits, compare)
        normal_elapsed = time.perf_counter() - normal_started
        fast_started = time.perf_counter()
        fast_plan, fast_reports = build_reuse_plan(hits, compare, fast_compare=True)
        fast_elapsed = time.perf_counter() - fast_started

        self.assertEqual(len(normal_plan.clusters), 3)
        self.assertEqual(len(fast_plan.clusters), 3)
        self.assertLess(len(fast_reports), len(normal_reports))
        self.assertLess(fast_elapsed, normal_elapsed)
        print(
            f"feature-order benchmark: normal={normal_elapsed:.4f}s/{len(normal_reports)} comparisons, "
            f"fast={fast_elapsed:.4f}s/{len(fast_reports)} comparisons"
        )

    def test_fast_mode_adversarial_two_clusters_can_change_first_match(self):
        hits = [_feature_hit(0, 0.0), _feature_hit(1, 100.0), _feature_hit(2, 100.0), _feature_hit(3, 50.0)]

        def compare(reference, candidate):
            pair = (reference.id, candidate.id)
            if pair == (0, 1):
                return _report(reference, candidate, "DIFFERENT")
            if pair == (0, 2):
                return _report(reference, candidate, "GAIN_VARIANT")
            if pair == (1, 2):
                return _report(reference, candidate, "SAME")
            return _report(reference, candidate, "DIFFERENT")

        normal_plan, normal_reports = build_reuse_plan(hits, compare)
        fast_plan, fast_reports = build_reuse_plan(hits, compare, fast_compare=True)

        normal_hit2 = next(report for report in normal_reports if report.candidate_id == 2)
        fast_hit2 = next(report for report in fast_reports if report.candidate_id == 2)
        self.assertEqual(normal_hit2.classification, "GAIN_VARIANT")
        self.assertEqual(fast_hit2.classification, "SAME")
        self.assertNotEqual(normal_plan.to_dict()["events"][2]["sample_id"], fast_plan.to_dict()["events"][2]["sample_id"])
        self.assertEqual(len(fast_plan.clusters), 3)
        self.assertEqual(len(fast_reports), 4)


if __name__ == "__main__":
    unittest.main()
