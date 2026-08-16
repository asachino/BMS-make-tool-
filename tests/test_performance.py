import time
import unittest

from bms_reuse.clustering.reuse_plan import build_reuse_plan
from bms_reuse.extraction.hit_extractor import Hit
from bms_reuse.similarity.score import SimilarityReport


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


if __name__ == "__main__":
    unittest.main()
