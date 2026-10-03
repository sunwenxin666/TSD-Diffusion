"""CPU checks of real search utilities with synthetic objectives, not run results."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class SearchUtilityTests(unittest.TestCase):
    def test_dominance_and_fronts(self):
        from sd_autodiffusion_distill.distill.nas.searcher import dominates, fast_non_dominated_sort
        rows = [{"x": 0, "y": 2}, {"x": 2, "y": 0}, {"x": 3, "y": 3}]
        self.assertTrue(dominates(rows[0], rows[2], ["x", "y"]))
        self.assertFalse(dominates(rows[0], rows[1], ["x", "y"]))
        fronts, ranks = fast_non_dominated_sort(rows, ["x", "y"])
        self.assertEqual(fronts[0], [0, 1])
        self.assertEqual(fronts[1], [2])

if __name__ == "__main__":
    unittest.main()
