import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ReleaseHelperTests(unittest.TestCase):
    def test_export_selects_normalized_ideal_point_and_descending_schedule(self):
        path = ROOT / "scripts/export_timesteps.py"
        self.assertTrue(path.is_file(), "Missing Pareto-to-schedule export helper")
        spec = importlib.util.spec_from_file_location("release_export", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Synthetic unit-test objectives, not experimental measurements.
        rows = [([0, 2, 4, 6], (0.0, 10.0, 10.0)),
                ([1, 3, 5, 7], (4.0, 4.0, 4.0)),
                ([2, 4, 6, 8], (10.0, 0.0, 0.0))]
        self.assertEqual(module.select_schedule(rows), [7, 5, 3, 1])
        self.assertEqual(module.parse_line('(1, 3, 5, 7)\tfid=4\tlat=4\tstd=4'), rows[1])
        with self.assertRaises(ValueError):
            module.select_schedule([])
        with self.assertRaises(ValueError):
            module.select_schedule([([1, 1, 2, 3], (1, 1, 1))])

if __name__ == "__main__":
    unittest.main()
