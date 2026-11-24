import csv
from dataclasses import dataclass
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from selection_bench.run import run_case
from selection_bench.verify import verify


@dataclass
class Case:
    name = "example"
    reference = 1.0

    def initial(self, rng):
        return (int(rng.integers(200)),)

    def propose(self, best, rng):
        return ((best[0] + int(rng.integers(1, 100))) % 200,)

    def evaluate(self, solution):
        if len(solution) != 1 or not 0 <= solution[0] < 200:
            raise ValueError("Invalid state")
        return float(solution[0] + 1)

    def features(self, solution):
        return np.array(solution)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        suite = SimpleNamespace(load_cases=lambda cache: [Case()])
        self.enterContext(patch("selection_bench.verify.import_module", return_value=suite))
        self.enterContext(patch.dict(sys.modules, {"selection_bench.sr": None, "selection_bench.api": None}))
        picks = list(range(8)) + [3, 4, 0, 2]
        result = run_case("ufl", Case(), 11, "deepseek-operon", self.root, replay=picks)
        self.row = {k: result[k] for k in
                    ("suite", "case", "seed", "policy", "initial_best", "best", "reference", "reference_kind")}
        self.row.update(picks=json.dumps(picks), values=json.dumps([s["value"] for s in result["trace"]]),
                        solution=json.dumps(result["solution"]), gap_pct=100 * (result["best"] - 1),
                        initial_gap_pct=100 * (result["initial_best"] - 1), hit_reference=int(result["best"] == 1))
        self.csv = self.root / "per_run.csv"
        trace = self.root / "ufl/example/11/deepseek-operon.json"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps(result))

    def write(self, *rows):
        with self.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerows(rows)

    def test_replay_is_offline_and_checks_coverage(self):
        self.write(self.row)
        counts = verify(self.csv, self.root, partial=True, traces=self.root)
        self.assertEqual(counts, {"trajectories": 1, "step_objectives": 12, "by_suite": {"ufl": 1}})
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify(self.csv, self.root)
        self.write(self.row, self.row)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify(self.csv, self.root, partial=True)

    def test_rejects_tampered_fields(self):
        for field, corrupted in (("picks", "[true]"), ("values", "[1000]"), ("solution", "[-7]"),
                                 ("reference", "0"), ("reference_kind", "wrong"), ("gap_pct", "nan"),
                                 ("initial_gap_pct", "-1000"), ("hit_reference", "7"), ("best", "-1"),
                                 ("initial_best", "12345")):
            with self.subTest(field=field):
                self.write({**self.row, field: corrupted})
                with self.assertRaises(ValueError):
                    verify(self.csv, self.root, partial=True)


if __name__ == "__main__":
    unittest.main()
