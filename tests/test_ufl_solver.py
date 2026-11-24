import itertools
import unittest

import numpy as np
from scipy.optimize import Bounds, milp

from selection_bench.ufl import UFLCase
from selection_bench.ufl_solver import build_master, solve


class UFLMILPTests(unittest.TestCase):
    def test_certificate_matches_enumeration_without_reference(self):
        case = UFLCase("tiny", 999, [9, 2, 8, 1], [[1, 4, 8, 4], [3, 9, 0, 6], [6, 5, 0, 8]])
        best = min(case.evaluate(bits) for bits in itertools.product((0, 1), repeat=4) if any(bits))
        record = solve(case)
        self.assertEqual(record["status"], "optimal_proven")
        self.assertAlmostEqual(record["lower_bound"], best)
        self.assertAlmostEqual(record["objective"], best)
        self.assertEqual(record["hit_reference_0.001"], 0)

    def test_truncated_master_is_optimistic_for_all_facility_sets(self):
        costs = np.array([[13, 3, 1, 8], [1, 4, 6, 7], [7, 7, 3, 3.]])
        case = UFLCase("tiny", 1, [2, 5, 4, 6], costs)
        order = costs.argsort(axis=1)
        # Evaluate each relaxed LP at a fixed binary facility pattern and
        # verify both the optimistic threshold and the fully exact case.
        for widths in (np.array([1, 2, 1]), np.full(3, 3)):
            obj, _, constraint, t = build_master(case.fixed_costs, costs, order, widths)
            for bits in itertools.product((0, 1), repeat=4):
                if not any(bits):
                    continue
                u = len(obj)-4
                lp = milp(obj, bounds=Bounds(np.r_[bits, np.zeros(u)], np.r_[bits, np.ones(u)]),
                          constraints=constraint)
                self.assertTrue(lp.success)
                optimistic = np.dot(bits, case.fixed_costs) + np.minimum(costs[:, np.array(bits, dtype=bool)].min(axis=1), t).sum()
                self.assertAlmostEqual(lp.fun+t.sum(), optimistic)
                self.assertLessEqual(optimistic, case.evaluate(bits))
                if min(widths) == 3:
                    self.assertAlmostEqual(optimistic, case.evaluate(bits))


if __name__ == "__main__":
    unittest.main()
