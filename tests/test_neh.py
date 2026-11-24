import importlib.util
import json
import unittest

import numpy as np

from selection_bench.flowshop import FlowShopCase
from selection_bench.neh import solve


def scalar_makespan(times, order):
    finish = [0] * (len(times) + 1)
    for job in order:
        for machine, duration in enumerate(times[:, job], start=1):
            finish[machine] = max(finish[machine - 1], finish[machine]) + int(duration)
    return finish[-1]


@unittest.skipUnless(importlib.util.find_spec("pfsp"), "Install optional permutation-flowshop==1.0.3")
class NehTests(unittest.TestCase):
    def test_independent_enumeration_and_ties(self):
        rng = np.random.default_rng(11)
        for times in [np.ones((3, 8), dtype=int)] + [rng.integers(1, 9, size=(4, 12)) for _ in range(10)]:
            case = FlowShopCase("tiny", times, 1, 1, False)
            result = solve(case)
            order = sorted(range(case.jobs), key=lambda j: (-times[:, j].sum(), j))
            best = order[:1]
            checked = 0
            for j in order[1:]:
                options = [best[:i] + [j] + best[i:] for i in range(len(best) + 1)]
                checked += len(options)
                best = min(options, key=lambda opt: scalar_makespan(times, opt))
            self.assertEqual(json.loads(result["solution"]), best)
            self.assertEqual(result["makespan"], scalar_makespan(times, best))
            self.assertEqual(result["insertion_positions"], checked)


if __name__ == "__main__":
    unittest.main()
