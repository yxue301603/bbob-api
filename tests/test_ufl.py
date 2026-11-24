"""Unit tests for UFL parsing, scoring, candidate moves, and cached public data."""
from pathlib import Path
import unittest

import numpy as np

from selection_bench import ufl


class UFLTest(unittest.TestCase):
    def setUp(self):
        # Non-unit customer demand and capacity placeholders must not rescale service costs.
        self.case = ufl._parse("test", b"3 2 capacity 7 25 6 capacity 12 100 2 9 15 2 7 1 3", 13.)

    def test_evaluate(self):
        self.assertEqual(self.case.evaluate((1,0,0)), 16.)
        self.assertEqual(self.case.evaluate((1,1,0)), 16.)
        self.assertEqual(self.case.evaluate((0,1,0)), 16.)
        for bad in ((0,0,0),(1,2,0),(1,0)):
            with self.assertRaises(ValueError): self.case.evaluate(bad)

    def test_candidates(self):
        rng = np.random.default_rng(2026)
        starts = {self.case.initial(rng) for _ in range(200)}
        self.assertEqual(len(starts),7)
        for start in starts:
            for _ in range(12):
                proposed = self.case.propose(start,rng)
                self.assertIn(proposed, starts)
                self.assertNotEqual(proposed, start)
                self.assertEqual(len(self.case.features(proposed)), len(self.case.feature_names))

    def test_features_do_not_encode_allocation_costs(self):
        other = ufl.UFLCase("scaled",0.,self.case.fixed_costs,self.case.service_costs*11)
        np.testing.assert_array_equal(self.case.features((1,0,1)),other.features((1,0,1)))
        self.assertNotEqual(self.case.evaluate((1,0,1)),other.evaluate((1,0,1)))

    def test_public_cache_when_present(self):
        cache = Path(__file__).resolve().parents[1]/".cache"
        if not (cache/"ufl/uncapopt.txt").is_file(): self.skipTest("UFL cache is optional for unit tests")
        cases = ufl.load_cases(cache)
        self.assertEqual(len(cases),15)
        self.assertEqual([x.service_costs.shape for x in cases], [(50,16)]*4+[(50,25)]*4+[(50,50)]*4+[(1000,100)]*3)
        if (cache/"ufl/ORLIB.tgz").is_file():
            self.assertLessEqual(max(map(abs,ufl.verify_public_solutions(cache).values())),.001)
