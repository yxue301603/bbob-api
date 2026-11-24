import itertools
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from selection_bench import pmedian


GRAPH = b"4 5 2\n1 2 1\n2 3 1\n3 4 1\n1 4 2\n2 1 9\n"
CACHE = Path(__file__).resolve().parents[1] / ".cache"


class PMedianTests(TestCase):
    def test_last_duplicate_wins_and_all_scores(self):
        case = pmedian._case("fixture", GRAPH, 2)
        # Independently calculated shortest-path costs after replacing 1--2 by 9.
        expected = np.array([[0,4,3,2],[4,0,1,2],[3,1,0,1],[2,2,1,0]])
        np.testing.assert_array_equal(case.distances, expected)
        for centers in itertools.combinations(range(4), 2):
            total = sum(min(expected[i, c] for c in centers) for i in range(4))
            self.assertEqual(case.evaluate(centers), total)

    def test_zero_cost_and_illegal_solutions(self):
        case = pmedian._case("zero", b"3 2 1\n1 2 0\n2 3 4\n", 4)
        self.assertEqual(case.evaluate([0]), 4.)
        self.assertTrue(np.isfinite(case.features([2])).all())
        for illegal in ([], [1.2], [-1], [3], [True], [1,1]):
            with self.subTest(illegal=illegal), self.assertRaises(ValueError):
                case.evaluate(illegal)

    def test_neighbor_and_features_never_score_a_candidate(self):
        case = pmedian._case("fixture", GRAPH, 2)
        rng = np.random.default_rng(7)
        with patch.object(case, "evaluate", side_effect=AssertionError("unbudgeted objective call")):
            best = case.initial(rng)
            for _ in range(20):
                nxt = case.propose(best, rng)
                self.assertEqual(len(set(nxt) ^ set(best)), 2)
                values = case.features(nxt)
                np.testing.assert_array_equal(values, case.features(tuple(reversed(nxt))))
                self.assertEqual(values.shape, (len(case.feature_names),))
                self.assertTrue(np.isfinite(values).all())
                best = nxt
        # Cost-to-demand entries can change without changing the supplied features.
        before = case.features([0,2])
        case.distances[np.ix_([0,2],[1,3])] = 100
        np.testing.assert_array_equal(before, case.features([0,2]))

    @skipUnless((CACHE/"pmedian/pmedopt.txt").is_file(), "run data preparation for the full-suite check")
    def test_official_suite_and_independent_floyd(self):
        with patch.object(pmedian, "urlopen", side_effect=AssertionError("unexpected network")):
            pmedian.prepare(CACHE)
        cases = pmedian.load_cases(CACHE)
        self.assertEqual([case.name for case in cases], list(pmedian.NAMES))
        self.assertEqual((cases[0].reference, cases[-1].reference), (5819., 5128.))
        first = cases[0]
        edges = np.loadtxt(CACHE/"pmedian/pmed1.txt", dtype=int)
        n = edges[0,0]
        independent = np.full((n,n), np.inf)
        np.fill_diagonal(independent, 0)
        for a,b,cost in edges[1:]:
            independent[a-1,b-1] = independent[b-1,a-1] = cost
        for k in range(n):
            independent = np.minimum(independent, independent[:,k,None] + independent[None,k,:])
        np.testing.assert_array_equal(first.distances, independent)
        for case in cases:
            init = case.initial(np.random.default_rng(11))
            self.assertGreaterEqual(case.evaluate(init), case.reference)
