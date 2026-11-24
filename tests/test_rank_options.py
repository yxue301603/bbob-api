import unittest

from selection_bench.rank_options import rank_pick


class RankOptionsTests(unittest.TestCase):
    def test_midrank_and_tied_minimum(self):
        scores = [2, 2, 4, 7, 7, 7, 9, 10]
        self.assertEqual(rank_pick(scores, 0, 3), (1.5, 1, 1, 0))
        self.assertEqual(rank_pick(scores, 3, 7), (5, 0, 1, 1))
        self.assertEqual(rank_pick(scores, 7, 2), (8, 0, 0, 0))

    def test_full_tie_is_a_best_pick_but_not_a_strict_improvement(self):
        self.assertEqual(rank_pick([3]*8, 5, 3), (4.5, 1, 0, 0))
        self.assertEqual(rank_pick([3]*8, 5, 4), (4.5, 1, 1, 0))


if __name__ == "__main__":
    unittest.main()
