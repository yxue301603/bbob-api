import unittest
import numpy as np

from selection_bench.flowshop import FlowShopCase, processing_times


class TestFlowShop(unittest.TestCase):
    def test_original_generator(self):
        matrix = processing_times(20, 5, 873654221)
        np.testing.assert_array_equal(matrix[:, 0], [54, 79, 16, 66, 58])
        np.testing.assert_array_equal(matrix[:, -1], [94, 77, 40, 31, 28])

    def test_evaluator_and_features(self):
        matrix = np.array([[1, 5, 2, 4], [6, 2, 3, 1], [2, 4, 9, 7]])
        case = FlowShopCase("example", matrix, 0, 0, False)
        rng = np.random.default_rng(9)
        order = case.initial(rng)
        for _ in range(20):
            previous = order.copy()
            order = case.propose(order, rng)
            self.assertFalse(np.array_equal(previous, order))
            finish = [0]*(case.machines+1)
            for j in order:
                for k in range(1, case.machines+1):
                    finish[k] = max(finish[k-1], finish[k]) + matrix[k-1, j]
            self.assertEqual(finish[-1], case.evaluate(order))
            values = case.features(order)
            self.assertEqual(len(case.feature_names), len(values))
            self.assertTrue(np.isfinite(values).all())
        with self.assertRaises(ValueError):
            case.evaluate([0, 0, 2, 3])
