import unittest
import numpy as np
from selection_bench.analyse_confirmation import randomization_p, scalar_makespan
from selection_bench.run import prompt_for


class TestConfirm(unittest.TestCase):
    def test_full_history_adds_observations_only(self):
        X=np.arange(8.,dtype=float).reshape(-1,1)
        y=np.arange(8.,dtype=float)
        Q=np.array([[2.]]*8)
        short=prompt_for("flowshop",["name"],X,y,Q,None,None)
        full=prompt_for("flowshop",["name"],X,y,Q,None,None,True)
        self.assertNotIn("3.00|3.00",short)
        self.assertIn("3.00|3.00",full)
        self.assertEqual(short.split("Options ID:")[1],full.split("Options ID:")[1])

    def test_scalar_waiting_and_null(self):
        self.assertEqual(scalar_makespan(np.array([[3,2],[1,5]]),[0,1]),10)
        self.assertEqual(scalar_makespan(np.array([[3,2],[1,5]]),[1,0]),8)
        self.assertEqual(randomization_p(np.zeros(6)),1.)
        d=np.array([1.,-1.,2.,-2.])
        self.assertEqual(randomization_p(d),1.)
