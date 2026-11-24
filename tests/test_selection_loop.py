import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path

from selection_bench.run import INITIAL, OPTIONS, ROUNDS, bind_model, run_case


class CountingCase:
    name, reference = "smoke", 1.0
    feature_names = ("integer",)

    def __init__(self):
        self.evaluations = []

    def initial(self, rng):
        return (int(rng.integers(0, 1000)),)

    def propose(self, best, rng):
        return (int(best[0] + rng.integers(-32, 33)),)

    def features(self, solution):
        return solution

    def evaluate(self, solution):
        self.evaluations.append(solution)
        return float(1 + abs(solution[0]))


class TestSelectionLoop(unittest.TestCase):
    def test_queries_resume_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            calls = []

            def select(job, step, prompt):
                self.assertEqual(prompt.count("Options ID:"), 1)
                if len(calls) == 4 and step == 4 and not failed[0]:
                    failed[0] = True
                    raise RuntimeError("interrupted")
                calls.append(step)
                return step % OPTIONS

            failed = [False]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_case("ufl", CountingCase(), 11, "deepseek", out, select)
            problem = CountingCase()
            run = run_case("ufl", problem, 11, "deepseek", out, select)
            self.assertEqual(calls, list(range(ROUNDS)))
            self.assertEqual(len(problem.evaluations), INITIAL + ROUNDS)
            self.assertEqual(len(set(problem.evaluations)), INITIAL + ROUNDS)
            replay = run_case("ufl", CountingCase(), 11, "deepseek", out,
                              replay=[r["pick"] for r in run["trace"]])
            self.assertEqual(replay["best"], run["best"])
            self.assertEqual(replay["solution"], run["solution"])
            self.assertEqual([r["value"] for r in replay["trace"]], [r["value"] for r in run["trace"]])
            other = CountingCase()
            run_case("ufl", other, 11, "random", out)
            self.assertEqual(problem.evaluations[:INITIAL], other.evaluations[:INITIAL])
            self.assertEqual(len(other.evaluations), INITIAL + ROUNDS)

    def test_resume_with_other_model_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            bind_model(out, "model-a", "https://api.example.org")
            bind_model(out, "model-a", "https://api.example.org/")
            with self.assertRaisesRegex(ValueError, "Model settings changed"):
                bind_model(out, "model-b", "https://api.example.org")

    def test_concurrent_launches_share_one_set_of_picks(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def choose(job, step, prompt):
                calls.append(step)
                return step % OPTIONS
            with ThreadPoolExecutor(max_workers=2) as workers:
                jobs = [workers.submit(run_case, "ufl", CountingCase(), 11, "deepseek",
                                       Path(folder), choose) for _ in range(2)]
                for job in jobs:
                    job.result()
            self.assertEqual(calls, list(range(ROUNDS)))
