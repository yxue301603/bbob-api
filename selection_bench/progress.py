"""Best-so-far gaps and paired instance-level bootstrap sensitivity intervals.

Every case has equal weight. Average the three paired seeds within each case
before resampling cases with replacement. Percentile intervals describe
sensitivity to the mix of these benchmark cases; they do not correct for
multiple comparisons or establish generalization to new problem families.
Positive differences favor the named method (baseline gap minus method gap).
"""
import argparse
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np

from .run import INITIAL, POLICIES, ROUNDS, SEEDS, SUITES
from .summarize import CASES


CHECKPOINTS = (0, 4, 8, 12)
PAIRS = (("deepseek", "random"),
         ("deepseek-operon", "greedy-operon"), ("deepseek-gplearn", "greedy-gplearn"),
         ("deepseek-operon", "deepseek"), ("deepseek-gplearn", "deepseek"))
BOOTSTRAP_SEED = 20261124
RESAMPLES = 10_000


def read_gaps(path, suites):
    expected = {(suite, *key) for suite in suites for key in product(CASES[suite], SEEDS, POLICIES)}
    gaps, starts, references = {}, {}, {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            suite = row["suite"]
            if suite not in SUITES:
                raise ValueError(f"Unknown suite {suite}")
            if suite not in suites:
                continue
            key = suite, row["case"], int(row["seed"]), row["policy"]
            if key not in expected or key in gaps:
                raise ValueError(f"Unexpected or duplicate row {key}")
            ref, initial = float(row["reference"]), float(row["initial_best"])
            if not np.isfinite([ref, initial]).all() or ref <= 0:
                raise ValueError(f"Invalid objective scale on {key}")
            if references.setdefault(key[:2], ref) != ref or starts.setdefault(key[:3], initial) != initial:
                raise ValueError(f"Paired starts or reference differ on {key}")
            steps = np.asarray(json.loads(row["values"]), dtype=float)
            if steps.shape != (ROUNDS,) or not np.isfinite(steps).all():
                raise ValueError(f"Invalid step values for {key}")
            best = np.minimum.accumulate(np.r_[initial, steps])[list(CHECKPOINTS)]
            if not np.isclose(best[-1], float(row["best"]), rtol=0, atol=1e-7):
                raise ValueError(f"Final score disagrees with trajectory on {key}")
            gaps[key] = 100 * (best - ref) / ref
    if gaps.keys() != expected:
        raise ValueError(f"Incomplete grid, missing {len(expected - gaps.keys())} trajectories")
    return gaps


def tables(gaps, suites):
    progress, robustness = [], []
    for suite in suites:
        cases = CASES[suite]
        # Each array has case, seed, checkpoint axes, in a fixed order.
        matrix = {p: np.asarray([[gaps[suite, c, s, p] for s in SEEDS] for c in cases]) for p in POLICIES}
        averages = {p: values.mean(axis=1) for p, values in matrix.items()}
        for p in POLICIES:
            per_case = averages[p]
            per_seed = matrix[p].mean(axis=0)
            for i, step in enumerate(CHECKPOINTS):
                progress.append(dict(suite=suite, policy=p, step=step, queries=INITIAL+step,
                                     instances=len(cases), seeds=len(SEEDS), mean_gap_pct=per_case[:, i].mean(),
                                     instance_mean_gap_sample_sd_pct=per_case[:, i].std(ddof=1),
                                     seed_mean_gap_sample_sd_pct=per_seed[:, i].std(ddof=1)))
        # Use identical case draws across checkpoints and contrasts in a suite.
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, SUITES.index(suite)]))
        draws = rng.integers(len(cases), size=(RESAMPLES, len(cases)))
        for method, baseline in PAIRS:
            differences = averages[baseline] - averages[method]
            for i, step in enumerate(CHECKPOINTS):
                difference = differences[:, i]
                low, high = np.quantile(difference[draws].mean(axis=1), [.025, .975])
                robustness.append(dict(suite=suite, method=method, baseline=baseline, step=step,
                                       queries=INITIAL+step, instances=len(cases), seeds=len(SEEDS),
                                       mean_gap_reduction_pp=difference.mean(), ci95_low_pp=low, ci95_high_pp=high,
                                       bootstrap_unit="instance", bootstrap_resamples=RESAMPLES,
                                       bootstrap_seed=BOOTSTRAP_SEED))
    return {"progress": progress, "robustness": robustness}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/per_run.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--suite", nargs="+", choices=SUITES, default=SUITES)
    args = parser.parse_args()
    if len(set(args.suite)) != len(args.suite):
        parser.error("Each suite should appear only once")
    results = tables(read_gaps(args.csv, args.suite), args.suite)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in results.items():
        path = args.out / (name + ".csv")
        pending = path.with_suffix(".csv.tmp")
        with pending.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(path)
    print(json.dumps({name: len(rows) for name, rows in results.items()}))


if __name__ == "__main__":
    main()
