"""Compare two COCO datasets on common problems at a fixed query budget."""
import argparse
from pathlib import Path

import cocopp
import numpy as np


def by_problem(path):
    datasets = {}
    for data in cocopp.load(str(path)):
        key = data.funcId, data.dim
        if key in datasets or data.suite_name != "bbob":
            raise ValueError("Each input must contain a single algorithm on the noiseless bbob suite")
        datasets[key] = data
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--target", type=float, default=1.0, help="Target objective gap")
    args = parser.parse_args()
    if args.budget < 1 or not 0 < args.target < float("inf"):
        parser.error("budget and target must be positive and finite")
    left, right = by_problem(args.experiment), by_problem(args.reference)
    rates = []
    print("function,dimension,instance,experiment_hits,experiment_runs,reference_hits,reference_runs")
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        for instance in sorted(set(a.instancenumbers) & set(b.instancenumbers)):
            counts = []
            for data in (a, b):
                evaluations = data.detEvals([args.target], instance=instance)[0]
                counts.append((int(np.sum(evaluations <= args.budget)), len(evaluations)))
            (a_hits, a_runs), (b_hits, b_runs) = counts
            print(*key, instance, a_hits, a_runs, b_hits, b_runs, sep=",")
            rates.append((a_hits / a_runs, b_hits / b_runs))
    if not rates:
        parser.error("No shared function, dimension and instance")
    a_rate, b_rate = np.mean(rates, axis=0)
    print(f"{len(rates)} shared problems, target gap <= {args.target:g} within {args.budget} "
          f"evaluations: experiment {a_rate:.1%}, reference {b_rate:.1%} (equal problem weights).")


if __name__ == "__main__":
    main()
