"""Run the original FasterPAM implementation on all OR-Library p-median cases.

Optional dependency: python -m pip install kmedoids==0.5.5
Algorithm: https://doi.org/10.1016/j.is.2021.101804
Solver: https://python-kmedoids.readthedocs.io/en/stable/#fasterpam

Each run starts from one uniform sample of p distinct vertices. The package
uses the seed for initialization and the order in which potential swaps are
visited. Eager strictly improving swaps are accepted; equal costs do not
force a unique center set. The algorithm sees the full distance matrix and
stops at convergence or 100 passes. Its passes and accepted swaps are not
black-box objective queries. The independent checks below occur after search.
"""
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from .pmedian import load_cases, prepare
from .run import SEEDS


PACKAGE_VERSION = "0.5.5"
MAX_ITER = 100


def best_swap_delta(distances, centers):
    """Independent certificate: minimum objective change over all one-center swaps."""
    # For a client whose closest center is removed, the incumbent's second-best
    # center takes over, unless the incoming center is at least as close.
    cost = distances[centers]
    owners = np.argmin(cost, axis=0)
    clients = np.arange(cost.shape[1])
    nearest = cost[owners, clients]
    second = np.partition(cost, kth=1, axis=0)[1]
    incumbent = nearest.sum(dtype=np.int64)
    unused = np.ones(len(clients), dtype=bool)
    unused[centers] = False
    smallest = np.inf
    for new_cost in distances[unused]:
        with_new = np.minimum(nearest, new_cost)
        extra = np.minimum(second, new_cost) - with_new
        corrections = np.bincount(owners, weights=extra, minlength=len(centers))
        smallest = min(smallest, with_new.sum(dtype=np.int64) - incumbent + corrections.min())
    return int(smallest)


def measure(case, seed):
    from kmedoids import fasterpam

    # Version 0.5.5 uses RandomState(seed).choice(n,p,replace=False) internally.
    start = np.random.RandomState(seed).choice(len(case.distances), case.p, replace=False)
    wall, cpu = perf_counter(), process_time()
    fit = fasterpam(case.distances, case.p, init="random", random_state=seed,
                    max_iter=MAX_ITER, n_cpu=1)
    solver_wall, solver_cpu = perf_counter() - wall, process_time() - cpu
    centers = np.asarray(fit.medoids)
    score = case.evaluate(centers)
    if score != fit.loss or score < case.reference:
        raise ValueError(f"Final score mismatch for {case.name}, seed {seed}")
    delta = best_swap_delta(case.distances, centers)
    if int(fit.n_iter) < MAX_ITER and delta < 0:
        raise ValueError(f"Improving single-center swap remains on {case.name}, seed {seed}")
    return dict(suite="pmedian", case=case.name, seed=seed, method="FasterPAM", kmedoids_version=PACKAGE_VERSION,
                n=len(case.distances), p=case.p, reference=case.reference, reference_kind="proven_optimum",
                initial_centers=json.dumps(start.tolist(), separators=(",", ":")), initial_loss=case.evaluate(start),
                centers=json.dumps(sorted(map(int, centers)), separators=(",", ":")), solver_loss=fit.loss,
                verified_loss=score, gap_pct=100 * (score - case.reference) / case.reference,
                hit_optimum=int(score == case.reference), n_threads=1, max_iter=MAX_ITER,
                n_iter=fit.n_iter, n_swap=fit.n_swap, best_single_swap_delta=delta,
                solver_wall_s=solver_wall, solver_cpu_s=solver_cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--out", type=Path, default=Path("results/pam.csv"))
    args = parser.parse_args()
    if version("kmedoids") != PACKAGE_VERSION:
        parser.error(f"This reference run requires kmedoids=={PACKAGE_VERSION}")
    prepare(args.cache)
    cases = load_cases(args.cache)
    rows = [measure(case, seed) for case in cases for seed in SEEDS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.out)
    print(json.dumps(dict(trajectories=len(rows), hits=sum(r["hit_optimum"] for r in rows),
                          mean_gap_pct=np.mean([r["gap_pct"] for r in rows]),
                          solver_wall_s=sum(r["solver_wall_s"] for r in rows))))


if __name__ == "__main__":
    main()
