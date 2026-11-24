"""Run the external NEH heuristic on all 120 Taillard instances using Brunner v13 references.

Install the optional dependency with `python -m pip install permutation-flowshop==1.0.3`.
Method: https://doi.org/10.1016/0305-0483(83)90088-9 .
The package's NEHT function evaluates all insertion positions with Taillard acceleration.
"""
import argparse
import csv
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from statistics import mean

import numpy as np

from .flowshop import load_cases


def solve(case):
    from pfsp.NEHT import NEHT

    # NEHT uses reversed np.argsort without a stable tie order. Exchanging only the
    # labels of equal-work jobs makes the original job ID the portable tie breaker.
    totals = case.processing.sum(axis=0)
    priority = np.lexsort((np.arange(case.jobs), -totals))
    package_order = np.argsort(totals.astype(float))[::-1]
    job_ids = np.empty(case.jobs, dtype=int)
    job_ids[package_order] = priority
    indices, score = NEHT(case.jobs, case.machines, case.processing[:, job_ids])
    sequence = job_ids[indices].tolist()
    verified = case.evaluate(sequence)
    if verified != score:
        raise ValueError(f"Scorer disagrees with NEHT on {case.name}")
    # NEHT tests both orders of the first two jobs, then all k insertion slots
    # for job k (one-based). An insertion tie always uses the earliest slot.
    positions = case.jobs * (case.jobs + 1) // 2 - 1
    return dict(case=case.name, jobs=case.jobs, machines=case.machines, makespan=int(verified),
                reference=int(case.reference), reference_kind=case.reference_kind,
                gap_pct=100 * (verified - case.reference) / case.reference, insertion_positions=positions,
                solution=json.dumps(sequence, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--out", type=Path, default=Path("results/neh.csv"))
    args = parser.parse_args()
    try:
        installed = version("permutation-flowshop")
    except PackageNotFoundError:
        installed = None
    if installed != "1.0.3":
        parser.error("Install permutation-flowshop==1.0.3")
    rows = [solve(case) for case in load_cases(args.cache)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(args.out)
    print(json.dumps(dict(cases=len(rows), mean_gap_pct=mean(r["gap_pct"] for r in rows),
                          hits=sum(r["makespan"] == r["reference"] for r in rows),
                          insertion_positions=sum(r["insertion_positions"] for r in rows))))


if __name__ == "__main__":
    main()
