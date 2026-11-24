"""Single-threaded SciPy/HiGHS exact UFL relaxation and primal/dual certificates.

For each customer retain its k cheapest sites. Treat the next-cheapest cost t
as available unconditionally. This can only underestimate the true service
cost, and is exact whenever an open site costs at most t. The continuous
variables allocate at most one unit to retained, open sites and earn savings
t-c from the optimistic charge. At k=m-1 this represents the full problem.
The master requires at least one open site. Refine only customers whose true
cost exceeds t; any MIP dual bound remains valid for the original problem.
Published reference values are consulted only after the search.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .ufl import load_cases


def build_master(fixed, costs, order, widths):
    n, m = costs.shape
    customer, rank = np.nonzero(np.arange(m)[None, :] < widths[:, None])
    sites = order[customer, rank]
    threshold = costs[np.arange(n), order[np.arange(n), widths]]
    u = len(sites)
    assign = m + np.arange(u, dtype=np.int32)
    link = 1 + n + np.arange(u, dtype=np.int32)
    rows = np.concatenate((np.zeros(m, dtype=int), customer + 1, link, link))
    cols = np.concatenate((np.arange(m), assign, assign, sites))
    data = np.concatenate((np.ones(m + 2*u), -np.ones(u)))
    matrix = coo_matrix((data, (rows, cols)), shape=(1+n+u, m+u)).tocsc()
    constraint = LinearConstraint(matrix, np.r_[1., np.full(n+u, -np.inf)],
                                  np.r_[np.inf, np.ones(n), np.zeros(u)])
    objective = np.r_[fixed, costs[customer, sites] - threshold[customer]]
    return objective, np.r_[np.ones(m, dtype=np.uint8), np.zeros(u, dtype=np.uint8)], constraint, threshold


def solve(case, seconds=10):
    if not 0 < seconds <= 10:
        raise ValueError("time budget must be greater than zero and at most 10 s")
    start = perf_counter()
    cost, fixed = case.service_costs, case.fixed_costs
    n, m = cost.shape
    order = np.argsort(cost, axis=1, kind="stable")
    widths = np.full(n, m - 1 if n <= 50 else min(8, m-1), dtype=int)
    # Cheap, independent feasible solutions. SciPy milp accepts no primal start.
    singleton = np.zeros(m, dtype=int)
    singleton[np.argmin(fixed + cost.sum(axis=0))] = 1
    incumbent = min((singleton, np.ones(m, dtype=int)), key=case.evaluate)
    best = case.evaluate(incumbent)
    source = "independent_singleton_or_all"
    lower = float(cost.min(axis=1).sum() + fixed.min())
    passes, native_primal, native_status = 0, False, None
    proof = False
    while perf_counter() - start < seconds:
        objective, integrality, constraints, threshold = build_master(fixed, cost, order, widths)
        remaining = seconds - (perf_counter() - start)
        if remaining <= 0:
            break
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=RuntimeWarning)
            result = milp(objective, integrality=integrality, bounds=Bounds(0, 1),
                          constraints=constraints, options={"threads": 1, "mip_rel_gap": 0.0,
                                                            "time_limit": remaining})
        passes += 1
        native_status = int(result.status)
        if native_status not in (0, 1):
            raise RuntimeError(f"{case.name}: HiGHS {result.message}")
        bound = getattr(result, "mip_dual_bound", None)
        if bound is not None and math.isfinite(bound):
            lower = max(lower, float(bound + threshold.sum()))
        if result.x is None:
            break
        flags = result.x[:m]
        bits = np.rint(flags).astype(int)
        if np.max(abs(flags-bits)) > 1e-5:
            raise ValueError("HiGHS returned a nonintegral primal facility vector")
        actual = case.evaluate(bits)
        native_primal = True
        if actual <= best:
            incumbent, best, source = bits, actual, "HiGHS"
        if lower > best + 1e-5:
            raise ValueError("dual bound exceeds independently rescored feasible cost")
        if best - lower <= 1e-5:
            proof = True
            break
        if native_status == 1:
            break
        actual_costs = cost[:, bits.astype(bool)].min(axis=1)
        affected = actual_costs > threshold
        if not affected.any():
            raise ValueError("exact optimal master disagrees with its dual bound")
        # Include all sites strictly cheaper than the new threshold next pass.
        widths[affected] = np.minimum(m-1, (cost[affected] < actual_costs[affected, None]).sum(axis=1))
    elapsed = perf_counter() - start
    recomputed = case.evaluate(incumbent)
    if best != recomputed:
        raise RuntimeError("primal certificate changed")
    return {"case": case.name, "facilities": m, "customers": n, "status": "optimal_proven" if proof else "time_limit",
            "feasible_source": source, "highs_primal_returned": int(native_primal), "last_highs_status": native_status,
            "objective": recomputed, "lower_bound": lower, "certificate_gap_abs": max(0., best-lower),
            "official_reference": case.reference, "gap_pct": 100*(best-case.reference)/case.reference,
            "hit_reference_0.001": int(abs(best-case.reference) <= .001),
            "open_sites_1based": json.dumps((np.flatnonzero(incumbent)+1).tolist(), separators=(",", ":")),
            "master_solves": passes, "seconds": round(elapsed, 6), "limit_s": seconds,
            "threads": 1, "scipy_version": scipy.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--out", type=Path, default=Path("results/ufl_solver.csv"))
    parser.add_argument("--case", nargs="+")
    parser.add_argument("--seconds", type=float, default=10.)
    args = parser.parse_args()
    cases = load_cases(args.cache)
    if args.case and not set(args.case) <= {case.name for case in cases}:
        parser.error("unknown case")
    records = [solve(case, args.seconds) for case in cases if not args.case or case.name in args.case]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"cases": len(records), "proved": sum(r["status"] == "optimal_proven" for r in records),
                      "official_hits": sum(r["hit_reference_0.001"] for r in records)}))


if __name__ == "__main__":
    main()
