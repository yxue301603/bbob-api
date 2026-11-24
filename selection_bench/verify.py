"""Rebuild the published choices with local benchmark data. No model or fitter is loaded."""

import argparse
import csv
from collections import Counter
from importlib import import_module
import json
from math import isclose, isfinite
from pathlib import Path

from .run import OPTIONS, POLICIES, ROUNDS, SEEDS, SUITES, run_case


def require_equal(label, expected, observed):
    if expected != observed:
        raise ValueError(f"{label} differs (published {expected!r}, rebuilt {observed!r})")


def number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean objective")
    result = float(value)
    if not isfinite(result):
        raise ValueError("Nonfinite objective")
    return result


def vector(text, *, integers):
    values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a nonempty JSON array")
    if integers and any(type(x) is not int for x in values):
        raise ValueError("Expected integer elements")
    return values if integers else [number(x) for x in values]


def verify_row(row, case, traces=None):
    suite, seed, policy = row["suite"], int(row["seed"]), row["policy"]
    picks = vector(row["picks"], integers=True)
    if len(picks) != ROUNDS or any(not 0 <= p < OPTIONS for p in picks):
        raise ValueError("Invalid number of picks or candidate index")
    result = run_case(suite, case, seed, policy, Path("."), replay=picks)
    actual_steps = [step["value"] for step in result["trace"]]
    for field in ("initial_best", "best", "reference"):
        require_equal(field, number(row[field]), result[field])
    require_equal("reference_kind", row["reference_kind"], result["reference_kind"])
    require_equal("step values", vector(row["values"], integers=False), actual_steps)
    solution = vector(row["solution"], integers=True)
    require_equal("final solution", solution, list(result["solution"]))
    require_equal("scored final solution", result["best"], case.evaluate(solution))
    reference = result["reference"]
    tolerance = 0.001 if suite == "ufl" else 1e-7
    for field, value in (("initial_gap_pct", result["initial_best"]), ("gap_pct", result["best"])):
        gap = 100 * (value - reference) / reference
        if not isclose(number(row[field]), gap, rel_tol=1e-13, abs_tol=1e-13):
            raise ValueError(f"{field} differs from recomputed {gap}")
    require_equal("hit_reference", int(row["hit_reference"]), int(result["best"] <= reference + tolerance))
    if result["reference_kind"] == "proven_optimum" and result["best"] < reference - tolerance:
        raise ValueError("Best solution improves on a proven optimum outside the source tolerance")
    if traces is not None:
        path = traces / suite / case.name / str(seed) / (policy + ".json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        for field in ("suite", "case", "seed", "policy", "initial_best", "best", "reference", "reference_kind"):
            require_equal(f"trace {field}", raw[field], result[field])
        require_equal("trace solution", raw["solution"], solution)
        require_equal("trace picks", [s["pick"] for s in raw["trace"]], picks)
        require_equal("trace step values", [s["value"] for s in raw["trace"]], actual_steps)


def verify(csv_path, cache, *, partial=False, traces=None):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input contains no trajectories")
    suites = {r["suite"] for r in rows}
    invalid = suites.difference(SUITES)
    if invalid:
        raise ValueError(f"Unknown suites: {sorted(invalid)}")
    cases = {}
    for suite in sorted(suites if partial else SUITES):
        adapter = import_module("." + suite, __package__)
        cases[suite] = {c.name: c for c in adapter.load_cases(Path(cache))}
    expected = {(suite, case, seed, policy) for suite, members in cases.items() for case in members
                for seed in SEEDS for policy in POLICIES}
    seen = set()
    counts = Counter()
    for row in rows:
        suite, name, seed, policy = key = row["suite"], row["case"], int(row["seed"]), row["policy"]
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unrecognized run {key}")
        seen.add(key)
        try:
            verify_row(row, cases[suite][name], traces)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise ValueError(f"{key}: {exc}") from exc
        counts[suite] += 1
    if not partial and seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} trajectories, e.g. {min(expected - seen)}")
    return {"trajectories": len(seen), "step_objectives": len(seen) * ROUNDS, "by_suite": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Published per_run.csv")
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="Previously prepared benchmark data")
    parser.add_argument("--traces", type=Path, help="Optionally cross-check original trajectory JSON files")
    parser.add_argument("--partial", action="store_true", help="Allow an incomplete instance/seed/policy grid")
    args = parser.parse_args()
    print(json.dumps(verify(args.csv, args.cache, partial=args.partial, traces=args.traces), sort_keys=True))


if __name__ == "__main__":
    main()
