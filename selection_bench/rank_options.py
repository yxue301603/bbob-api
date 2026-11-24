"""Score the eight options retrospectively on each frozen trajectory.

Only published picks and their observed scores update the search history. Other
candidate scores are used strictly for the diagnostic, never for the next menu.
Ranks start at 1, ties receive their mean rank, and any tied minimum counts as a
best pick. An improvement must strictly beat the best previously observed value.
"""
import argparse
import csv
from collections import defaultdict
from importlib import import_module
from itertools import product
import json
from pathlib import Path

import numpy as np

from .run import INITIAL, OPTIONS, POLICIES, ROUNDS, SEEDS, SUITES, candidate_set, stream
from .verify import require_equal, vector


def rank_pick(scores, pick, incumbent):
    chosen = scores[pick]
    rank = 1 + sum(v < chosen for v in scores) + (sum(v == chosen for v in scores) - 1) / 2
    top = min(scores)
    opportunity = top < incumbent
    return rank, int(chosen == top), int(opportunity), int(opportunity and chosen >= incumbent)


def replay(row, case):
    suite, seed = row["suite"], int(row["seed"])
    picks = vector(row["picks"], integers=True)
    logged = vector(row["values"], integers=False)
    if len(picks) != ROUNDS or len(logged) != ROUNDS or any(not 0 <= p < OPTIONS for p in picks):
        raise ValueError("Invalid trace dimensions or candidate index")
    rng = np.random.default_rng(stream(suite, case.name, seed, "init"))
    observed = candidate_set(lambda: case.initial(rng), (), INITIAL)
    values = [case.evaluate(s) for s in observed]
    require_equal("initial", float(row["initial_best"]), min(values))
    ranks = []
    for step, (pick, saved) in enumerate(zip(picks, logged)):
        rng = np.random.default_rng(stream(suite, case.name, seed, step, "menu"))
        idx = int(np.argmin(values))
        options = candidate_set(lambda: case.propose(observed[idx], rng), observed, OPTIONS)
        scores = [case.evaluate(solution) for solution in options]
        require_equal(f"selected value at step {step}", saved, scores[pick])
        ranks.append(rank_pick(scores, pick, values[idx]))
        observed.append(options[pick])
        values.append(saved)
    idx = int(np.argmin(values))
    require_equal("final", float(row["best"]), values[idx])
    require_equal("solution", vector(row["solution"], integers=True), list(observed[idx]))
    return ranks


def summarize(paths, cache, suites):
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("Select each benchmark once")
    cases = {suite: {c.name: c for c in import_module("." + suite, __package__).load_cases(cache)}
             for suite in suites}
    expected = {(suite, *key) for suite, members in cases.items() for key in product(members, SEEDS, POLICIES)}
    seen = set()
    groups = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                suite, name, seed, policy = key = row["suite"], row["case"], int(row["seed"]), row["policy"]
                if suite not in cases:
                    continue
                if key not in expected or key in seen:
                    raise ValueError(f"Duplicate or unexpected run {key}")
                seen.add(key)
                try:
                    groups[suite, policy].extend(replay(row, cases[suite][name]))
                except (ValueError, KeyError, IndexError) as exc:
                    raise ValueError(f"{key}: {exc}") from exc
    if expected != seen:
        raise ValueError(f"Missing {len(expected-seen)} trajectories, e.g. {min(expected-seen)}")
    result = []
    for suite, policy in product(suites, POLICIES):
        steps = groups[suite, policy]
        rank, best, possible, missed = map(sum, zip(*steps))
        result.append(dict(suite=suite, policy=policy, trajectories=len(steps)//ROUNDS, rounds=len(steps),
                           mean_rank=rank/len(steps), best_picks=best, best_pick_rate=best/len(steps),
                           improving_pools=possible, missed_improvements=missed,
                           missed_given_improving_rate=missed/possible if possible else ""))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, required=True, help="One combined or separate per_run CSVs")
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--suite", nargs="+", default=SUITES, choices=SUITES)
    parser.add_argument("--out", type=Path, default=Path("results/option_ranks.csv"))
    args = parser.parse_args()
    rows = summarize(args.csv, args.cache, args.suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(args.out)
    print(json.dumps(dict(groups=len(rows), selected_scores_verified=sum(r["rounds"] for r in rows))))


if __name__ == "__main__":
    main()
