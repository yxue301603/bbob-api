"""Run a resumable BBOB API experiment and report paired scores and usage."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import cocoex
import numpy as np
from importlib.metadata import version
from api_client import choose, validate_url
from methods import prepare
from pilot import initial, pool, seed, write_atomic

ARMS = ("history", "raw_numeric", "raw_formula", "raw_rule", "k_numeric", "k_formula",
        "k_ei", "l_numeric", "l_formula", "l_ei")
RULES = {"raw_rule", "k_ei", "l_ei"}


def trajectory(args, arm, function, instance, key):
    path = args.out / f"f{function}-i{instance}-{arm}.json"
    problem = cocoex.BareProblem("bbob", function, 5, instance)
    if path.exists():
        run = json.loads(path.read_text(encoding="utf-8"))
    else:
        points = initial(function, 11 * instance)
        run = {"function": function, "instance": instance, "arm": arm,
               "init_z": points.tolist(), "init_y": [float(problem(5 * z)) for z in points],
               "optimum": float(problem.best_value()), "steps": []}
        write_atomic(path, run)
    points = np.array(run["init_z"])
    values = np.array(run["init_y"])
    for step in run["steps"]:
        points = np.vstack((points, step["z"]))
        values = np.append(values, step["value"])
    for t in range(len(run["steps"]), args.rounds):
        candidates = pool(points, values, seed(function, 11 * instance, t + 1))
        prompt, pick, fit, seconds = prepare(points, values, candidates, t, arm, args.rounds)
        call = None
        if prompt is not None:
            call = choose(prompt, args.model, args.base_url, key, args.effort, len(candidates))
            pick = call["pick"]
        point = candidates[pick]
        value = float(problem(5 * point))
        run["steps"].append({"index": pick, "z": point.tolist(), "value": value,
                             "best": min(float(values.min()), value), "fit_seconds": seconds,
                             "candidates": candidates.tolist(), "fit": fit, "call": call})
        write_atomic(path, run)
        points = np.vstack((points, point))
        values = np.append(values, value)
    gap = min(run["init_y"]) - run["optimum"]
    gain = 100 * (1 - (min(values) - run["optimum"]) / gap) if gap > 0 else 100.
    calls = [s["call"] for s in run["steps"] if s["call"] is not None]
    row = dict(function=function, instance=instance, arm=arm, gain=float(gain),
               best=float(min(values)), queries=len(values), calls=len(calls))
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        tokens = [(c["usage"] or {}).get(field) for c in calls]
        row[field] = sum(tokens) if all(type(n) is int for n in tokens) else None
    return row


def experiment(args):
    arms = tuple(args.arms)
    cases = len(args.functions) * len(args.instances)
    planned = cases * args.rounds * len(set(arms) - RULES)
    print(f"{cases} tasks, {cases*len(arms)} trajectories, {planned} planned model decisions, "
          f"{cases*len(arms)*(20+args.rounds)} function queries.", flush=True)
    if args.plan:
        return
    key = os.getenv(args.key_env)
    if planned and (not key or not args.model):
        raise ValueError(f"Set {args.key_env} and specify --model for API arms.")
    source = Path(__file__).resolve().parent
    config = dict(arms=arms, functions=args.functions, instances=args.instances, rounds=args.rounds,
                  model=args.model, base_url=args.base_url, effort=args.effort,
                  python=sys.version.split()[0],
                  dependencies={p:version(p) for p in ("numpy","scipy","coco-experiment")},
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(source.glob("*.py"))})
    args.out.mkdir(parents=True, exist_ok=True)
    lock = sqlite3.connect(args.out / ".run-lock.sqlite", timeout=0)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        manifest = args.out / "config.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
                raise ValueError("Configuration or code differs. Choose a new --out directory.")
        else:
            lock_files = {".run-lock.sqlite", ".run-lock.sqlite-journal"}
            if any(p.name not in lock_files for p in args.out.iterdir()):
                raise ValueError("Choose an empty output directory for a new experiment.")
            write_atomic(manifest, config)
        rows = []
        for function in args.functions:
            for instance in args.instances:
                for arm in arms:
                    row = trajectory(args, arm, function, instance, key)
                    rows.append(row)
                    print(f"F{function} i{instance} {arm}: {row['gain']:.2f}%", flush=True)
        with (args.out / "results.csv").open("w", encoding="utf-8-sig", newline="") as table:
            writer = csv.DictWriter(table, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary = {}
        for arm in arms:
            group = [r for r in rows if r["arm"] == arm]
            summary[arm] = {"average_gap_closed_percent": float(np.mean([r["gain"] for r in group]))}
            for field in ("queries", "calls", "prompt_tokens", "completion_tokens", "total_tokens"):
                numbers = [r[field] for r in group]
                summary[arm][field] = sum(numbers) if None not in numbers else None
        write_atomic(args.out / "summary.json", summary)
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("BBO_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("BBO_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--key-env", default="BBO_API_KEY", help="Name of the environment variable containing the API key")
    parser.add_argument("--effort", help="Optional reasoning_effort if supported by the model")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--functions", type=int, nargs="+", default=[8,9,15,19])
    parser.add_argument("--instances", type=int, nargs="+", default=[18,19,20])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("outputs/full"))
    parser.add_argument("--plan", action="store_true", help="Show planned calls without making any")
    args = parser.parse_args()
    if any(len(set(seq)) != len(seq) for seq in (args.arms,args.functions,args.instances)):
        parser.error("Duplicate arms, functions or instances.")
    if min(args.functions)<1 or max(args.functions)>24 or min(args.instances)<1 or args.rounds<1:
        parser.error("Use BBOB functions 1..24, positive instances and a positive round count.")
    try:
        validate_url(args.base_url)
        experiment(args)
    except (RuntimeError, ValueError, sqlite3.OperationalError) as error:
        parser.exit(1, f"{error}\n")
