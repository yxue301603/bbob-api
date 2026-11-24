"""Paired experiments. Only the initial solutions and the selected option are scored."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path

import numpy as np

SUITES = ("ufl", "pmedian", "flowshop")
SEEDS = (11, 29, 47)
POLICIES = ("random", "deepseek", "deepseek-operon", "deepseek-gplearn", "greedy-operon", "greedy-gplearn")
INITIAL, ROUNDS, OPTIONS = 12, 12, 8
TASKS = {
    "ufl": "Minimize warehouse opening plus customer assignment costs.",
    "pmedian": "Choose a fixed number of graph centers to minimize total distance to nearest center.",
    "flowshop": "Order jobs to minimize the finish time on the last machine.",
}


def stream(*labels):
    key = "/".join(map(str, labels)).encode()
    return int.from_bytes(sha256(key).digest()[:4], "little")


def candidate_set(draw, excluded, size):
    result = []
    unique = set(excluded)
    for _ in range(10000):
        solution = tuple(map(int, draw()))
        if solution not in unique:
            unique.add(solution)
            result.append(solution)
            if len(result) == size:
                return result
    raise ValueError("Unable to generate enough distinct feasible options")


def compact(row):
    return ",".join(format(float(value), ".2f") for value in row)


def prompt_for(suite, names, X, y, Q, predictions, expression):
    best = int(np.argmin(y))
    lines = [TASKS[suite], "Lower observed score is better. Columns x0 onward are standardized from initial observations.",
             ",".join(names), "Observed features | score (best, then two most recent):"]
    for i in dict.fromkeys([best, len(y)-2, len(y)-1]):
        lines.append(compact(X[i]) + "|" + compact([y[i]]))
    lines.append("Options ID:features" + ("|predicted score" if predictions is not None else ""))
    for i, row in enumerate(Q):
        lines.append(f"{i}:" + compact(row) + ("|" + compact([predictions[i]]) if predictions is not None else ""))
    if expression is not None:
        lines.append("Fitted standardized-score formula: " + expression)
    return "\n".join(lines)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temp.replace(path)


def bind_model(out, model, endpoint):
    """Refuse to mix runs made with different model settings in one output tree."""
    if not model or not endpoint:
        raise ValueError("Set DEEPSEEK_MODEL and DEEPSEEK_BASE_URL for live selections")
    from api_client import validate_url
    from filelock import FileLock
    validate_url(endpoint)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model-settings.txt"
    expected = json.dumps(dict(protocol=1, model=model, endpoint=endpoint.rstrip("/"),
                               initial=INITIAL, rounds=ROUNDS, options=OPTIONS), sort_keys=True)
    with FileLock(str(path) + ".lock"):
        if path.is_file() and path.read_text(encoding="utf-8") != expected:
            raise ValueError("Model settings changed. Use another --out directory.")
        if not path.is_file():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(expected, encoding="utf-8")
            tmp.replace(path)


def run_case(suite, case, seed, policy, out, choose=None, replay=None):
    job = f"{suite}/{case.name}/{seed}/{policy}"
    dest = out / (job + ".json")
    if replay is not None:
        return trajectory(suite, case, seed, policy, job, dest, choose, replay)
    if dest.is_file():
        return
    from filelock import FileLock
    dest.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(dest) + ".lock"):
        if not dest.is_file():
            return trajectory(suite, case, seed, policy, job, dest, choose, replay)


def trajectory(suite, case, seed, policy, job, dest, choose, replay):
    receipt = dest.with_suffix(".steps")
    cached = json.loads(receipt.read_text()) if receipt.exists() and replay is None else []
    rng = np.random.default_rng(stream(suite, case.name, seed, "init"))
    observed = candidate_set(lambda: case.initial(rng), (), INITIAL)
    values = [case.evaluate(s) for s in observed]
    initial_best = min(values)
    use_features = replay is None and policy != "random"
    if use_features:
        initial_x = np.asarray([case.features(s) for s in observed])
        mean = initial_x.mean(axis=0)
        scale = initial_x.std(axis=0)
        scale[scale < 1e-10] = 1.
        center, unit = np.mean(values), max(np.std(values), 1e-10)
        history = list((initial_x - mean) / scale)
    steps = []
    method = policy.split("-")[-1] if "-" in policy else None
    for t in range(ROUNDS):
        rng = np.random.default_rng(stream(suite, case.name, seed, t, "menu"))
        incumbent = observed[int(np.argmin(values))]
        options = candidate_set(lambda: case.propose(incumbent, rng), observed, OPTIONS)
        if use_features:
            Q = (np.asarray([case.features(s) for s in options]) - mean) / scale
            X = np.asarray(history)
            y = (np.asarray(values) - center) / unit
        predictions = expression = digest = None
        if method and replay is None:
            from .sr import predict
            predictions, expression = predict(method, X, y, Q, stream(suite, case.name, seed, t, method))
        if replay is not None:
            pick = replay[t]
        elif policy == "random":
            pick = int(np.random.default_rng(stream(suite, case.name, seed, t, "pick")).integers(OPTIONS))
        elif policy.startswith("greedy"):
            pick = int(np.argmin(predictions))
        else:
            prompt = prompt_for(suite, case.feature_names, X, y, Q, predictions, expression)
            digest = sha256(prompt.encode()).hexdigest()
            if t < len(cached):
                if cached[t]["prompt_hash"] != digest:
                    raise ValueError(f"Changed prompt on resume for {job} step {t}")
                pick = cached[t]["pick"]
            else:
                pick = choose(job, t, prompt)
                if type(pick) is not int or not 0 <= pick < OPTIONS:
                    raise ValueError(f"Invalid selection for {job} step {t}")
                cached.append(dict(prompt_hash=digest, pick=pick))
                write_json(receipt, cached)
        if type(pick) is not int or not 0 <= pick < OPTIONS:
            raise ValueError("Invalid selection in trajectory")
        solution = options[pick]
        value = case.evaluate(solution)
        observed.append(solution)
        values.append(value)
        if use_features:
            history.append(Q[pick])
        steps.append(dict(value=value, pick=pick, prompt_hash=digest))
    idx = int(np.argmin(values))
    result = dict(suite=suite, case=case.name, seed=seed, policy=policy, reference=case.reference,
                  reference_kind=getattr(case, "reference_kind", "proven_optimum"), initial_best=initial_best,
                  best=values[idx], solution=observed[idx], trace=steps)
    if replay is None:
        write_json(dest, result)
        receipt.unlink(missing_ok=True)
    return result


def main(choose=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", nargs="+", choices=SUITES, default=SUITES)
    parser.add_argument("--policy", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--case", nargs="+", help="Optional exact instance IDs for a smoke test")
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--prepare-only", action="store_true", help="Download and check data, then exit")
    parser.add_argument("--out", type=Path, default=Path("outputs/selection"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not args.prepare_only and choose is None and any(p.startswith("deepseek") for p in args.policy):
        from .api import choose
        bind_model(args.out, os.environ.get("DEEPSEEK_MODEL"), os.environ.get("DEEPSEEK_BASE_URL"))
    jobs = []
    for suite in args.suite:
        adapter = import_module(f".{suite}", __package__)
        adapter.prepare(args.cache)
        if args.prepare_only:
            print(f"{suite} data ready")
            continue
        cases = adapter.load_cases(args.cache)
        jobs.extend((suite, case, seed, policy, args.out, choose)
                    for seed in args.seed for case in cases if not args.case or case.name in args.case
                    for policy in args.policy)
    if args.prepare_only:
        return
    if not jobs:
        parser.error("No matching instances")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda job: run_case(*job), jobs), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} trajectories", flush=True)


if __name__ == "__main__":
    main()
