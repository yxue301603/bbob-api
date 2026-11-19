"""Run one optimizer with the official BBOB observer and package its traces."""
import argparse
from contextlib import chdir, closing
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys

from run import ARMS, RULES, choose, cocoex, initial, np, pool, prepare, seed, validate_url


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="raw_rule")
    parser.add_argument("--full", action="store_true", help="Use the full default cocoex bbob suite")
    parser.add_argument("--dimensions", nargs="+", type=int, choices=[2, 3, 5, 10, 20, 40])
    parser.add_argument("--functions", nargs="+", type=int, choices=range(1, 25))
    parser.add_argument("--instances", nargs="+", type=int)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--budget", type=int, help="Total evaluations per problem (default 70)")
    budgets.add_argument("--budget-per-dim", type=int, help="Total evaluations per dimension")
    parser.add_argument("--model", default=os.getenv("BBO_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("BBO_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--key-env", default="BBO_API_KEY")
    parser.add_argument("--effort")
    parser.add_argument("--name", help="Short, unique algorithm name for COCO plots and archive")
    parser.add_argument("--out", type=Path, default=Path("outputs/coco"))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.full and any((args.dimensions, args.functions, args.instances)):
        parser.error("--full cannot be combined with suite filters")
    for values in (args.instances, args.dimensions, args.functions):
        if values and (min(values) < 1 or len(set(values)) != len(values)):
            parser.error("Suite filters must be distinct positive integers")
    if args.budget is None and args.budget_per_dim is None:
        args.budget = 70
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model or "model")
    args.name = args.name or (args.arm if args.arm in RULES else f"{args.arm}-{model_tag}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", args.name):
        parser.error("Use a short ASCII algorithm name with letters, digits, dots, underscores or dashes")
    return args


def suite_options(args):
    if args.full:
        return "", ""
    def ids(values):
        return ",".join(map(str, values))
    return ("instances: " + ids(args.instances or [1]),
            "dimensions: " + ids(args.dimensions or [5]) +
            " function_indices: " + ids(args.functions or [1]))


def budget(args, dimension):
    return args.budget if args.budget is not None else args.budget_per_dim * dimension


def optimize(problem, args, key, db):
    limit = budget(args, problem.dimension)
    # The instance selects only a random stream shared by competing algorithms.
    stream = problem.id_instance
    points = initial(0, stream, problem.dimension)
    values = []
    def evaluate(point):
        # All BBOB variables of interest are scaled from [-1,1] to [-5,5].
        return float(problem(5 * point))
    for point in points:
        values.append(evaluate(point))
        if problem.final_target_hit:
            return
    values = np.asarray(values)
    for t in range(limit - len(values)):
        cached = db.execute("SELECT point FROM decisions WHERE problem=? AND step=?",
                            (problem.id, t)).fetchone()
        if cached:
            point = np.array(json.loads(cached[0]))
        else:
            candidates = pool(points, values, seed(0, stream, t + 1))
            prompt, pick, _, _ = prepare(points, values, candidates, t, args.arm, limit - 20)
            call = {}
            if prompt is not None:
                call = choose(prompt, args.model, args.base_url, key, args.effort, len(candidates))
                pick = call["pick"]
            point = candidates[pick]
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)", (problem.id, t,
                       json.dumps(point.tolist()), call.get("returned_model"), json.dumps(call.get("usage"))))
            db.commit()
        value = evaluate(point)
        if problem.final_target_hit:
            break
        points = np.vstack((points, point))
        values = np.append(values, value)


def collect(args, key, db, options):
    suite = cocoex.Suite("bbob", *options)
    observer = cocoex.Observer("bbob", dict(result_folder=args.name, algorithm_name=args.name))
    folder = Path(observer.result_folder)
    total = 0
    for problem in suite:
        problem.observe_with(observer)
        try:
            optimize(problem, args, key, db)
            total += problem.evaluations
            print(f"{problem.id}: {problem.evaluations} evaluations", flush=True)
        finally:
            problem.free()
    return folder, total


def main():
    args = arguments()
    options = suite_options(args)
    suite = cocoex.Suite("bbob", *options)
    limits = [budget(args, p.dimension) for p in suite]
    if not limits or min(limits) < 20:
        raise ValueError("Each selected problem needs at least 20 evaluations for initialization")
    decisions = sum(limits) - 20 * len(limits) if args.arm not in RULES else 0
    print(f"{len(limits)} problems, at most {sum(limits)} objective evaluations and "
          f"{decisions} model decisions.", flush=True)
    if args.plan:
        return
    key = os.getenv(args.key_env)
    if decisions:
        validate_url(args.base_url)
        if not key or not args.model:
            raise ValueError(f"Set {args.key_env} and --model for API decisions")
    source = Path(__file__).resolve().parent
    config = dict(name=args.name, arm=args.arm, suite="bbob", options=options, budget=args.budget,
                  budget_per_dim=args.budget_per_dim, model=args.model if decisions else None,
                  base_url=args.base_url if decisions else None, effort=args.effort if decisions else None,
                  python=sys.version.split()[0], dependencies={p: version(p) for p in
                  ("coco-experiment", "numpy", "scipy")}, source_sha256={
                  name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in
                  ("coco_run.py", "run.py", "pilot.py", "backend.py", "methods.py", "api_client.py")})
    work = args.out.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with chdir(work), closing(sqlite3.connect("lock.sqlite", timeout=0)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        manifest = Path("config.json")
        encoded = json.dumps(config, indent=2, sort_keys=True)
        if manifest.exists() and manifest.read_text(encoding="utf-8") != encoded:
            raise ValueError("This output directory belongs to another configuration. Choose a new --out.")
        if not manifest.exists():
            with manifest.open("x", encoding="utf-8") as f:
                f.write(encoded)
        archive = Path(args.name + ".zip")
        if not archive.exists():
            with closing(sqlite3.connect("decisions.sqlite")) as db:
                db.execute("CREATE TABLE IF NOT EXISTS decisions (problem TEXT, step INTEGER, "
                           "point TEXT, model TEXT, usage TEXT, PRIMARY KEY(problem, step))")
                folder, total = collect(args, key, db, options)
            if not list(folder.glob("*.info")):
                raise RuntimeError("COCO observer produced no index")
            temp = shutil.make_archive(".packing", "zip", root_dir=folder.parent, base_dir=folder.name)
            Path(temp).replace(archive)
            print(f"Completed {len(limits)} problems with {total} objective evaluations.")
        print(work / archive)


if __name__ == "__main__":
    main()
