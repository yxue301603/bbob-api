"""Repeat the registered Operon comparison on Taillard and the complete OR-Library flowshop1 file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

from . import flowshop
from .run import bind_model, run_case

URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/flowshop1.txt"
DIGEST = "c7f50ed7204d91509f00bef71e7e0c6fbe4ad800a6c30bd5a298f8d3c07d002f"
SEEDS = (101, 131, 173)
ARMS = ("deepseek", "deepseek-operon", "greedy-operon")


def cases(cache):
    path = cache / "flowshop1.txt"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != DIGEST:
            raise ValueError("Unexpected upstream digest")
        path.write_bytes(data)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != DIGEST:
        raise ValueError("OR-Library checksum mismatch")
    chunks = re.split(r"(?im)^\s*instance[ \t]+(\w+)[ \t]*\r?$", raw.decode("ascii"))
    result = []
    for name, part in zip(chunks[1::2], chunks[2::2]):
        lines = [list(map(int, s.split())) for s in part.splitlines() if re.fullmatch(r"[\d\s]+", s) and s.strip()]
        n,m = lines[0]
        jobs = np.array(lines[1:])
        if jobs.shape != (n,2*m) or not (jobs[:,::2] == np.arange(m)).all():
            raise ValueError(f"Invalid job data: {name}")
        processing = jobs[:,1::2].T
        lb = max(processing.sum(axis=0).max(), processing.sum(axis=1).max())
        case = flowshop.FlowShopCase(name.lower(), processing, lb, lb, False)
        case.reference_kind = "workload_lower_bound"
        result.append(case)
    expected = ({f"car{i}" for i in range(1,9)} | {"hel1","hel2"}
                | {f"rec{i:02d}" for i in range(1,42,2)})
    if len(result) != 31 or {c.name for c in result} != expected:
        raise ValueError(f"Incorrect OR-Library coverage: {[c.name for c in result]}")
    flowshop.prepare(cache)
    return {"orlib31": result, "taillard": flowshop.load_cases(cache)}


def main(choose=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path(".cache"))
    p.add_argument("--out", type=Path, default=Path("outputs/confirmation"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--all-history", action="store_true", help="Auxiliary matched-history control on OR-Library")
    p.add_argument("--seed", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    args=p.parse_args()
    data = cases(args.cache)
    arms = ARMS
    if args.all_history:
        data = {"orlib31": data["orlib31"]}
        arms = ARMS[:2]
        args.out = args.out/"all-history"
    if choose is None:
        import os
        from .api import choose
        bind_model(args.out, os.environ.get("DEEPSEEK_MODEL"), os.environ.get("DEEPSEEK_BASE_URL"))
    jobs = [(s,c,seed,arm) for s,cs in data.items() for seed in args.seed for c in cs for arm in arms]
    def execute(job):
        suite,case,seed,arm=job
        def select(job,step,prompt):
            return choose(("all-history/" if args.all_history else "")+job,step,prompt)
        run_case("flowshop", case, seed, arm, args.out/suite, select, all_history=args.all_history)
    with ThreadPoolExecutor(args.workers) as pool:
        for i,_ in enumerate(pool.map(execute,jobs),1):
            if i%60==0 or i==len(jobs): print(i,len(jobs),flush=True)


if __name__=="__main__":
    main()
