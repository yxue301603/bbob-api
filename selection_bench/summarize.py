"""Aggregate complete paired selector experiments, weighting instances equally within each seed."""
import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .run import OPTIONS, POLICIES, ROUNDS, SEEDS, SUITES


CASES = {
    "pmedian": tuple(f"pmed{i}" for i in range(1,41)),
    "flowshop": tuple(f"ta{i:03d}" for i in range(1,121)),
    "ufl": ("cap71","cap72","cap73","cap74","cap101","cap102","cap103","cap104",
            "cap131","cap132","cap133","cap134","capa","capb","capc"),
}
COMPARISONS = (("deepseek","random"), ("deepseek-operon","random"), ("deepseek-gplearn","random"),
               ("deepseek-operon","greedy-operon"), ("deepseek-gplearn","greedy-gplearn"),
               ("deepseek-operon","deepseek"), ("deepseek-gplearn","deepseek"))


def read_run(path):
    raw = json.loads(path.read_bytes())
    suite, case, seed, policy = (raw[key] for key in ("suite","case","seed","policy"))
    ref, initial, best = (raw[key] for key in ("reference","initial_best","best"))
    kind, steps = raw["reference_kind"], raw["trace"]
    if (type(seed) is not int or seed not in SEEDS or policy not in POLICIES or case not in CASES.get(suite,())
            or not isinstance(steps,list) or len(steps)!=ROUNDS or not raw.get("solution")
            or kind not in ("proven_optimum","best_known") or (suite!="flowshop" and kind!="proven_optimum")):
        raise ValueError(f"Invalid experimental design or incomplete trace in {path}")
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (ref,initial,best)):
        raise ValueError(f"Invalid objective values in {path}")
    picks = [s["pick"] for s in steps]
    values = [s["value"] for s in steps]
    if (any(type(k) is not int or not 0<=k<OPTIONS for k in picks)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values)
            or not math.isclose(best,min(initial,*values),rel_tol=0,abs_tol=1e-7)):
        raise ValueError(f"Invalid candidate selection or final minimum in {path}")
    tolerance = .001 if suite=="ufl" else 1e-7
    if kind=="proven_optimum" and best<ref-tolerance:
        raise ValueError(f"Observed score is better than the proved optimum: {path}")
    return dict(suite=suite,case=case,seed=seed,policy=policy,initial_best=initial,best=best,reference=ref,
                reference_kind=kind,initial_gap_pct=100*(initial-ref)/ref,gap_pct=100*(best-ref)/ref,
                hit_reference=int(best<=ref+tolerance),picks=json.dumps(picks,separators=(",",":")),
                values=json.dumps(values,separators=(",",":")),solution=json.dumps(raw['solution'],separators=(",",":")))


def collect(raw_dir, suites):
    rows, references, starts = {}, {}, {}
    for path in sorted(raw_dir.rglob("*.json")):
        prefix = path.relative_to(raw_dir).parts[0]
        if prefix in SUITES and prefix not in suites: continue
        row = read_run(path)
        suite, case, seed, policy = (row[k] for k in ("suite","case","seed","policy"))
        if suite not in suites: continue
        key = (suite,case,seed,policy)
        if key in rows: raise ValueError(f"Duplicate trajectory {key}")
        rows[key] = row
        ref = (row["reference"],row["reference_kind"])
        if references.setdefault((suite,case),ref)!=ref or starts.setdefault(key[:3],row["initial_best"])!=row["initial_best"]:
            raise ValueError(f"Paired reference or initial scores differ: {key}")
    required = {(suite,*keys) for suite in suites for keys in product(CASES[suite],SEEDS,POLICIES)}
    if set(rows)!=required:
        missing=sorted(required-rows.keys()); extras=sorted(rows.keys()-required)
        raise ValueError(f"Incomplete experiment: {len(missing)} missing (e.g. {missing[:2]}), {len(extras)} unexpected")
    return rows


def tables(rows, suites):
    runs, summary, comparisons = [], [], []
    for suite in suites:
        cases = CASES[suite]
        for policy in POLICIES:
            group = [rows[suite,case,s,policy] for s in SEEDS for case in cases]
            runs.extend(group)
            seed_gaps = {s:mean(r["gap_pct"] for r in group if r["seed"]==s) for s in SEEDS}
            summary.append(dict(suite=suite,policy=policy,instances=len(cases),seeds=len(SEEDS),trajectories=len(group),
                                mean_gap_pct=mean(seed_gaps.values()),seed_mean_gap_sample_sd_pct=stdev(seed_gaps.values()),
                                **{f"seed_{s}_mean_gap_pct":v for s,v in seed_gaps.items()},
                                hits_reference=sum(r["hit_reference"] for r in group),
                                proven_reference_runs=sum(r["reference_kind"]=="proven_optimum" for r in group),
                                best_known_reference_runs=sum(r["reference_kind"]=="best_known" for r in group)))
        for method, baseline in COMPARISONS:
            gain, wins, ties, losses = [], 0, 0, 0
            for case, s in product(cases, SEEDS):
                one, two = rows[suite,case,s,method],rows[suite,case,s,baseline]
                gain.append(two["gap_pct"]-one["gap_pct"])
                delta = two["best"]-one["best"]
                if abs(delta)<=1e-7: ties+=1
                elif delta>0: wins+=1
                else: losses+=1
            comparisons.append(dict(suite=suite,method=method,baseline=baseline,paired_runs=len(gain),
                                    wins=wins,ties=ties,losses=losses,mean_gap_reduction_pp=mean(gain)))
    return {"per_run":runs,"summary":summary,"paired":comparisons}


def summarize(raw_dir, out, suites=SUITES):
    if len(set(suites))!=len(suites) or not suites or not set(suites)<=set(SUITES):
        raise ValueError("Select each requested benchmark suite once")
    parts=tables(collect(Path(raw_dir),suites),suites)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    for name, group in parts.items():
        temp=out/(name+".csv.tmp")
        with temp.open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(group[0]))
            writer.writeheader()
            writer.writerows(group)
        temp.replace(out/(name+".csv"))
    return {name:len(group) for name,group in parts.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,default=Path("outputs/selection"))
    parser.add_argument("--out",type=Path,default=Path("outputs/selection-summary"))
    parser.add_argument("--suite",nargs="+",choices=SUITES,default=SUITES)
    args=parser.parse_args()
    print(json.dumps(summarize(args.raw,args.out,args.suite)))


if __name__=="__main__":
    main()
