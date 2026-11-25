"""Verify full trajectories and analyse the two prespecified paired makespan contrasts."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np

from .confirm import ARMS, SEEDS, cases
from .run import INITIAL, ROUNDS, run_case

RANDOM_SEED = 20261125


def randomization_p(differences):
    rng = np.random.default_rng(RANDOM_SEED)
    observed = abs(differences.sum())
    tail = 0
    n = 199999
    for start in range(0, n, 1000):
        signs = rng.integers(0,2,(min(1000,n-start),len(differences)),dtype=np.int8)*2-1
        tail += np.count_nonzero(abs((signs*differences).sum(axis=1))>=observed-1e-12)
    return (tail+1)/(n+1)


def percentile_interval(x):
    indices=np.random.default_rng(RANDOM_SEED).integers(len(x),size=(10000,len(x)))
    return np.percentile(x[indices].mean(axis=1),[2.5,97.5]).tolist()


def scalar_makespan(processing,solution):
    finish=[0]*(processing.shape[0]+1)
    for job in solution:
        for i,time in enumerate(processing[:,job],1):
            finish[i]=max(finish[i-1],finish[i])+int(time)
    return finish[-1]


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyse(cache,raw,out,replay_csv=None):
    history, sr, greedy = ARMS
    per_run, paired, statistics = [], [], []
    supplied={}
    if replay_csv:
        with replay_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key=row["suite"],row["case"],int(row["seed"]),row["policy"]
                if key in supplied:
                    raise ValueError(f"Duplicate trajectory {key}")
                supplied[key]=row
    available=cases(cache)
    experiments=[(s,cs,raw/s,ARMS) for s,cs in available.items()]
    experiments.append(("orlib31-full",available["orlib31"],raw/"all-history"/"orlib31",ARMS[:2]))
    for suite, instances, folder, arms in experiments:
        measured={}
        for case,seed in product(instances,SEEDS):
            records={}
            for policy in arms:
                job=folder/'flowshop'/case.name/str(seed)/(policy+'.json')
                if replay_csv:
                    r=supplied.pop((suite,case.name,seed,policy))
                    picks=json.loads(r['picks'])
                    vs=json.loads(r['step_values'])
                    data=dict(case=r['case'],seed=int(r['seed']),policy=r['policy'],
                              initial_best=float(r['initial']),best=float(r['best']),
                              reference=float(r['reference']),reference_kind=r['reference_kind'],
                              solution=json.loads(r['solution']),trace=[dict(pick=k,value=v)
                              for k,v in zip(picks,vs,strict=True)])
                else:
                    data=json.loads(job.read_bytes())
                assert (case.name,seed,policy)==(data['case'],data['seed'],data['policy'])
                picks=[v['pick'] for v in data['trace']]
                assert len(picks)==ROUNDS
                replay=run_case('flowshop',case,seed,policy,folder,replay=picks)
                for k in ('initial_best','best','reference','reference_kind'):
                    assert replay[k]==data[k]
                assert list(replay['solution'])==data['solution']
                assert scalar_makespan(case.processing,data['solution'])==data['best']
                values=[v['value'] for v in data['trace']]
                assert values==[v['value'] for v in replay['trace']]
                records[policy]=data
                per_run.append(dict(suite=suite,case=case.name,seed=seed,policy=policy,
                                    initial=data['initial_best'],best=data['best'],reference=data['reference'],
                                    reference_kind=data['reference_kind'],picks=json.dumps(picks),
                                    step_values=json.dumps(values),solution=json.dumps(data['solution'])))
            assert len({r['initial_best'] for r in records.values()})==1
            h,s=(records[k]['best'] for k in (history,sr))
            g=records[greedy]['best'] if greedy in records else None
            gain=100*(h-s)/h
            paired.append(dict(suite=suite,case=case.name,seed=seed,history=h,operon=s,greedy=g,
                               reduction_pct=gain,vs_greedy_pct=100*(g-s)/g if g is not None else None))
            measured[case.name,seed]=gain
        differences=np.asarray([[measured[c.name,k] for k in SEEDS] for c in instances])
        mean_cases=differences.mean(axis=1)
        lo,hi=percentile_interval(mean_cases)
        w=int((differences>0).sum()); t=int((differences==0).sum()); l=int((differences<0).sum())
        sw,st,sl=[int(v.sum()) for v in (mean_cases>0,mean_cases==0,mean_cases<0)]
        subset=[r for r in paired if r['suite']==suite]
        log_ratios=np.asarray([np.mean([np.log(r['history']/r['operon']) for r in subset if r['case']==c.name]) for c in instances])
        stats=dict(suite=suite,instances=len(instances),runs=len(subset),reduction_pct=float(mean_cases.mean()),
                   median_case=float(np.median(mean_cases)),ci_low=lo,ci_high=hi,p=randomization_p(mean_cases),
                   log_ratio_p=randomization_p(log_ratios),
                   wins=w,ties=t,losses=l,better_cases=sw,tied_cases=st,worse_cases=sl,
                   by_seed=differences.mean(axis=0).tolist(),
                   vs_greedy_pct=float(np.mean([r['vs_greedy_pct'] for r in subset])) if greedy in arms else None)
        if suite=='taillard':
            rows=[r for r in per_run if r['suite']==suite]
            stats['gap_pct']={p:float(np.mean([100*(r['best']/r['reference']-1) for r in rows if r['policy']==p])) for p in ARMS}
        statistics.append(stats)
    if supplied:
        raise ValueError("Unexpected extra trajectories")
    ordered=sorted([r for r in statistics if r['suite']!='orlib31-full'],key=lambda r:r['p'])
    previous=0
    for i,r in enumerate(ordered):
        r['p_holm']=min(1.,max(previous,(len(ordered)-i)*r['p']))
        previous=r['p_holm']
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'runs.csv',per_run)
    write_csv(out/'paired.csv',paired)
    (out/'statistics.json').write_text(json.dumps(statistics,indent=2)+'\n',encoding='utf-8')
    return statistics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('.cache'))
    p.add_argument('--raw',type=Path,default=Path('outputs/confirmation'))
    p.add_argument('--out',type=Path,default=Path('outputs/confirmation-stats'))
    p.add_argument('--replay-csv',type=Path,help='Rebuild from a published runs.csv without model calls')
    args=p.parse_args()
    print(json.dumps(analyse(args.cache,args.raw,args.out,args.replay_csv),indent=2))
