import csv
import json
from pathlib import Path
import tempfile
import unittest

from selection_bench import summarize as stats


def fixture(case,seed,policy):
    ref=100.
    gains={"random":0.,"deepseek":1.,"deepseek-operon":2.,"greedy-operon":2.,"deepseek-gplearn":3.,"greedy-gplearn":2.}
    best=ref+3+stats.SEEDS.index(seed)-gains[policy]
    return dict(suite="ufl",case=case,seed=seed,policy=policy,reference=ref,reference_kind="proven_optimum",
                initial_best=ref+12,best=best,solution=[0,1],trace=[dict(value=best,pick=7)]*12)


class SummaryTest(unittest.TestCase):
    def test_complete_whole_suite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root=Path(folder); raw=root/"raw"; out=root/"tables"; raw.mkdir()
            for case in stats.CASES['ufl']:
                for seed in stats.SEEDS:
                    for policy in stats.POLICIES:
                        row=fixture(case,seed,policy)
                        (raw/f"{case}-{seed}-{policy}.json").write_text(json.dumps(row))
            self.assertEqual(stats.summarize(raw,out,["ufl"]), {"per_run":270,"summary":6,"paired":7})
            def read(name):
                with (out/f"{name}.csv").open(newline='',encoding='utf-8') as f:
                    return list(csv.DictReader(f))
            group=next(r for r in read('summary') if r['policy']=='deepseek-gplearn')
            self.assertEqual(float(group['mean_gap_pct']),1.)
            self.assertEqual(float(group['seed_mean_gap_sample_sd_pct']),1.)
            self.assertEqual(int(group['hits_reference']),15)
            pair=next(r for r in read('paired') if r['method']=='deepseek-operon' and r['baseline']=='greedy-operon')
            self.assertEqual(int(pair['ties']),45)
            pair=next(r for r in read('paired') if r['method']=='deepseek-gplearn' and r['baseline']=='random')
            self.assertEqual(int(pair['wins']),45)
            self.assertEqual(float(pair['mean_gap_reduction_pp']),3.)
            one=read('per_run')[0]
            self.assertEqual(json.loads(one['picks']),[7]*12)
            self.assertEqual(json.loads(one['values']),[float(one['best'])]*12)
            self.assertEqual(json.loads(one['solution']),[0,1])
            (raw/'duplicate.json').write_text(json.dumps(fixture('cap71',11,'random')))
            with self.assertRaisesRegex(ValueError,"Duplicate trajectory"): stats.collect(raw,['ufl'])

    def test_incomplete_and_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); item=root/"case.json"; data=fixture('cap71',11,'random')
            item.write_text(json.dumps(data));
            with self.assertRaisesRegex(ValueError,"269 missing"): stats.collect(root,['ufl'])
            data['trace'][0]['pick']=8; item.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"Invalid candidate"): stats.read_run(item)
            data['trace']=data['trace'][:11]; item.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"incomplete trace"): stats.read_run(item)

    def test_published_ufl_rounding(self):
        with tempfile.TemporaryDirectory() as folder:
            item=Path(folder)/'case.json'; data=fixture('capb',11,'random')
            data['best']=data['reference']+.00057
            data['trace']=[dict(pick=0,value=data['best']) for _ in range(12)]; item.write_text(json.dumps(data))
            self.assertEqual(stats.read_run(item)['hit_reference'],1)
