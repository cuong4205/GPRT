"""Summarize saved 32-cycle outputs without retraining or selecting on test."""
from __future__ import annotations

import ast
import csv
import hashlib
import itertools
import json
import re
import statistics as st
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'end_to_end_32cycle/run_01'
OUT = ROOT / 'docs/analysis/end_to_end_32cycle'

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def mean(rows, key):
    return st.mean(float(r[key]) for r in rows)

def describe(values):
    values = list(values)
    return dict(mean=st.mean(values), sd=st.stdev(values) if len(values)>1 else 0,
                median=st.median(values), min=min(values), max=max(values))

def edges(r):
    return {(a,b) for route in json.loads(r['routes']) for a,b in zip(route,route[1:])}

def canonical(r):
    return sorted(tuple(route) for route in json.loads(r['routes']))

def compare(a,b):
    ea,eb=edges(a),edges(b)
    return dict(exact=canonical(a)==canonical(b), jaccard=len(ea&eb)/len(ea|eb))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    static=read_csv(RUN/'static_test/test_results.csv')
    episodes=read_csv(RUN/'dynamic/final/episodes.csv')
    search=read_csv(RUN/'training/search_metrics.csv')
    manifest=read_json(RUN/'pipeline_manifest.json')
    heuristics=read_json(RUN/'heuristics_manifest.json')['policies']
    checks=[]
    for stage,data in manifest['stages'].items():
        for rel,expected in data.get('artifacts',{}).items():
            path=RUN/rel
            actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            checks.append(dict(stage=stage,path=rel,matched=actual==expected))
    seeds=sorted({r['seed'] for r in search})
    result=dict(artifact_checks=dict(total=len(checks),mismatches=[c for c in checks if not c['matched']]),
                counts=dict(static=len(static),final=len(episodes),pilot=len(read_csv(RUN/'dynamic/pilot/episodes.csv'))))
    assert result['counts']==dict(static=132,final=528,pilot=396)
    result['source_checks']=[]
    for rel,expected in manifest['plan']['source_sha256'].items():
        data=(ROOT/rel).read_bytes()
        result['source_checks'].append(dict(path=rel,raw_match=hashlib.sha256(data).hexdigest()==expected,
            lf_match=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==expected))
    result['stages']={k:dict(status=v['status'],seconds=(datetime.fromisoformat(v['completed_at_utc'])-datetime.fromisoformat(v['started_at_utc'])).total_seconds()) for k,v in manifest['stages'].items()}
    result['offline']={m:{k:describe(float(r[k]) for r in search if r['method']==m) for k in ['end_to_end_seconds','peak_rss_mb','instance_evaluations','train_instance_evaluations','screening_instance_evaluations','validation_instance_evaluations','diagnostic_instance_evaluations','fitness_requests']} for m in ['gp','gprt']}
    result['static']={m:{k:describe(float(r[k]) for r in static if r['method']==m) for k in ['vehicles','distance','runtime_ms']} for m in ['edd','gp','gprt']}
    pairs=[]
    lookup={(r['method'],r['seed'],r['instance']):r for r in static}
    for seed in seeds:
        for instance in sorted({r['instance'] for r in static}):
            a,b=(lookup[m,seed,instance] for m in ['gp','gprt'])
            ka,kb=(tuple(float(r[k]) for k in ['unserved','vehicles','distance']) for r in [a,b])
            pairs.append(dict(seed=seed,instance=instance,family=a['family'],distance_delta=float(b['distance'])-float(a['distance']),vehicles_delta=float(b['vehicles'])-float(a['vehicles']),lex_result='win' if kb<ka else 'loss' if kb>ka else 'tie',**compare(a,b)))
    result['static_pairs']={k:sum(r['lex_result']==k for r in pairs) for k in ['win','loss','tie']}
    result['static_pairs'].update(exact=sum(r['exact'] for r in pairs),jaccard=describe(r['jaccard'] for r in pairs),distance_wins=sum(r['distance_delta'] < -1e-8 for r in pairs),distance_losses=sum(r['distance_delta'] > 1e-8 for r in pairs))
    result['seed_quality']=[]
    for seed in seeds:
        row={'seed':seed}
        for m in ['gp','gprt']:
            ss=[r for r in static if r['method']==m and r['seed']==seed]
            dd=[r for r in episodes if r['policy']==m and r['training_seed']==seed and r['scenario_mode']=='dynamic']
            row[m]=dict(vehicles=mean(ss,'vehicles'),distance=mean(ss,'distance'),dynamic_distance=mean(dd,'distance'),dynamic_served=mean(dd,'served'))
        pp=[r for r in pairs if r['seed']==seed]
        row['lex_wlt']=[sum(r['lex_result']==k for r in pp) for k in ['win','loss','tie']]
        result['seed_quality'].append(row)
    result['family_quality']={f:{m:{k:mean([r for r in static if r['family']==f and r['method']==m],k) for k in ['vehicles','distance']} for m in ['gp','gprt']} for f in sorted({r['family'] for r in static})}
    result['within_method']={}
    for m in ['gp','gprt']:
        cc=[compare(lookup[m,s1,i],lookup[m,s2,i]) for s1,s2 in itertools.combinations(seeds,2) for i in sorted({r['instance'] for r in static})]
        result['within_method'][m]=dict(pairs=len(cc),exact=sum(c['exact'] for c in cc),jaccard=describe(c['jaccard'] for c in cc))
    result['heuristics']=[]
    for p in heuristics:
        tree=ast.parse(re.sub(r'\b(and|or)\(', r'bool_\1(', p['expression']),mode='eval').body
        def size(n): return 1+sum(size(a) for a in n.args) if isinstance(n,ast.Call) else 1
        def depth(n): return 1+max(depth(a) for a in n.args) if isinstance(n,ast.Call) else 0
        result['heuristics'].append(dict(method=p['method'],seed=p['seed'],expression=p['expression'],nodes=size(tree),depth=depth(tree),features=sorted({n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}-{n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call)})))
        summary=read_json(RUN/f"training/{p['method']}/{p['seed']}/training_summary.json")
        assert summary['completed_cycles']==32 and summary['training_coverage_count']==32
        assert hashlib.sha256((RUN/f"training/{p['method']}/{p['seed']}/final_heuristic.txt").read_bytes()).hexdigest()==p['sha256']
    result['learning']=[]
    for seed in seeds:
        rows=read_csv(RUN/f'training/gprt/{seed}/learning_curves.csv')
        assert len(rows)==32
        keys=['neural_unique_samples','entropy','neural_direct_survival_rate','neural_better_than_random_rate','neural_direct_elite_count']
        result['learning'].append(dict(seed=seed,first8={k:mean(rows[:8],k) for k in keys},last8={k:mean(rows[-8:],k) for k in keys}))
    result['dynamic']={}
    for mode in ['static','dynamic']:
        result['dynamic'][mode]={}
        for m in ['edd','gp','gprt']:
            rr=[r for r in episodes if r['scenario_mode']==mode and r['policy']==m]
            result['dynamic'][mode][m]={k:mean(rr,k) for k in ['served','unserved','cancelled','served_rate','distance','distance_per_served','vehicles_used','trip_count','callback_latency_p95_ms','wall_clock_seconds','violation_count','late_served','infeasible_orders']}
    dyn=[r for r in episodes if r['scenario_mode']=='dynamic']
    dl={(r['policy'],r['training_seed'],r['instance'],r['scenario_seed']):r for r in dyn}
    dp=[]
    for r in dyn:
        if r['policy']!='gprt':continue
        a=dl['gp',r['training_seed'],r['instance'],r['scenario_seed']]
        assert a['scenario_sha256']==r['scenario_sha256']
        # Equal unserved and cancelled ID sets imply equal served sets, since all requests are shared.
        same_ids=all(set(json.loads(a[k]))==set(json.loads(r[k])) for k in ['unserved_customer_ids','cancelled_customer_ids'])
        dp.append(dict(seed=r['training_seed'],instance=r['instance'],scenario_seed=r['scenario_seed'],served_delta=float(r['served'])-float(a['served']),distance_delta=float(r['distance'])-float(a['distance']),same_workload=same_ids))
    result['dynamic_pairs']=dict(count=len(dp),served_wins=sum(r['served_delta']>0 for r in dp),served_losses=sum(r['served_delta']<0 for r in dp),served_ties=sum(r['served_delta']==0 for r in dp),same_workload=sum(r['same_workload'] for r in dp),distance_delta_equal_workload=describe(r['distance_delta'] for r in dp if r['same_workload']),distance_wins_equal_workload=sum(r['same_workload'] and r['distance_delta'] < -1e-8 for r in dp),distance_losses_equal_workload=sum(r['same_workload'] and r['distance_delta'] > 1e-8 for r in dp))
    for name,rows in [('paired_static',pairs),('paired_dynamic',dp)]:
        with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['heuristics','static','offline','learning']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
