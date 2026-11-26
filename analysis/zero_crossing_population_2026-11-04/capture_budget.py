"""Frozen-table extrema and saved-state arithmetic only; no model solver."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np

HERE=Path(__file__).resolve().parent
TABLE=Path('/private/tmp/vulcan-runtime-cache/transport_g010_v0025.npz')
TABLE_SHA='00d6d9ea0b1dfe855181e7016afc6dd5e23d169f83f19cfcedfc9d3b35f9e4e2'
POP_SHA='5d6d55e7f89022888dd16577d33196e374795987df5bd0bd93af69a446b61db0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-authorized',action='store_true')
    if not parser.parse_args().run_authorized:parser.error('Explicit finite-plan execution required')
    destination=HERE/'capture_budget_result.json'
    if destination.exists():raise FileExistsError('No automatic rerun')
    load_started=time.monotonic()
    assert sha(TABLE)==TABLE_SHA and sha(HERE/'population_result.json')==POP_SHA
    pop=json.loads((HERE/'population_result.json').read_text())
    assert pop['status']=='completed_finite_population_screen' and pop['counts']['valid']==31
    with np.load(TABLE) as z:
        lg,lv,li=(z[key].copy() for key in ['log_gamma','log_voltage','log_current'])
        r0=json.loads(str(z['metadata']))['shape']['ron_ohm']
    sources={str(p):sha(p) for p in [TABLE,HERE/'population_result.json',HERE/'CAPTURE_BOUND_PLAN.md',Path(__file__)]}
    result={'status':'started','source_sha256':sources,'input_IO_wall_s':time.monotonic()-load_started,
        'parameters':pop['parameters'],'Ron_ohm':pop['Ron_ohm'],'gamma_c_ev':pop['gamma_c_ev'],
        'new_state_inversions':0,'new_circuit_solves':0,'new_ODEs':0,'new_fits':0,'new_b_trials':0,
        'original_holdout_accessed':False,'state_reanchoring':False,'table_bounds':[],'bridges':[]}
    started=time.monotonic()
    def check_budget():
        if time.monotonic()-started>5.:raise TimeoutError('5 s table-arithmetic budget exceeded')
    try:
        delta=.2; cut=math.log(delta); scale=math.log(r0/pop['Ron_ohm'])
        assert lv[0]<cut<lv[-1] and np.all(np.diff(lg)>0) and np.all(np.diff(lv)>0)
        j=int(np.searchsorted(lv,cut)-1);fraction=(cut-lv[j])/(lv[j+1]-lv[j])
        exact_cut=(1.-fraction)*li[:,:,j]+fraction*li[:,:,j+1]
        keep=lv<cut; vertices=np.r_[lv[keep],cut]
        for index,sign in [(0,1),(1,-1)]:
            check_budget()
            cut_table=np.column_stack([li[index][:,keep],exact_cut[index]])+scale-lg[:,None]
            assert cut_table.shape==(len(lg),len(vertices)) and np.all(np.isfinite(cut_table))
            row_max=np.max(cut_table,axis=1);column_max=np.max(cut_table,axis=0)
            maximum=float(np.max(cut_table));arg=np.unravel_index(int(np.argmax(cut_table)),cut_table.shape)
            assert float(max(row_max))==maximum and float(max(column_max))==maximum
            c=math.exp(maximum)
            result['table_bounds'].append({'sign':sign,'log_ratio_matrix_shape':list(cut_table.shape),
                'gamma_domain_ev':[math.exp(lg[0]),math.exp(lg[-1])],
                'local_voltage_domain_v':[math.exp(lv[0]),delta],
                'cut_original_indices':[j,j+1],'cut_fraction':fraction,
                'log_gamma_nodes':lg.tolist(),'max_log_ratio_per_gamma':row_max.tolist(),
                'maximum_log_ratio':maximum,'maximum_gamma_ev':math.exp(lg[arg[0]]),
                'maximum_local_voltage_v':math.exp(vertices[arg[1]]),
                'maximum_vertex_indices':[int(arg[0]),int(arg[1])],
                'C_a_per_ev':c,'C_padded_a_per_ev':c*(1.+1e-6)})
        factor=pop['parameters']['kappa_per_v2']*pop['gamma_c_ev']/pop['parameters']['J_positive_a']
        raw_L=factor*sum(row['C_a_per_ev'] for row in result['table_bounds'])*delta**2/2.
        decision_L=factor*sum(row['C_padded_a_per_ev'] for row in result['table_bounds'])*delta**2/2.*(1.+1e-6)
        result['bound']={'delta_v':delta,'kappa_gc_over_J':factor,'raw_maximum_u_decrease':raw_L,
            'padded_maximum_u_decrease':decision_L,'necessary_delta_u_lower':-decision_L,
            'C_margin_fraction':1e-6,'L_margin_fraction':1e-6}
        for row in pop['bridges']:
            check_budget();minus,plus=row['readpoints']
            most_favorable_delta=plus['numerical_padded_u_bracket'][1]-minus['numerical_padded_u_bracket'][0]
            capture_fail=most_favorable_delta < -decision_L
            recovery_fail=row['status']=='conditional_violation'
            result['bridges'].append({'reset_record':row['reset_record'],'set_record':row['set_record'],
                'largest_required_delta_u':most_favorable_delta,
                'required_decrease_over_upper':-most_favorable_delta/decision_L,
                'capture_bound_violated':capture_fail,'original_m2_growth_bound_violated':recovery_fail,
                'either_direction_violated':capture_fail or recovery_fail})
        result['counts']={'predeclared':31,'valid':31,
            'capture_violation':sum(r['capture_bound_violated'] for r in result['bridges']),
            'm2_growth_violation':sum(r['original_m2_growth_bound_violated'] for r in result['bridges']),
            'both':sum(r['capture_bound_violated'] and r['original_m2_growth_bound_violated'] for r in result['bridges']),
            'either':sum(r['either_direction_violated'] for r in result['bridges']),
            'neither_excluded':sum(not r['either_direction_violated'] for r in result['bridges'])}
        check_budget();result['status']='completed_fixed_table_capture_bound'
    except Exception as exc:
        result['status']='failed_or_unresolved';result['error']=repr(exc);raise
    finally:
        result['numerical_wall_s']=time.monotonic()-started
        payload=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        with destination.open('xb') as handle:handle.write(payload)
        print(json.dumps({'status':result['status'],'counts':result.get('counts'),
            'numerical_wall_s':result['numerical_wall_s'],'result_sha256':hashlib.sha256(payload).hexdigest()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
