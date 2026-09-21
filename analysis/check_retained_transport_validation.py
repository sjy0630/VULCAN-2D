"""Numerical validation for full transport fits; preserves any failed refinement."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from fit_retained_transport import digest
from compare_activated_configuration import assemble,lookups,scores
from check_retained_charge_contact import ratio34
from vulcan2d.retained_charge_configuration import trajectory,surfaces
from vulcan2d.retained_transport_parameters import unpack
from vulcan2d.side_charge_circuit import load_roots


def run(paths,output):
    d=assemble();lookup=lookups();mask=d['weights']>0;w=d['weights'][mask];rows={};inputs=[Path(__file__)];arrays={}
    for stem in paths:
        path=ROOT/'analysis'/f'{stem}.json';source=json.loads(path.read_text());assert source['status']=='complete'
        for name,sha in source['input_sha256'].items():assert digest(ROOT/name)==sha,name
        assert digest(path.with_suffix('.npz'))==source['prediction_sha256'];inputs.extend([path,path.with_suffix('.npz')])
        saved=np.load(path.with_suffix('.npz'));s,p=unpack(source['coordinates'])
        np.testing.assert_allclose(s,source['shared_parameters'],atol=0,rtol=0)
        np.testing.assert_allclose(p,source['parameters'],atol=0,rtol=0)
        for k in ('voltage','observed','cycle','window','weights','low_signal'):np.testing.assert_array_equal(saved[k],d[k])
        state=saved['states'];qw=saved['quadrature_weights'];settings=source['settings']
        repeated=np.flatnonzero(np.diff(d['voltage'])==0)+1
        np.testing.assert_array_equal(state[repeated],state[repeated-1])
        assert state.min()>=0 and np.max(abs(state.sum(axis=1)-1))<2e-11
        train=mask&np.isin(d['cycle'],source['training_ids']);err=np.log10(saved['current'][train]/d['observed'][train])
        metric=float(np.sqrt(np.dot(d['weights'][train],err**2)/d['weights'][train].sum()))
        assert abs(metric-source['training_scores']['rmse_decade'])<1e-12
        index=np.flatnonzero(mask)[np.linspace(0,mask.sum()-1,120).astype(int)]
        grid=np.arange(int(round(5/settings['table_step']))+1)*settings['table_step']
        table={sign:surfaces(sign*grid,s,p,panels=settings['panels'],order=settings['order']) for sign in (-1,1)}
        errors=[]
        for j in index:
            sign=1 if d['voltage'][j]>0 else -1;frozen=np.einsum('sn,svn,n->v',state[j],table[sign]['current'],qw)
            roots=load_roots(abs(d['voltage'][j]),grid,frozen,*lookup[0 if sign>0 else 1])
            vh=abs(saved['internal_voltage'][j]);best=min(roots,key=lambda r:abs(r['internal_voltage_V']-vh))
            error=abs(best['internal_voltage_V']-vh);assert error<1e-8 and best['stable_load_line'] and len(roots)==saved['root_count'][j]
            errors.append(error)
        row={'independent_training_rmse':metric,'selected_converged':source['selected_converged'],
             'independent_frozen_root_checks':len(index),'maximum_root_voltage_error_V':max(errors),'refinements':[]}
        for label,factor in [('refined',2),('finest',4)]:
            config=dict(settings,panels=settings['panels']*(2 if factor==2 else 3),
                        table_step=settings['table_step']/factor,max_step=settings['max_step']/factor)
            r=trajectory(d['voltage'],s,p,*lookup,**config,diagnostics=True)
            delta=np.log10(r['current_A'][mask]/saved['current'][mask]);worst=float(abs(delta).max())
            weighted=float(np.sqrt(np.dot(w,delta**2)/w.sum()))
            norm=float(np.max(abs(r['states'].sum(axis=1)-1)))
            entry={'label':label,'settings':config,'maximum_log_change':worst,'weighted_log_change':weighted,
                   'maximum_kcl_relative_error':r['maximum_kcl_relative_error'],'maximum_probability_sum_error':norm,
                   'full_scores':scores(d,r['current_A']),'training_scores':scores(d,r['current_A'],source['training_ids']),
                   'target_scores':scores(d,r['current_A'],source['target_ids']),
                   'SET_return_4_over_3':ratio34(d,r['current_A'],2),'RESET_outward_4_over_3':ratio34(d,r['current_A'],4),
                   'passes_declared_tolerances':bool(worst<.005 and weighted<.001 and norm<2e-11 and r['maximum_kcl_relative_error']<1e-8)}
            row['refinements'].append(entry);arrays[f'{stem}_{label}_current']=r['current_A']
            print(stem,label,worst,weighted,entry['passes_declared_tolerances'],flush=True)
        row['all_refinements_pass']=all(r['passes_declared_tolerances'] for r in row['refinements'])
        rows[stem]=row;saved.close()
    destination=ROOT/'analysis'/f'{output}.npz';np.savez_compressed(destination,**arrays)
    result={'scope':'Numerical validation of jointly fitted full native histories','candidates':rows,
            'all_refinements_pass':all(r['all_refinements_pass'] for r in rows.values()),'arrays_sha256':digest(destination),
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in inputs}}
    (ROOT/'analysis'/f'{output}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.paths,args.output)
