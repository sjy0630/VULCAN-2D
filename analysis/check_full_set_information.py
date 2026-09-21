"""Nested target-excluded information audit of complete SET and its history.

Ridge maps are diagnostic controls, not structural evolution laws. They do not
consume target RESET current as an input. Previous-RESET persistence is a
separate explicitly measurement-conditioned reference, scored only for 2--53.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from diagnose_cross_polarity_physics import read_pairs
from compare_parallel_paths import FOLDS


def features_and_target(pairs):
    rows=[]; cumulative=0.; previous=[]
    windows=(('low',.1,.3),('middle',.5,.8),('upper',1.,1.5),('high',2.,3.))
    for pair in pairs:
        raw=pair['set'];v=np.abs(raw['voltage']);current=np.abs(raw['current'])
        first=int(np.argmax(v));last=len(v)-1-int(np.argmax(v[::-1]));idx=np.arange(len(v))
        row={'cycle_id':pair['cycle_id']}
        for name,lo,hi in windows:
            for phase,mask in [('outward',idx<=first),('return',idx>=last)]:
                use=mask&(v>=lo-1e-10)&(v<=hi+1e-10)
                if sum(use)<5 or np.any(current[use]<=0):raise RuntimeError('Insufficient positive SET observations')
                row[f'{phase}_{name}_log10_A']=float(np.mean(np.log10(current[use])))
            row[f'{name}_return_minus_outward']=row[f'return_{name}_log10_A']-row[f'outward_{name}_log10_A']
        exposure=float(np.trapezoid(current[:first+1],v[:first+1]));cumulative+=exposure
        row['log10_SET_exposure_A_V']=float(np.log10(exposure))
        row['log10_cumulative_SET_exposure_A_V']=float(np.log10(cumulative))
        row['log10_cycle_index']=float(np.log10(pair['cycle_id']))
        for fraction in (.1,.5):
            selected=np.flatnonzero(current[:first+1]>=fraction*max(current[:first+1]))
            row[f'first_SET_{int(100*fraction)}pct_max_current_applied_V']=float(v[selected[0]])
        row['previous_SET_return_low_log10_A']=previous[-1]['return_low_log10_A'] if previous else row['outward_low_log10_A']
        previous.append(row)
        raw=pair['reset'];vr=np.abs(raw['voltage']);ir=np.abs(raw['current']);apex=int(np.argmax(vr))
        use=(np.arange(len(vr))<=apex)&(vr>=.1-1e-10)&(vr<=.3+1e-10)
        if sum(use)!=11 or np.any(ir[use]<=0):raise RuntimeError('Native RESET target incomplete')
        row['target_RESET_low_mean_log10_A']=float(np.mean(np.log10(ir[use])))
        row['target_RESET_zero_reference_A']=float(np.max(ir[vr<=1e-12]))
        row['target_RESET_low_points_not_above_3zero']=int(sum(ir[use]<=3*row['target_RESET_zero_reference_A']))
        rows.append(row)
    return rows


def train_ridge(x,y,penalty):
    center=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
    z=(x-center)/scale;intercept=float(y.mean())
    coefficient=np.linalg.solve(z.T@z+penalty*np.eye(x.shape[1]),z.T@(y-intercept)) if x.shape[1] else np.array([])
    return {'center':center.tolist(),'scale':scale.tolist(),'intercept':intercept,'coefficients':coefficient.tolist(),'penalty':penalty}


def evaluate_ridge(x,model):
    return model['intercept']+((x-model['center'])/model['scale'])@np.array(model['coefficients'])


def select_and_fit(x,y,train):
    blocks=np.array_split(train,3);results=[]
    for penalty in (.1,1.,10.,100.,1000.):
        residual=[]
        for target in blocks:
            fitting=np.setdiff1d(train,target)
            model=train_ridge(x[fitting],y[fitting],penalty)
            residual.extend(evaluate_ridge(x[target],model)-y[target])
        results.append({'penalty':penalty,'inner_validation_rmse_decade':float(np.sqrt(np.mean(np.square(residual))))})
    chosen=min(results,key=lambda r:r['inner_validation_rmse_decade'])['penalty']
    return train_ridge(x[train],y[train],chosen),results


def summaries(y,prediction,indices):
    output={}
    for group,mask in [('all',np.ones(len(indices),bool)),('first3',indices<3),('later',indices>=3)]:
        selected=indices[mask]
        if not len(selected):continue
        residual=prediction[selected]-y[selected]
        output[group]={'n_pairs':len(selected),'rmse_decade':float(np.sqrt(np.mean(residual**2))),
                       'bias_decade':float(np.mean(residual)),'maximum_abs_error_decade':float(np.max(np.abs(residual)))}
    return output


def run():
    rows=features_and_target(read_pairs('1.1'));ids=np.array([r['cycle_id'] for r in rows]);y=np.array([r['target_RESET_low_mean_log10_A'] for r in rows])
    full=[key for key in rows[0] if not key.startswith('target_') and key not in ('cycle_id','log10_cumulative_SET_exposure_A_V','log10_cycle_index','previous_SET_return_low_log10_A')]
    groups={'constant':[],'SET_return_low':['return_low_log10_A'],'full_SET':full,
            'full_SET_and_history':full+['log10_cumulative_SET_exposure_A_V','log10_cycle_index','previous_SET_return_low_log10_A'],
            'cycle_only':['log10_cycle_index']}
    protocols={'five_fold':[(f'fold_{lo}_{hi}',np.flatnonzero((ids<lo)|(ids>hi)),np.flatnonzero((ids>=lo)&(ids<=hi))) for lo,hi in FOLDS],
               'leave_one_pair_out':[(f'hold_{cid}',np.flatnonzero(ids!=cid),np.flatnonzero(ids==cid)) for cid in ids],
               'chronological':[('first35_to_last18',np.flatnonzero(ids<=35),np.flatnonzero(ids>35))]}
    result={'scope':'Nested target-excluded SET information audit; statistical controls, not microscopic dynamics or autonomous cycles',
            'raw_feature_rows':rows,'feature_groups':groups,'candidates':{},
            'interpretation':['SET inputs include the entire target SET, so they are available before its RESET, but this does not simulate SET.',
                              'A V exposure is a voltage-domain stress proxy, not charge or physical energy without scan timing.',
                              'Zero-reference low-current flags remain in primary targets.',
                              'All feature standardization and penalty selection are nested within the training pairs.',
                              'Leave-one-pair-out allows other initial cycles in training; grouped first-fold extrapolation does not. These are different generalization questions.']}
    for name,keys in groups.items():
        x=np.array([[r[k] for k in keys] for r in rows]);out={}
        for protocol,splits in protocols.items():
            predicted=np.full(len(ids),np.nan);details=[]
            for label,train,target in splits:
                model,selection=select_and_fit(x,y,train)
                predicted[target]=evaluate_ridge(x[target],model)
                details.append({'label':label,'training_ids':ids[train].tolist(),'target_ids':ids[target].tolist(),
                                'model':model,'inner_penalty_selection':selection})
            indices=np.flatnonzero(np.isfinite(predicted))
            out[protocol]={'predictions_log10_A':[float(v) if np.isfinite(v) else None for v in predicted],
                           'groups':summaries(y,predicted,indices),'splits':details}
        result['candidates'][name]=out
        print(name,{p:r['groups'] for p,r in out.items()},flush=True)
    persistence=np.r_[np.nan,y[:-1]]
    result['previous_measured_RESET_reference']={'scope':'Uses prior measured RESET, no prediction for cycle1; not autonomous.',
           'predictions_log10_A':[None]+y[:-1].tolist(),'groups':summaries(y,persistence,np.arange(1,53))}
    files=[Path(__file__),ROOT/'analysis/diagnose_cross_polarity_physics.py',ROOT/'analysis/compare_parallel_paths.py',
           ROOT/'1T1M写入.xlsx',ROOT/'1T1M擦除.xlsx',ROOT/'analysis/COUPLED_CHARGE_NETWORK_DESIGN_2026-09-21.md']
    result['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'analysis/full_set_information_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':run()
