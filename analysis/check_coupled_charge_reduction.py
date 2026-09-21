"""Exact current reduction on every fitted state and target-leakage checks."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from vulcan2d.coupled_charge import predict as full_predict
from vulcan2d.coupled_charge_reduced import predict as reduced_predict
from check_full_set_information import select_and_fit,evaluate_ridge


def run():
    sources=[ROOT/'analysis/coupled_charge_comparison.json',ROOT/'analysis/full_set_information_audit.json']
    main,information=[json.loads(p.read_text()) for p in sources]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for report in (main,information):
        for p,sha in report['input_sha256'].items():
            if digest(ROOT/p)!=sha:raise RuntimeError('Input changed: '+p)
    voltage=np.array(main['voltage_V']);checks=[]
    for label,row in main['splits'].items():
        fit=row['fit'];shared=fit['shared_parameters']
        states=[('training',fit['states'])]+[(key,value['identification']['states']) for key,value in row['targets'].items()]
        for key,local in states:
            old=full_predict(voltage,shared,local);new=reduced_predict(voltage,shared[:5],local)
            difference=float(np.max(np.abs(old-new)))
            if difference>1e-10:raise RuntimeError('Exact reduction failed')
            checks.append({'split':label,'scope':key,'n_state_vectors':len(local),'maximum_difference_decade':difference})
    raw=information['raw_feature_rows'];y=np.array([r['target_RESET_low_mean_log10_A'] for r in raw]);leakage=[]
    for name,columns in information['feature_groups'].items():
        x=np.array([[r[c] for c in columns] for r in raw])
        for row in information['candidates'][name]['five_fold']['splits']:
            train=np.array(row['training_ids'])-1;target=np.array(row['target_ids'])-1
            assert not set(train)&set(target)
            changed=y.copy();changed[target]+=100.
            model,selection=select_and_fit(x,changed,train)
            assert model==row['model'] and selection==row['inner_penalty_selection']
            actual=evaluate_ridge(x[target],model)
            recorded=np.array(information['candidates'][name]['five_fold']['predictions_log10_A'])[target]
            np.testing.assert_allclose(actual,recorded,rtol=0,atol=1e-12)
            leakage.append({'candidate':name,'split':row['label'],'target_RESET_100_decade_perturbation_changes_prediction':False})
    files=[Path(__file__),ROOT/'vulcan2d/coupled_charge_reduced.py',ROOT/'tests/test_coupled_charge_reduced.py',
           ROOT/'analysis/COUPLED_CHARGE_EXACT_REDUCTION_2026-09-21.md']+sources+[ROOT/p for r in (main,information) for p in r['input_sha256']]
    result={'scope':'Exact steady-current equivalence and nested information-map target exclusion; not a mechanism or full-cycle validation',
            'exact_reduction_checks':checks,'SET_information_leakage_checks':leakage,
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)}}
    (ROOT/'analysis/coupled_charge_reduction_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Equivalent state groups',len(checks),'maximum difference',max(r['maximum_difference_decade'] for r in checks),flush=True)
    print('Target perturbation checks',len(leakage),flush=True)


if __name__=='__main__':run()
