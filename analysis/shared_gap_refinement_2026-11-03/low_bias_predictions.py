"""Unfitted low-bias predictions after bare SET; no new observations used."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vulcan2d.shared_gap import GapLaw
from vulcan2d.shared_access_gap import AccessGapLaw

if __name__=='__main__':
    folder=Path(__file__).parent
    rows=[]; sources={}
    for name,cls in [('accurate',GapLaw),('block',GapLaw),('access',AccessGapLaw)]:
        fit_path=folder/f'{name}_fit.json'; eval_path=folder/f'{name}_evaluation.json'
        if not fit_path.exists() or not eval_path.exists():continue
        fit=json.loads(fit_path.read_text());ev=json.loads(eval_path.read_text())
        law=cls(**fit['selected_law'])
        u=ev['same_initial_state_circuit_counterfactual']['circuits']['1r']['final_u']
        reads=[]
        for voltage in [.001,.005,.01,.025,.05]:
            current=law.current(voltage,u)
            reads.append({'V':voltage,'I_A':current,'R_ohm':voltage/current,
                          'positive_state_rate':law.derivative(u,current),
                          'negative_state_rate':law.derivative(u,-current)})
        rows.append({'candidate':name,'after_bare_SET_u':u,'frozen_reads':reads})
        for path in (fit_path,eval_path):
            sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output={'scope':'Unfitted frozen-state read predictions, not measured validation; rate checks delimit nondisturbing reads. Access is not accepted by the declared training criterion.',
            'candidates':rows,'source_sha256':sources}
    (folder/'low_bias_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
