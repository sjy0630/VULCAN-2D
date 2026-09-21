"""Follow-up diagnostics prompted by the prespecified transport comparison.

These tests are explicitly exploratory: shared low-conductance shape across
polarities, plus local SET-return power exponents. No model is overwritten.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from compare_transport_candidates import assemble_curves, subset, score, summarize
from vulcan2d.transport_candidates import Curve, fit_shared_shape, profile_amplitude


def run():
    points_path=HERE/"transport_candidate_native_points.csv"
    frame=pd.read_csv(points_path)
    records=frame.to_dict("records")
    branches={}
    for name in ("set_pre","reset_hrs"):
        curves,_,_,split=assemble_curves(records,name,"primary","signal_guard")
        branches[name]={c.cycle_id:c for c in curves}
    ids=sorted(set.intersection(*[set(x) for x in branches.values()]))
    output={"scope":"Exploratory follow-up after candidate comparison; fixed 1.1-V protocol",
            "voltage_window_V":[.5,1.2],"low_high_split_V":split,
            "shared_shape_assumption":"Same temperature, full nominal thickness field, and curvature parameter on opposite low-conductance branches; separate amplitudes.",
            "pair_ids":ids,"joint_shape":{},"connected_local_exponents":[]}
    for candidate in ("pf_free","schottky_free","hopping"):
        rows=[]
        all_curves=[Curve(1000*(j+1)+i,branches[b][i].voltage_v,branches[b][i].log10_current_a)
                    for j,b in enumerate(branches) for i in ids]
        fit_all=fit_shared_shape(all_curves,candidate)
        for target_id in ids:
            training=[subset(c,True,split) for c in all_curves if c.cycle_id%1000 != target_id]
            shared=fit_shared_shape(training,candidate)
            assert all(i%1000 != target_id for i in shared["training_cycle_ids"])
            for branch,curves in branches.items():
                target=curves[target_id]
                amplitude=profile_amplitude(subset(target,True,split),candidate,shared["parameter"])
                row=score(subset(target,False,split),shared,amplitude,
                          "pair_loco_shared_shape_low_read_conditioned_high",split)
                row["branch"]=branch;rows.append(row)
        output["joint_shape"][candidate]={"all_data_fit":fit_all,"evaluation":{
            b:summarize([r for r in rows if r["branch"]==b]) for b in branches},"cycle_scores":rows}
    connected=frame[(frame.branch=="set_connected")&frame.valid]
    for lo,hi in ((.15,.4),(.4,.8),(.8,1.2),(1.2,2.0),(2.0,3.0),(3.0,4.2)):
        group=connected[connected.hbn_voltage_V.between(lo,hi)&(connected.current_A>10*connected.zero_reference_A)]
        curves=[Curve(int(i),x.hbn_voltage_V.to_numpy(),np.log10(x.current_A.to_numpy()))
                for i,x in group.groupby("cycle_id") if len(x)>=5]
        fit=fit_shared_shape(curves,"power")
        local=[]
        for c in curves:
            x=np.log10(c.voltage_v);y=c.log10_current_a
            local.append(float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)))
        output["connected_local_exponents"].append({
            "window_V":[lo,hi],"n_cycles":len(curves),"shared_exponent":fit["parameter"],
            "per_cycle_exponent_q10_median_q90":np.quantile(local,[.1,.5,.9]).tolist(),
            "transistor_fraction_max":float(group.transistor_fraction.max()),
            "capacity_fraction_max":float(group.capacity_fraction.max()),
            "profiled_rmse_decade":fit["profiled_equal_cycle_rmse_decade"]})
    output["limitations"]=[
        "Sharing is a test of this simple uniform-field assumption; failure would not exclude every interface mechanism.",
        "Local exponent is a curve descriptor, not a measured SCLC exponent or unique transport mechanism.",
        "SET return changes along the sweep: state evolution and positive-transistor interpolation can confound an apparent crossover.",
        "Positive transistor has gates 1.0 and 1.5 V around 1.1; numerical gate interpolation is not a measurement at 1.1 V.",
        "Static monotone convex transport in series with a positive resistor has log slope at least one; observed sublinear intervals need further explanation.",
    ]
    output["input_sha256"]={points_path.name:hashlib.sha256(points_path.read_bytes()).hexdigest()}
    path=HERE/"transport_shape_consistency.json"
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(path)
    for name,r in output["joint_shape"].items():
        print(name,'shared',r['all_data_fit']['parameter'],
              {b:round(x['equal_cycle_rmse_decade'],4) for b,x in r['evaluation'].items()})
    print('Local SET exponents:',[(r['window_V'],round(r['shared_exponent'],4)) for r in output['connected_local_exponents']])


if __name__=="__main__":run()
