"""Score new measurements against the frozen IV41 challenge without refitting."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from analysis import refit_iv41 as refit
from analysis import validate_paper_claim as plan
from vulcan2d import validate_shared_device as shared
from vulcan2d import validate_transition_regions as audit


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def calibration_key(row, phase):
    return row["experiment"],float(row["gate_set_V"]),float(row["gate_reset_V"]),phase


def load_new(path, schedule, forecasts):
    by_id={r["cycle_id"]:r for r in schedule}
    scans=defaultdict(list)
    for row in read_csv(path):
        ident, phase = row["cycle_id"], row["phase"]
        if ident not in by_id or phase not in plan.PHASES:
            raise ValueError(f"Unknown cycle or phase: {ident}, {phase}")
        run=by_id[ident]
        if any(not np.isclose(float(row[k]),float(run[k]),atol=1e-5,rtol=0)
               for k in ("gate_set_V","gate_reset_V")):
            raise ValueError(f"Recorded gates do not match the frozen schedule: {ident}")
        scans[ident,phase].append((int(row["point_index"]),float(row["voltage_V"]),
                                   float(row["current_A"]),row["device_id"].strip()))
    devices={x[-1] for samples in scans.values() for x in samples}
    if len(devices) != 1 or "" in devices:
        raise ValueError("Score each device separately and provide its device_id")
    clusters=defaultdict(list)
    measured=set()
    for (ident,phase), samples in scans.items():
        run=by_id[ident]
        if run["experiment"]=="control" and phase=="SET":
            continue
        k=calibration_key(run,phase)
        wave=next(iter(forecasts[k].values()))[0]
        points=sorted(samples)
        idx,v,i=np.asarray([r[:3] for r in points]).T
        if (not np.array_equal(idx,np.arange(len(idx))) or len(v)!=len(wave)
                or not np.allclose(v,wave,atol=1e-5,rtol=0) or not np.all(np.isfinite(i))):
            raise ValueError(f"Invalid grid or unreported missing points: {ident}/{phase}")
        gate=float(run["gate_set_V"] if phase=="SET" else run["gate_reset_V"])
        circuit=refit.circuits(gate)[plan.PHASES.index(phase)]
        masks,_=audit.prepare_curve(wave,i,phase,transistor=circuit.transistor)
        _,fixed,event=audit.filter_unclamped(wave,i,phase,masks)
        clusters[k].append(shared.Sweep(ident,wave,i,fixed,event))
        measured.add((ident,phase))
    return clusters,measured,devices.pop()


def matched_contrast(high, low, schedule, metric):
    by_id={r["cycle_id"]:r["block"] for r in schedule}
    def block_values(rows):
        result={}
        for r in rows:
            value=plan.reads(r.voltage,r.current,r.masks["use_unclamped"],fixed=True)[metric]
            if value is not None:
                result[by_id[r.label]]=value
        return result
    a,b=block_values(high),block_values(low)
    shared_blocks=sorted(a.keys()&b.keys())
    if not shared_blocks:
        return {"blocks":0,"median":None,"bootstrap_95":None}
    differences=np.array([a[k]-b[k] for k in shared_blocks])
    rng=np.random.default_rng(351)
    draw=rng.choice(differences,(2000,len(differences)),replace=True)
    return {"blocks":len(shared_blocks),"median":float(np.median(differences)),
            "bootstrap_95":np.quantile(np.median(draw,axis=1),[.025,.975]).tolist()
            if len(shared_blocks)>=5 else None}


def score(forecasts, measured, spec, schedule):
    result={}
    for key,rows in measured.items():
        experiment,gate_set,_,phase=key
        median,support=plan.phase_median(rows)
        v=rows[0].voltage
        if not support.any():
            raise ValueError(f"No common valid points at {key}")
        controlled=experiment=="control"
        items=[plan.measures(v,r.current,phase,r.masks["use_unclamped"],controlled) for r in rows]
        actual={k:plan.summarize([r[k] for r in items if r[k] is not None],len(rows)) for k in items[0]}
        methods={}
        for name,(_,current) in forecasts[key].items():
            predicted=plan.measures(v,current,phase,support,controlled)
            if controlled:
                predictions=spec["control"][str(gate_set)]["models"][name]
                predicted={k:qs["q10_q50_q90"][1] if qs["q10_q50_q90"] else None
                           for k,qs in predictions.items()}
            residue=refit.log_i(current[support])-median[support]
            methods[name]={"typical_rmse_decade":float(np.sqrt(np.mean(residue**2))),
                           "parameters":plan.compare_point_measures(actual,predicted)}
        result[f"{experiment}_{gate_set:g}_{phase}"]={"scans":len(rows),"observed":actual,"models":methods}
    high,low=(max(spec["write_gates_v"]),min(spec["write_gates_v"]))
    fixed=spec["control_reset_gate_v"]
    contrasts={}
    keys=(f"control_{high:g}_RESET",f"control_{low:g}_RESET")
    if all(k in result for k in keys):
        a,b=[result[k] for k in keys]
        for k in ("log10_lrs_s","log10_hrs_s"):
            actual=[r["observed"][k]["q10_q50_q90"] for r in (a,b)]
            change=actual[0][1]-actual[1][1] if all(x is not None for x in actual) else None
            estimated={}
            for name in a["models"]:
                q=[r["models"][name]["parameters"][k]["value"] for r in (a,b)]
                estimated[name]=q[0]-q[1] if all(x is not None for x in q) else None
            contrasts[k]={"observed_high_minus_low":change,"predicted_high_minus_low":estimated,
                          "matched_block_contrast":matched_contrast(
                              measured["control",high,fixed,"RESET"],
                              measured["control",low,fixed,"RESET"],schedule,k)}
    return result, contrasts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen_directory",type=Path)
    parser.add_argument("measurements",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args(argv)
    folder=args.frozen_directory
    spec=json.loads((folder/"paper_evidence.json").read_text())
    for key,path in (("forecast","frozen_predictions.csv"),("schedule","measurement_schedule.csv")):
        if refit.sha_file(folder/path)!=spec["sha256"][key]:
            raise ValueError("Frozen plan or predictions changed")
    for name,digest in spec["sha256"]["lookup"].items():
        if refit.sha_file(audit.ROOT/"vulcan2d/data"/name)!=digest:
            raise ValueError(f"Frozen transistor lookup changed: {name}")
    groups=defaultdict(lambda:defaultdict(list))
    for row in read_csv(folder/"frozen_predictions.csv"):
        key=calibration_key(row,row["phase"])
        groups[key][row["method"]].append((float(row["voltage_V"]),float(row["typical_current_A"])))
    forecasts={k:{m:np.array(v).T for m,v in x.items()} for k,x in groups.items()}
    schedule=read_csv(folder/"measurement_schedule.csv")
    measurements, recorded, device = load_new(args.measurements,schedule,forecasts)
    expected={(row["cycle_id"],phase) for row in schedule
              for phase in (plan.PHASES if row["experiment"]=="typical" else ("RESET",))}
    result,contrasts=score(forecasts,measurements,spec["frozen_experiment"],schedule)
    report={"device":device,"protocol_fully_measured":expected==recorded,
            "missing_condition_scans":len(expected-recorded),
            "phase_results":result,"control_contrasts":contrasts}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
