"""Check RESET platform capacity and test a field-assisted transport candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_log_soft as readout
from analysis import validate_activated_reset as activated
from analysis import validate_entry_anchors as anchors_module
from analysis import validate_log_soft_cycle as cycle


def interval_envelope(local, start, end, anchors, shapes):
    # At fixed voltage, log current is piecewise affine in the log state.
    # Its extrema on a closed interval occur at an endpoint or an anchor.
    knots = np.r_[end,anchors[(anchors>end)&(anchors<start)],start]
    currents = cycle.current_law(local[None,:],knots[:,None],anchors,shapes)
    return currents.min(axis=0),currents.max(axis=0)


def capacity(rows, anchors, shapes):
    counts = np.zeros(4,dtype=int)
    cycles = 0
    over_by_cycle = []
    for _,r,_,rc in rows:
        start = cycle.observed_state(r,rc)
        end = activated.measured_end(r,rc)
        if start is None or end is None or end>start:
            continue
        ids,local,measured = readout.intrinsic(r,rc,outward=True)
        ids = ids[(np.abs(r.voltage[ids])>=.8)&(np.abs(r.voltage[ids])<=1.35)]
        if not len(ids):
            continue
        low,high = interval_envelope(local[ids],start,end,anchors,shapes)
        obs = measured[ids]
        counts += [len(ids),np.sum(obs>high),np.sum(obs>2*high),np.sum(obs<low)]
        over_by_cycle.append(float(np.median(np.log10(obs/high))))
        cycles += 1
    return {"cycles":cycles,"points":int(counts[0]),"above_max":int(counts[1]),
            "above_twice_max":int(counts[2]),"below_min":int(counts[3]),
            "median_cycle_log_ratio_to_max":float(np.median(over_by_cycle))}


def transport_fit(rows):
    early, returning = [], []
    for _,r,_,rc in rows:
        a = anchors_module.early_record(r,rc)
        b = readout.readout_record(r,rc)
        if a is not None:
            early.append(a)
        if b is not None:
            returning.append(b)
    v = readout.V_GRID
    x = np.sqrt(v)-np.sqrt(readout.REFERENCE_V)
    y0 = np.log10(v/readout.REFERENCE_V)
    xs,ys = [],[]
    for _,shape in early+returning:
        mask = np.isfinite(shape) & (v>=.20)
        if np.any(mask):
            w = np.sqrt(1/np.sum(mask))
            xs.extend(x[mask]*w)
            ys.extend((shape[mask]-y0[mask])*w)
    coefficient = max(0.,float(np.dot(xs,ys)/np.dot(xs,xs)))*np.log(10.)
    return coefficient,len(early),len(returning)


def score(rows, old, new):
    resets = [r for _,r,_,_ in rows]
    v = np.abs(resets[0].voltage)
    outward = resets[0].masks["branch_outward"]
    masks = np.stack([r.masks["use_unclamped"] for r in resets])
    current = np.abs(np.stack([r.current for r in resets]))
    result = {}
    for label,part in (("full",np.ones(len(v),dtype=bool)),("outward",outward),
                       ("return",~outward),("outward_platform",outward&(v>=.8)&(v<=1.35))):
        mask = masks & part
        result[label] = {name:readout.rmse(np.log10(
            np.broadcast_to(trace,current.shape)[mask]/current[mask]))
            for name,trace in (("interpolation",old),("field_assisted",new))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--figure",required=True,type=Path)
    parser.add_argument("--previous",required=True,type=Path)
    args = parser.parse_args()
    old = json.loads(args.previous.read_text())
    old_parameters = [np.log(old["rate_at_local_0p5V_per_V"]),old["log_rate_slope_per_V"]]
    floor = np.log10(old["background_uS"])
    pairs = cycle.data()
    seen = {.9:pairs[.9][:70],1.1:pairs[1.1][:32]}
    test = {.9:pairs[.9][70:],1.1:pairs[1.1][32:],1.3:pairs[1.3]}
    train = seen[.9]+seen[1.1]
    low = readout.learn_anchors([readout.readout_record(r,rc) for _,r,_,rc in train])
    records = [z for _,r,_,rc in train if (z:=anchors_module.early_record(r,rc)) is not None]
    anchors,shapes,_ = anchors_module.add_entry_anchors(records,*low)
    beta,n_early,n_return = transport_fit(train)

    def field_current(v,state,_anchors,_shapes):
        # Uses the same measured 0.15 V state scale for every gate.
        return 1e-6*10**state*v*np.exp(beta*(np.sqrt(v)-np.sqrt(readout.REFERENCE_V)))

    starts = np.linspace(-6,3,24)
    assert np.allclose(field_current(readout.REFERENCE_V,starts,None,None),
                       1e-6*10**starts*readout.REFERENCE_V,rtol=1e-12,atol=0)
    assert np.all(np.diff(field_current(np.linspace(0,2,101)[None,:],
                                       starts[:,None],None,None),axis=1)>0)
    fit_groups = []
    for rows in seen.values():
        known = [(r,s) for _,r,_,rc in rows if (s:=cycle.observed_state(r,rc)) is not None]
        mask = np.stack([r.masks["use_unclamped"] for r,_ in known])
        observed = np.log10(np.abs(np.stack([r.current for r,_ in known])[mask]))
        ramp = activated.ResetRamp(rows,anchors,shapes,.02,field_current)
        fit_groups.append((ramp,np.asarray([s for _,s in known]),mask,observed))

    def residual(parameters):
        penalties = []
        for ramp,states,mask,observed in fit_groups:
            predicted,_ = ramp.predict(states,floor,parameters)
            penalties.append((np.log10(predicted[mask])-observed)/np.sqrt(len(observed)))
        return np.concatenate(penalties)

    fits = [least_squares(residual,p,bounds=([-8.,0.],[9.,30.]),diff_step=1e-4,
                         ftol=1e-5,max_nfev=120) for p in ([0.,4.],[3.,8.])]
    best = min((fit for fit in fits if fit.success),key=lambda fit:fit.cost)
    result,panels = {},[]
    for gate,rows in test.items():
        initial = np.log10(old["results"][str(gate)]["entry_uS"])
        old_curve,_ = activated.ResetRamp(rows,anchors,shapes,.01).predict(
            [initial],floor,old_parameters)
        candidate = activated.ResetRamp(rows,anchors,shapes,.01,field_current)
        unrefitted,_ = candidate.predict([initial],floor,old_parameters)
        new_curve,end = candidate.predict([initial],floor,best.x)
        tight_curve,tight_end = activated.ResetRamp(rows,anchors,shapes,.005,field_current).predict(
            [initial],floor,best.x)
        valid = np.any([r.masks["use_unclamped"] for _,r,_,_ in rows],axis=0)
        result[str(gate)] = {"interval_capacity":capacity(rows,anchors,shapes),
                             "scores":score(rows,old_curve[0],new_curve[0]),
                             "predicted_end_uS":float(10**end[0]),
                             "max_grid_refinement_decade":float(np.max(np.abs(
                                 np.log10(tight_curve[0,valid]/new_curve[0,valid]))))}
        result[str(gate)]["readout_only_scores"] = score(rows,old_curve[0],unrefitted[0])
        panels.append((gate,[r for _,r,_,_ in rows],old_curve[0],new_curve[0]))
    report = {"protocol":"Use 0.9 V first 70 and 1.1 V cycles 4–35 only for fitting. Predict with the same previous SET entry and fixed background for both models. The capacity diagnostic uses measured endpoints and deembedded local voltage on 0.8–1.35 V external outward branch; it is an envelope bound conditional on monotone loss of the stored state. The field-assisted current formula is normalized to low-voltage conductance and fitted on presumed fixed-state early and return segments; it is a test of a shared field factor, not a microscopic identification.",
              "transport_beta_per_sqrt_V":beta,"training_early_cycles":n_early,
              "training_return_cycles":n_return,
              "eraser_rate_at_0p5V_per_V":float(np.exp(best.x[0])),
              "eraser_field_coefficient_per_V":float(best.x[1]),"results":result}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    activated.make_figure(args.figure,panels,old_label="Previous readout",new_label="Field-assisted readout")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
