"""Constrain low-field SET current and test faster field-driven state growth."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_log_soft as rd
from analysis import validate_activated_reset as reset
from analysis import validate_delayed_reset as delay
from analysis import validate_field_log_soft as field
from analysis import validate_log_soft_cycle as cy
from vulcan2d import model_shared_device as model
from vulcan2d import validate_shared_device as shared


WINDOWS = {"low_outward":(0.,.8),"transition_outward":(.8,2.),"high_outward":(3.,5.)}
REFERENCE_V = .5


def background_data(groups):
    samples = []
    for gate,rows in groups.items():
        traces = []
        for s,_,sc,_ in rows:
            ids,v,i = rd.intrinsic(s,sc,outward=True)
            ids = ids[(v[ids] >= .2)&(v[ids] <= .8)]
            if len(ids):
                traces.append((v[ids],np.log10(i[ids])))
        if not traces:
            raise ValueError(f"No low-bias observations at {gate:g} V gate")
        samples.append(traces)
    return samples


def background(v, p, form):
    amp, slope = p
    v = np.asarray(v)
    if form == "sqrt_field":
        return 10**amp*(v/REFERENCE_V)*np.exp(slope*(np.sqrt(v)-np.sqrt(REFERENCE_V)))
    return 10**amp*np.sinh(slope*v)/np.sinh(slope*REFERENCE_V)


def fit_background(samples, form):
    def residual(p):
        return np.concatenate([
            (np.log10(background(v,p,form))-obs)/np.sqrt(len(group)*len(obs))
            for group in samples for v,obs in group])
    fit = least_squares(residual,[-9.,5.],bounds=([-14.,.05],[-5.,16.]))
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def currents(v,s,initial,ratio,exponent,p,form):
    conductance = np.maximum(10**s-10**initial,0.)
    return background(v,p,form)+ratio*1e-6*field.REF*conductance*(v/field.REF)**exponent


def collect_metrics(rows, prediction):
    result = {}
    for name,limits in {"whole_set":None,**WINDOWS}.items():
        errors = []
        for s,_,_,_ in rows:
            mask = s.masks["use_unclamped"].copy()
            if limits is not None:
                lo,hi = limits
                mask &= s.masks["branch_outward"] & (s.voltage>=lo) & (s.voltage<=hi)
            errors.extend(cy.log_errors(s,prediction,mask))
        result[name] = {"rmse_decade":rd.rmse(errors),"points":len(errors)}
    return result


def fit_growth(groups, device, prep, initial, ratio, p, form):
    def law(v,s):
        return currents(v,s,initial,ratio,device.soft_exponent,p,form)
    ramps = {g:field.SetRamp(rows,device,prep,initial,ratio,.02,current_law=law)
             for g,rows in groups.items()}
    targets = {g:float(np.median(field.entry_targets(rows))) for g,rows in groups.items()}
    datasets = {}
    for g,rows in groups.items():
        v = rows[0][0].voltage
        masks = [((v>=lo)&(v<=hi)&rows[0][0].masks["branch_outward"])
                 for lo,hi in WINDOWS.values()]
        masks.append(~rows[0][0].masks["branch_outward"])
        datasets[g] = [shared._training_data([s for s,_,_,_ in rows],
                            use_transition=False,additional_mask=region) for region in masks]

    def objective(x):
        parts = []
        for g,ramp in ramps.items():
            state,current,_ = ramp.predict(x)
            for idx,(_,mask,y,w) in enumerate(datasets[g]):
                parts.append(np.sqrt([1.,2.,.5,.5][idx])*w*(np.log10(current[mask])-y))
            parts.append(np.array([state-targets[g]]))
        return np.concatenate(parts)

    fits = [least_squares(objective,x,bounds=([-6.,.6,.65],[10.,2.35,3.]),
                         diff_step=1e-4,ftol=1e-5,max_nfev=200)
            for x in ([3.,1.1,1.2],[5.,1.3,2.5],[.5,1.7,2.])]
    best = min((f for f in fits if f.success),key=lambda f:f.cost)
    return law,best.x,float(best.cost)


def draw(path, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(1,3,figsize=(13,4.4),layout="constrained")
    for ax,(g,rows,old,new) in zip(axes,panels):
        v = rows[0][0].voltage
        all_current = np.abs(np.stack([s.current for s,_,_,_ in rows]))
        mask = np.stack([s.masks["use_unclamped"]&s.masks["branch_outward"] for s,_,_,_ in rows])
        ids = np.flatnonzero(mask.any(axis=0))
        lo,mid,hi = np.nanquantile(np.where(mask[:,ids],all_current[:,ids],np.nan),[.1,.5,.9],axis=0)*1e6
        ax.fill_between(v[ids],lo,hi,color=".88",label="Measured p10–p90")
        ax.plot(v[ids],mid,color=".15",label="Measured median")
        ax.plot(v[ids],old[ids]*1e6,"--",color="#486d93",label="Previous SET")
        ax.plot(v[ids],new[ids]*1e6,color="#9b2e3e",label="New shared SET")
        ax.set(yscale="log",title=f"{g:g} V gate · {len(rows)} held-out cycles",
               xlabel="Applied SET voltage (V)",ylabel="Current magnitude (µA)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("SET outward sweep | identical experimental screening")
    fig.savefig(path,dpi=185)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-set","reset-fit","reset-readout","output","figure"):
        ap.add_argument("--"+name,type=Path,required=True)
    args = ap.parse_args()
    previous = json.loads(args.previous_set.read_text())["field_model"]
    rfit = json.loads(args.reset_fit.read_text())
    beta = json.loads(args.reset_readout.read_text())["transport_beta_per_sqrt_V"]
    pairs = cy.data()
    train = {.9:pairs[.9][:70],1.1:pairs[1.1][:32]}
    test = {.9:pairs[.9][70:],1.1:pairs[1.1][32:],1.3:pairs[1.3]}
    nominal = train[1.1]
    device,prep = shared.fit_soft_set([s for s,_,_,_ in nominal],nominal[0][2])
    initial,ratio = previous["initial_log_conductance"],previous["readout_positive_negative_ratio"]
    old_growth = [np.log(previous["rate_per_V3"]),previous["threshold_local_v"],
                  previous["maximum_log_conductance"]]
    points = background_data(train)
    fits = {}
    for form in ("sqrt_field","hopping_sinh"):
        p = fit_background(points,form)
        law,growth,cost = fit_growth(train,device,prep,initial,ratio,p,form)
        fits[form] = (p,law,growth,cost)
    selected = min(fits,key=lambda k:fits[k][3])
    report,panels = {},[]
    fit = rfit["fit_parameters"]["continuous"]
    params = [np.log(fit["rate_at_local_0p5V_per_V"]),fit["field_slope_per_V"]]
    floor = np.log10(rfit["background_uS"])

    def reverse(v,state,_a,_b):
        return 1e-6*10**state*delay.response(v,beta)

    for g,rows in test.items():
        old_st,old,_ = field.SetRamp(rows,device,prep,initial,ratio,.0125).predict(old_growth)
        rr = reset.ResetRamp(rows,None,None,.01,reverse,evolve_return=True)
        estimators = {"previous":{"entry_predicted_uS":10**old_st,"set":collect_metrics(rows,old)}}
        traces = {"previous":old}
        _,i0,_ = field.SetRamp(rows,device,prep,initial,ratio,.0125).predict(old_growth)
        np.testing.assert_allclose(i0,old,rtol=1e-12)
        for name,(_,law,growth,_) in fits.items():
            state,current,_ = field.SetRamp(rows,device,prep,initial,ratio,.0125,current_law=law).predict(growth)
            fine,fine_current,_ = field.SetRamp(rows,device,prep,initial,ratio,.00625,current_law=law).predict(growth)
            valid = np.any([s.masks["use_unclamped"] for s,_,_,_ in rows],axis=0)
            traces[name] = current
            estimators[name] = {"entry_predicted_uS":10**state,"set":collect_metrics(rows,current),
                                "refinement_state_decade":abs(fine-state),
                                "refinement_iv_decade":float(np.max(np.abs(np.log10(fine_current[valid]/current[valid]))))}
        for estimator in estimators.values():
            state = np.log10(estimator["entry_predicted_uS"])
            waveform,_ = rr.predict([state],floor,params)
            errors = np.concatenate([cy.log_errors(r,waveform[0],r.masks["use_unclamped"]) for _,r,_,_ in rows])
            estimator["frozen_parameters_reset_rmse_decade"] = rd.rmse(errors)
        cuts = {}
        v = rows[0][0].voltage
        for target in (.5,1.5,3.):
            index = int(np.argmin(np.abs(v[:np.argmax(v)+1]-target)))
            obs = [abs(s.current[index]) for s,_,_,_ in rows if s.masks["use_unclamped"][index]]
            if obs:
                cuts[str(target)] = {"observed_median_uA":float(np.median(obs)*1e6),
                                     **{k:float(y[index]*1e6) for k,y in traces.items()}}
        report[str(g)] = {"cycles":len(rows),"entry_observed_median_uS":float(10**np.median(field.entry_targets(rows))),
                          "predictions":estimators,"outward_current_at_selected_V_uA":cuts}
        panels.append((g,rows,old,traces[selected]))
    result = {"protocol":"Shared SET parameters. First 70 cycles of 0.9 V and cycles 4–35 of 1.1 V train the off-current and state kinetics. 1.3 V is entirely excluded from training; the other gates use chronological holdouts. Select on training loss only. hBN model never receives gate voltage; gate is used for the transistor lookup. Identical experimental mask. RESET refit is excluded so any changes in its score follow from the SET state only.",
              "selected":selected,
              "fits":{k:{"background_parameters":p.tolist(),"growth_parameters":x.tolist(),
                         "training_cost":cost} for k,(p,_,x,cost) in fits.items()},"results":report}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    draw(args.figure,panels)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
