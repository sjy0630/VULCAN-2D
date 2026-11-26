"""Test whether one shared finite-supply bottleneck improves SET extrapolation."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import audit_iv41_high_conductance as reads
from analysis import validate_compact_set_physics as compact
from analysis import validate_set_independent_slope as shape
from analysis import validate_set_retained_links as retained


ALPHAS = (0.,.005,.015,.05)


def limited_ratio(voltage,slope,alpha):
    weight = np.sinh(slope*voltage)/np.sinh(slope*compact.REFERENCE_V)
    return weight/(1+alpha*(weight-1))


def estimate_slopes(rows,circuit,alpha):
    if not alpha:
        return shape.estimate_slopes(rows,circuit)[0]
    values = []
    for _,row,g in rows:
        stop = compact.prefix_end(row,circuit)
        ids,local,i = reads.reader.intrinsic(row,circuit,True)
        ids = ids[(ids<=stop)&(local[ids]>=shape.MIN_LOCAL_V)&(local[ids]<=shape.MAX_LOCAL_V)]
        if len(ids)<shape.MIN_POINTS:
            values.append(np.nan)
            continue
        observed = np.log10(i[ids]/(g*compact.REFERENCE_V))
        candidate = np.log10(limited_ratio(local[ids],shape.SLOPE_GRID[:,None],alpha))
        error = np.abs(candidate-observed)
        loss = np.mean(np.where(error<=.15,.5*error**2,.15*(error-.075)),axis=1)
        values.append(float(shape.SLOPE_GRID[np.argmin(loss)]))
    return np.asarray(values)


class LimitedRamp(shape.VariableSlopeRamp):
    def __init__(self,wave,circuit,step,slopes,alpha):
        super().__init__(wave,circuit,step,slopes)
        self.alpha = alpha

    def current(self,v,g,initial,_hop):
        slope = self.slopes[:,None] if np.ndim(v)==2 else self.slopes
        return (initial*compact.REFERENCE_V*limited_ratio(v,slope,self.alpha)
                + np.maximum(g-initial,0)*v)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("cache","output","figure"):
        ap.add_argument("--"+k,type=Path,required=True)
    args=ap.parse_args()
    index=json.loads((args.cache/"index.json").read_text())
    blocks=compact.get_cohorts(index,args.cache)
    train,test={},{}
    for gate,(all_rows,rows,sc) in blocks.items():
        ok = np.isfinite(estimate_slopes(rows,sc,0.))
        kept = [r for r,k in zip(rows,ok) if k]
        cut = int(.7*len(all_rows)) if gate<=1.3 else 0
        if cut:
            train[gate]=(all_rows,[r for r in kept if r[0]<cut],sc)
        test[gate]=(all_rows,[r for r in kept if r[0]>=cut],sc)
    fits={}
    for alpha in ALPHAS:
        def factory(rows,sc,step):
            return LimitedRamp(rows[0][1].voltage,sc,step,estimate_slopes(rows,sc,alpha),alpha)
        fits[alpha]=compact.fit_growth(train,1.,False,ramp_factory=factory)
    chosen=min(fits,key=lambda a:fits[a].cost)
    candidate=min(ALPHAS[1:],key=lambda a:fits[a].cost)
    results,panels={},[]
    for gate,(all_rows,rows,sc) in test.items():
        v=rows[0][1].voltage
        obs,mask,g=compact.data_matrix(rows,sc)
        curves={}
        for alpha in (0.,candidate):
            slopes=estimate_slopes(rows,sc,alpha)
            ramp=LimitedRamp(v,sc,.0125,slopes,alpha)
            name="unlimited" if alpha==0 else "finite_supply"
            curves[name],_,states=ramp.predict(g,1.,fits[alpha].x,False)
            assert np.all(np.diff(states[:,:ramp.apex+1],axis=1)>=-1e-12)
            assert np.allclose(ramp.current(np.full_like(g,compact.REFERENCE_V),g,g,1.),
                               g*compact.REFERENCE_V,rtol=1e-12,atol=0.)
        tight=LimitedRamp(v,sc,.00625,estimate_slopes(rows,sc,candidate),candidate).predict(
            g,1.,fits[candidate].x,False)[0]
        results[str(gate)]={"n":len(rows),"scores":retained.scored(rows,sc,curves),
                            "max_refinement_error_decade":float(np.max(np.abs(
                                np.log10(curves["finite_supply"][mask]/tight[mask]))))}
        if gate in (.9,1.4):
            panels.append((f"{gate:g} V gate",rows,curves["unlimited"],curves["finite_supply"]))
        if gate==1.5:
            after=np.array([reads.terminal_estimate(r,False,.15) for _,r,_ in rows])
            for label,flags in (("lower written conductance",after<=30),("higher written conductance",after>30)):
                ix=np.argmin(np.abs(v[:ramp.apex+1]-1.4))
                valid=flags&mask[:,ix]
                results[str(gate)][label]={"n":int(flags.sum()),
                    **{f"at_1p4V_{name}_uA":float(np.median(trace[valid,ix])*1e6)
                       for name,trace in {"observed":obs,**curves}.items()}}
                panels.append((f"1.5 V gate · {label}",[r for r,k in zip(rows,flags) if k],
                               curves["unlimited"][flags],curves["finite_supply"][flags]))
    report={"protocol":"Fit shared saturation on 0.9–1.3 V training groups only. Per-scan field sensitivity is estimated from the 0.30–0.49 V measured hBN prefix for each candidate. Per-scan current scale at hBN 0.5 V is matched exactly. A shared finite-rate serial transport step limits the hopping part; the new conducting links remain in parallel. No measured points after the prefix enter test prediction. Both 1.4/1.5 V are excluded from all shared fits, but previously inspected in model development. The plot shows the best nonzero bottleneck against zero to expose its effect.",
            "best_training_alpha":chosen,"tested_nonzero_alpha":candidate,
            "fits":{str(k):{"training_cost":float(f.cost),"log_rate_slope_log_cap_uS":f.x.tolist()}
                    for k,f in fits.items()},"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    retained.plot(args.figure,panels,baseline_label="Unrestricted hopping",
                  candidate_label="Finite-supply hopping",
                  title="SET outward | finite-rate bottleneck sensitivity")
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
