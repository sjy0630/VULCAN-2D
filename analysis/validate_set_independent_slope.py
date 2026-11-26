"""Separate the initial SET current scale from its field sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import audit_iv41_high_conductance as reads
from analysis import validate_compact_set_physics as compact
from analysis import validate_set_retained_links as retained


MIN_LOCAL_V = .30
MAX_LOCAL_V = .49
MIN_POINTS = 6
SLOPE_GRID = np.linspace(.10,18.,360)


def estimate_slopes(rows,circuit):
    """Estimate shape without changing the current at the reference voltage."""
    fits, counts = [], []
    for _,row,g in rows:
        stop = compact.prefix_end(row,circuit)
        ids,local,i = reads.reader.intrinsic(row,circuit,True)
        ids = ids[(ids<=stop)&(local[ids]>=MIN_LOCAL_V)&(local[ids]<=MAX_LOCAL_V)]
        counts.append(len(ids))
        if len(ids)<MIN_POINTS:
            fits.append(np.nan)
            continue
        observed = np.log10(i[ids]/(g*compact.REFERENCE_V))
        candidates = np.log10(np.sinh(SLOPE_GRID[:,None]*local[ids]) /
                              np.sinh(SLOPE_GRID[:,None]*compact.REFERENCE_V))
        # Huber loss limits the effect of isolated low-current noise.
        delta = np.abs(candidates-observed)
        objective = np.mean(np.where(delta<=.15,.5*delta**2,.15*(delta-.075)),axis=1)
        fits.append(float(SLOPE_GRID[np.argmin(objective)]))
    return np.asarray(fits),np.asarray(counts)


class VariableSlopeRamp(compact.PositiveRamp):
    def __init__(self,wave,circuit,step,slopes):
        super().__init__(wave,circuit,step)
        self.slopes = np.asarray(slopes)

    def current(self,v,g,initial,_hop):
        slope = self.slopes[:,None] if np.ndim(v)==2 else self.slopes
        return super().current(v,g,initial,slope)


def subset(rows, mask):
    return [r for r,has in zip(rows,mask) if has]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache","revision","output","figure"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    base = json.loads(args.revision.read_text())
    hop = base["hopping_slope_per_V"]
    previous = base["fits"]["independent_links"]["log_rate_slope_log_cap_uS"]
    blocks = compact.get_cohorts(json.loads((args.cache/"index.json").read_text()),args.cache)
    train,test = {},{}
    for gate,(all_rows,rows,sc) in blocks.items():
        slope,counts = estimate_slopes(rows,sc)
        cut = int(.7*len(all_rows)) if gate<=1.3 else 0
        good = subset(rows,np.isfinite(slope))
        if cut:
            train[gate] = (all_rows,[r for r in good if r[0]<cut],sc)
        test[gate] = (all_rows,[r for r in good if r[0]>=cut],sc)

    def factory(rows,circuit,step):
        slopes,_ = estimate_slopes(rows,circuit)
        return VariableSlopeRamp(rows[0][1].voltage,circuit,step,slopes)

    fit = compact.fit_growth(train,hop,False,ramp_factory=factory)
    results,panels = {},[]
    for gate,(all_rows,rows,circuit) in test.items():
        wave = rows[0][1].voltage
        slopes,counts = estimate_slopes(rows,circuit)
        obs,mask,g = compact.data_matrix(rows,circuit)
        reference = compact.PositiveRamp(wave,circuit,.0125).predict(g,hop,previous,False)[0]
        model = VariableSlopeRamp(wave,circuit,.0125,slopes)
        shape_only = model.predict(g,hop,previous,False)[0]
        refit,_,states = model.predict(g,hop,fit.x,False)
        fine = VariableSlopeRamp(wave,circuit,.00625,slopes).predict(g,hop,fit.x,False)[0]
        assert np.allclose(model.current(np.full_like(g,compact.REFERENCE_V),g,g,hop),
                           g*compact.REFERENCE_V,rtol=1e-12,atol=0)
        assert np.all(np.diff(states[:,:model.apex+1],axis=1)>=-1e-12)
        curves = {"fixed_slope":reference,"independent_slope":shape_only,
                  "independent_slope_and_refit":refit}
        results[str(gate)] = {"n":len(rows),"slope_p10_p50_p90_per_V":np.quantile(slopes,[.1,.5,.9]).tolist(),
            "near_slope_bounds_count":int(np.sum((slopes<=SLOPE_GRID[0])|(slopes>=SLOPE_GRID[-1]))),
            "scores":retained.scored(rows,circuit,curves),
            "max_refinement_error_decade":float(np.max(np.abs(np.log10(refit[mask]/fine[mask]))))}
        if gate in (.9,1.4):
            panels.append((f"{gate:g} V gate",rows,reference,refit))
        if gate==1.5:
            after = np.array([reads.terminal_estimate(r,False,.15) for _,r,_ in rows])
            for label,group in (("lower written conductance",after<=30.),("higher written conductance",after>30.)):
                ix = int(np.argmin(np.abs(wave[:model.apex+1]-1.4)))
                valid = group & mask[:,ix]
                results[str(gate)][label] = {
                    "n":int(group.sum()),"median_slope_per_V":float(np.median(slopes[group])),
                    **{f"at_applied_1p4V_{k}_uA":float(np.median(z[valid,ix])*1e6)
                       for k,z in (("observed",obs),*curves.items())}}
                panels.append((f"1.5 V gate · {label}",subset(rows,group),reference[group],refit[group]))
    result = {
        "protocol":"Separate measured current at intrinsic 0.5 V from per-scan hopping shape, estimated exclusively from intrinsic 0.30–0.49 V prefix. Use same scans, strict disjoint scoring and positive growth model as fixed-slope comparator. Fit shared growth only on first 70% of IV41 sweeps in 0.9–1.3 V, neither 1.4 nor 1.5 V enters fitting shared parameters. Individual slope is an inferred initial condition, not an independent per-gate fit, and uses measured target prefix. Compare unchanged growth with refitted common growth.",
        "growth_refit_log_rate_slope_log_cap_uS":fit.x.tolist(),"growth_refit_cost":float(fit.cost),
        "fixed_slope_per_V":hop,"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    retained.plot(args.figure,panels,baseline_label="Fixed field sensitivity",
                  candidate_label="Independently estimated sensitivity",
                  title="SET outward | current amplitude and field sensitivity separated")
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
