"""Test a residual ohmic path in parallel with a shared field-assisted SET current."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import audit_iv41_high_conductance as ivread
from analysis import validate_compact_set_physics as compact


def hop_ratio(v,hop):
    return np.sinh(hop*v)/np.sinh(hop*compact.REFERENCE_V)


def initials(rows,circuit,hop):
    """Estimate the ohmic fraction at the 0.5 V reference, from prefix only."""
    fractions,improvement,counts = [],[],[]
    candidates = np.linspace(0.,1.,101)
    for _,row,g in rows:
        stop = compact.prefix_end(row,circuit)
        ids,v,i = ivread.reader.intrinsic(row,circuit,True)
        ids = ids[(ids<=stop)&(v[ids]>=.12)&(v[ids]<=.49)]
        x = v[ids]
        if len(x)<4:
            fractions.append(0.)
            improvement.append(0.)
            counts.append(0)
            continue
        thermal = hop_ratio(x,hop)
        steady = x/compact.REFERENCE_V
        truth = np.log10(i[ids]/(g*compact.REFERENCE_V))
        curves = thermal[None,:]+candidates[:,None]*(steady-thermal)[None,:]
        losses = np.mean((np.log10(curves)-truth[None,:])**2,axis=1)
        best = int(np.argmin(losses))
        fractions.append(float(candidates[best]))
        improvement.append(float(np.sqrt(losses[0])-np.sqrt(losses[best])))
        counts.append(len(x))
    return np.asarray(fractions),np.asarray(improvement),np.asarray(counts)


class RetainedRamp(compact.PositiveRamp):
    def __init__(self,wave,circuit,step,fraction):
        super().__init__(wave,circuit,step)
        self.fraction = np.asarray(fraction)

    def current(self,v,g,initial,hop):
        f = self.fraction[:,None] if v.ndim==2 else self.fraction
        retained = initial*f
        return ((initial-retained)*compact.REFERENCE_V*hop_ratio(v,hop)
                +(retained+np.maximum(g-initial,0))*v)


def scored(rows,circuit,curves):
    scores = compact.score(rows,curves,circuit)
    for name,mask in (("lower_state",np.array([ivread.terminal_estimate(r,False,.15)<=30.
                                             for _,r,_ in rows])),
                      ("higher_state",np.array([ivread.terminal_estimate(r,False,.15)>30.
                                              for _,r,_ in rows]))):
        if mask.any():
            subrows = [r for r,keep in zip(rows,mask) if keep]
            scores[name] = {"n":int(mask.sum()),"scores":compact.score(
                subrows,{k:v[mask] for k,v in curves.items()},circuit)}
    return scores


def plot(path,panels,baseline_label="Hopping background only",
         candidate_label="Retained links + hopping",
         title="SET outward | measured and predicted response"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(2,2,figsize=(10,7.1),layout="constrained")
    for ax,(label,rows,baseline,candidate) in zip(axes.flat,panels):
        wave = rows[0][1].voltage
        obs,mask,_ = compact.data_matrix(rows)
        use = mask & (np.arange(len(wave))<=wave.argmax())
        inds = np.flatnonzero(use.any(axis=0))
        band = np.nanquantile(np.where(use[:,inds],obs[:,inds],np.nan),[.1,.5,.9],axis=0)*1e6
        ax.fill_between(wave[inds],band[0],band[2],color=".84",label="Measured p10–p90")
        ax.plot(wave[inds],band[1],color=".13",label="Measured median")
        ax.plot(wave[inds],np.median(baseline[:,inds],axis=0)*1e6,"--",color="#486d93",
                label=baseline_label)
        ax.plot(wave[inds],np.median(candidate[:,inds],axis=0)*1e6,color="#9b2e3e",
                label=candidate_label)
        ax.axvspan(0,.55,color=".92",alpha=.35,label="Typical input region")
        ax.set(xlim=(0,3.5),yscale="log",title=f"{label} · {len(rows)} scans",
               xlabel="Applied SET voltage (V)",ylabel="Current magnitude (µA)")
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(path,dpi=175)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","revision","output","figure"):
        ap.add_argument("--"+name,type=Path,required=True)
    args = ap.parse_args()
    base = json.loads(args.revision.read_text())
    hop = base["hopping_slope_per_V"]
    frozen = base["fits"]["independent_links"]["log_rate_slope_log_cap_uS"]
    blocks = compact.get_cohorts(json.loads((args.cache/"index.json").read_text()),args.cache)
    # Require both the prior anchor and at least four other prefix samples;
    # both competing models use exactly the same accepted traces and masks.
    train,test = {},{}
    for gate,(all_rows,rows,sc) in blocks.items():
        _,_,n = initials(rows,sc,hop)
        usable = [r for r,s in zip(rows,n) if s>=4]
        cut = int(.7*len(all_rows)) if gate<=1.3 else 0
        if cut:
            train[gate] = (all_rows,[r for r in usable if r[0]<cut],sc)
        test[gate] = (all_rows,[r for r in usable if r[0]>=cut],sc)

    def factory(rows,sc,step):
        frac,_,_ = initials(rows,sc,hop)
        return RetainedRamp(rows[0][1].voltage,sc,step,frac)

    fit = compact.fit_growth(train,hop,False,ramp_factory=factory)
    results,panels = {},[]
    for gate,(all_rows,rows,sc) in test.items():
        v = rows[0][1].voltage
        actual,eligible,g = compact.data_matrix(rows,sc)
        frac,delta,n = initials(rows,sc,hop)
        old = compact.PositiveRamp(v,sc,.0125).predict(g,hop,frozen,False)[0]
        ramp = RetainedRamp(v,sc,.0125,frac)
        mix,_,_ = ramp.predict(g,hop,frozen,False)
        updated,_,state = ramp.predict(g,hop,fit.x,False)
        precise = RetainedRamp(v,sc,.00625,frac).predict(g,hop,fit.x,False)[0]
        assert np.all(np.diff(state[:,:ramp.apex+1],axis=1)>=-1e-12)
        # Both current laws match the measured equivalent 0.5 V initial conductance.
        assert np.allclose(ramp.current(np.full_like(g,compact.REFERENCE_V),g,g,hop),
                           g*compact.REFERENCE_V,rtol=1e-12,atol=0)
        results[str(gate)] = {
            "n":len(rows),"median_retained_fraction":float(np.median(frac)),
            "retained_fraction_p10_p50_p90":np.quantile(frac,[.1,.5,.9]).tolist(),
            "prefix_median_rmse_gain_decade":float(np.median(delta)),
            "scores":scored(rows,sc,{"hopping_only":old,"new_readout_only":mix,"new_readout_and_refit":updated}),
            "max_refinement_error_decade":float(np.max(np.abs(np.log10(updated[eligible]/precise[eligible]))))}
        if gate in (.9,1.4):
            panels.append((f"{gate:g} V gate",rows,old,updated))
        if gate==1.5:
            end = np.asarray([ivread.terminal_estimate(r,False,.15) for _,r,_ in rows])
            for label,keep in (("lower written conductance",end<=30),("higher written conductance",end>30)):
                ix = np.argmin(np.abs(v[:ramp.apex+1]-1.4))
                take = keep & eligible[:,ix]
                results[str(gate)][label] = {"retained_fraction_median":float(np.median(frac[keep])),
                    "at_terminal_1p4V_measured_uA":float(np.median(actual[take,ix])*1e6),
                    "at_terminal_1p4V_baseline_uA":float(np.median(old[take,ix])*1e6),
                    "at_terminal_1p4V_new_uA":float(np.median(updated[take,ix])*1e6)}
                panels.append((f"1.5 V gate · {label}",[r for r,k in zip(rows,keep) if k],
                               old[keep],updated[keep]))
    output = {"protocol":"Identical split as previous: train on first 70% of 0.9–1.3 V, hold out 1.4/1.5 V entirely from parameter fit. Use one common hopping slope from the previous training. For each scan infer nonnegative ohmic and hopping fractions from the measured SET prefix, anchored at intrinsic 0.5 V; the low-voltage input points and earlier points are excluded from outward scoring. Both candidates use the same masks and initial low-voltage total conductance. Compare changing readout alone against shared growth refit. Post-write subgroup labels are used only to report scores.",
              "growth_refit_log_rate_slope_log_cap_uS":fit.x.tolist(),"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    plot(args.figure,panels)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
