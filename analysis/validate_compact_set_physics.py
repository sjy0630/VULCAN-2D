"""Compare minimal field-driven conductance growth with the previous shared SET."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from analysis import audit_iv41_high_conductance as reads
from analysis import refit_iv41 as iv
from analysis import validate_field_log_soft as field
from analysis import validate_log_soft_cycle as cycle
from analysis import validate_hard_initial_state as history
from analysis import validate_set_outward as previous
from vulcan2d import validate_shared_device as shared

REFERENCE_V = .5
GATES = (.9,1.,1.1,1.2,1.3,1.4,1.5)
PLOT_GATES = (.9,1.1,1.4,1.5)


class PositiveRamp:
    """Solve the transistor load line at each step of one shared SET law."""

    def __init__(self,wave,circuit,step):
        self.wave = wave
        self.circuit = circuit
        self.apex = int(wave.argmax())
        self.grid = np.linspace(0,wave[self.apex],int(np.ceil(wave[self.apex]/step))+1)

    def current(self,v,g,initial,hop):
        return (initial*REFERENCE_V*np.sinh(hop*v)/np.sinh(hop*REFERENCE_V)
                + np.maximum(g-initial,0)*v)

    def predict(self,initial,hop,params,cooperative):
        log_k,slope,log_cap = params
        cap = np.exp(log_cap)*1e-6
        capacity = np.maximum(cap,initial*1.0001)
        g = initial.copy()
        trajectory = [g]
        logits = np.log(initial/(capacity-initial)) if cooperative else None
        vt,it = self.circuit.transistor

        def local(applied,g):
            lo,hi = np.zeros_like(initial),np.full_like(initial,applied)
            for _ in range(20):
                v = (lo+hi)/2
                below = self.current(v,g,initial,hop)<np.interp(applied-v,vt,it)
                lo,hi = np.where(below,v,lo),np.where(below,hi,v)
            return (lo+hi)/2

        def rate(v):
            return np.exp(log_k+slope*(v-1.5))

        for j,delta in enumerate(np.diff(self.grid)):
            start = self.grid[j]
            advance = rate(local(start,g))*delta/2
            mid = capacity*expit(logits+advance) if cooperative else capacity-(capacity-g)*np.exp(-advance)
            advance = rate(local(start+delta/2,mid))*delta
            if cooperative:
                logits = logits+advance
                g = capacity*expit(logits)
            else:
                g = capacity-(capacity-g)*np.exp(-advance)
            trajectory.append(g)
        states = np.asarray(trajectory).T
        mapped = np.array([np.interp(self.wave[:self.apex+1],self.grid,row) for row in states])
        mapped = np.c_[mapped,np.broadcast_to(states[:,-1,None],(len(initial),len(self.wave)-self.apex-1))]
        current,local_v = self.circuit.operating_point(
            np.broadcast_to(self.wave,mapped.shape),
            lambda v:self.current(v,mapped,initial[:,None],hop))
        return current,local_v,mapped


def branch_windows(wave):
    outward = np.arange(len(wave))<=wave.argmax()
    return {"outward_after_prefix":outward&(wave>=.55),
            "low_outward":outward&(wave>=.55)&(wave<=.80),
            "transition_outward":outward&(wave>=.8)&(wave<=2),
            "transition_1p2_to_1p8":outward&(wave>=1.2)&(wave<=1.8),
            "high_outward":outward&(wave>=3),
            "return":~outward}


def get_cohorts(index, cache):
    blocks = {}
    for gate in GATES:
        rows,_ = iv.load_cohort(index,cache,round((gate-.9)*10)+5,gate)
        sc,_ = iv.circuits(gate)
        all_rows = rows["SET"]
        usable_rows = []
        for idx,row in enumerate(all_rows):
            g = reads.local_estimate(row,sc,True,REFERENCE_V)
            if g is not None:
                usable_rows.append((idx,row,g*1e-6))
        blocks[gate] = (all_rows,usable_rows,sc)
    return blocks


def fit_hop(training):
    xs,ys,weights = [],[],[]
    for gate,(_,rows,sc) in training.items():
        for _,row,g in rows:
            ids,v,i = reads.reader.intrinsic(row,sc,True)
            ids = ids[(v[ids]>=.18)&(v[ids]<=.8)&(abs(v[ids]-REFERENCE_V)>.025)]
            if len(ids):
                xs.extend(v[ids])
                ys.extend(np.log10(i[ids]/(REFERENCE_V*g)))
                weights.extend(np.full(len(ids),1/np.sqrt(len(rows)*len(ids))))
    x,y,w = map(np.asarray,(xs,ys,weights))
    p = least_squares(lambda b:w*(np.log10(np.sinh(b[0]*x)/np.sinh(b[0]*REFERENCE_V))-y),
                      [6.],bounds=([.1],[16.]))
    if not p.success:
        raise RuntimeError(p.message)
    return float(p.x[0])


def prefix_end(row,circuit):
    ids,local,_ = reads.reader.intrinsic(row,circuit,True)
    for a,b in zip(ids[:-1],ids[1:]):
        if b == a+1 and local[a] != local[b] and min(local[a],local[b]) <= REFERENCE_V <= max(local[a],local[b]):
            return b
    raise ValueError("Missing anchor source points")


def data_matrix(rows,circuit=None):
    masks = np.stack([r.masks["use_unclamped"] for _,r,_ in rows])
    if circuit is not None:
        for mask,(_,row,_) in zip(masks,rows):
            mask[:prefix_end(row,circuit)+1] = False
    return (np.abs(np.stack([r.current for _,r,_ in rows])),masks,
            np.asarray([g for _,_,g in rows]))


def fit_growth(training,hop,cooperative,ramp_factory=None):
    samples = []
    for gate,(_,all_rows,circuit) in training.items():
        ids = np.linspace(0,len(all_rows)-1,min(len(all_rows),11),dtype=int)
        rows = [all_rows[i] for i in ids]
        ramp = (PositiveRamp(rows[0][1].voltage,circuit,.065) if ramp_factory is None
                else ramp_factory(rows,circuit,.065))
        obs,mask,g = data_matrix(rows,circuit)
        regions = branch_windows(ramp.wave)
        segments = []
        for key,weight in (("low_outward",1),("transition_outward",2),
                           ("transition_1p2_to_1p8",1),("high_outward",.75),("return",.35)):
            kept = mask & regions[key]
            if np.any(kept):
                segments.append((kept,np.log10(obs[kept]),np.sqrt(weight/kept.sum())))
        samples.append((ramp,g,segments))

    def residual(p):
        chunks = []
        for ramp,g,segments in samples:
            pred,_,_ = ramp.predict(g,hop,p,cooperative)
            for mask,obs,w in segments:
                chunks.append(w*(np.log10(pred[mask])-obs))
        return np.concatenate(chunks)

    fits = [least_squares(residual,p, bounds=([-14.,0.,np.log(5.)],[7.,35.,np.log(5000.)]),
                         diff_step=1e-3,ftol=2e-4,max_nfev=55)
            for p in ((-6.,10.,np.log(110)),(-3.,20.,np.log(250)))]
    finite = [p for p in fits if p.success]
    if not finite:
        raise RuntimeError("SET calibration failed to converge")
    return min(finite,key=lambda p:p.cost)


def score(rows,curves,circuit):
    wave = rows[0][1].voltage
    obs,mask,_ = data_matrix(rows,circuit)
    result = {}
    for key,part in branch_windows(wave).items():
        use = mask&part
        result[key] = {"points":int(use.sum()),
                       **{name:float(np.sqrt(np.mean(np.log10(i[use]/obs[use])**2)))
                          for name,i in curves.items() if key!="return" or name!="previous_with_same_prefix"}}
    return result


def draw(path,panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(10,7.1),layout="constrained")
    for ax,(gate,rows,old,new) in zip(axes.flat,panels):
        wave = rows[0][1].voltage
        obs,mask,_ = data_matrix(rows)
        use = mask & (np.arange(len(wave))<=wave.argmax())
        idx = np.flatnonzero(use.any(axis=0))
        lo,med,hi = np.nanquantile(np.where(use[:,idx],obs[:,idx],np.nan),[.1,.5,.9],axis=0)*1e6
        qlo,qmed,qhi = np.quantile(new[:,idx],[.1,.5,.9],axis=0)*1e6
        ax.fill_between(wave[idx],lo,hi,color=".84",label="Measured p10–p90")
        ax.fill_between(wave[idx],qlo,qhi,color="#eab3bc",alpha=.65,label="Candidate p10–p90")
        ax.plot(wave[idx],med,color=".13",label="Measured median")
        ax.plot(wave[idx],old[idx]*1e6,"--",color="#486d93",label="Previous SET")
        ax.plot(wave[idx],qmed,color="#a1263d",label="Candidate median")
        ax.axvspan(0,.55,color=".92",alpha=.35,label="Low-voltage input window")
        ax.set(title=f"{gate:g} V gate · {len(rows)} scans",yscale="log",
               xlabel="Applied SET voltage (V)",ylabel="Current magnitude (µA)")
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncols=2)
    fig.suptitle("SET outward | shared device parameters, per-scan low-bias input")
    fig.savefig(path,dpi=175)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("cache","revisions","output","figure"):
        ap.add_argument("--"+key,type=Path,required=True)
    args = ap.parse_args()
    idx = json.loads((args.cache/"index.json").read_text())
    blocks = get_cohorts(idx,args.cache)
    tr,te = {},{}
    for gate,(full,rows,sc) in blocks.items():
        cut = int(.7*len(full)) if gate<=1.3 else 0
        if cut:
            tr[gate] = (full,[r for r in rows if r[0]<cut],sc)
        te[gate] = (full,[r for r in rows if r[0]>=cut],sc)
    hop = fit_hop(tr)
    variants = {"independent_links":False,"connection_assisted_growth":True}
    models = {k:fit_growth(tr,hop,v) for k,v in variants.items()}
    selected = min(models,key=lambda k:models[k].cost)
    p0 = json.loads((args.revisions/"field_log_soft_validation.json").read_text())["field_model"]
    prior = json.loads((args.revisions/"set_outward_refinement.json").read_text())
    form = prior["selected"]
    shape = prior["fits"][form]
    nominal = cycle.data()[1.1][:32]
    device,prep = shared.fit_soft_set([r[0] for r in nominal],nominal[0][2])
    initial,ratio = p0["initial_log_conductance"],p0["readout_positive_negative_ratio"]

    def prior_current(v,s):
        return previous.currents(v,s,initial,ratio,device.soft_exponent,
                                 shape["background_parameters"],form)

    results,panels = {},[]
    for gate,(full,rows,sc) in te.items():
        wave = rows[0][1].voltage
        _,old,_ = field.SetRamp([(rows[0][1],None,sc,None)],device,prep,initial,ratio,
                                .0125,current_law=prior_current).predict(shape["growth_parameters"])
        _,mask,g = data_matrix(rows,sc)
        ramp = PositiveRamp(wave,sc,.0125)
        preds = {"previous":np.broadcast_to(old,mask.shape)}
        for key,p in models.items():
            preds[key],_,states = ramp.predict(g,hop,p.x,variants[key])
            assert np.all(np.diff(states[:,:ramp.apex+1],axis=1)>=-1e-12)
        scale = (g*REFERENCE_V / previous.background(
            REFERENCE_V,shape["background_parameters"],form))
        old_ramp = field.SetRamp([(rows[0][1],None,sc,None)],device,prep,initial,ratio,.0125)
        path,local = history.simulate_conditioned(old_ramp,device,initial,ratio,
            shape["background_parameters"],form,shape["growth_parameters"],scale,.0125)
        outward = wave[:old_ramp.apex+1]
        device_v = np.array([np.interp(outward,path,row) for row in local])
        fair = np.full(mask.shape,np.nan)
        fair[:,:len(outward)] = sc.transistor_current(outward-device_v)
        preds["previous_with_same_prefix"] = fair
        refined,_,_ = PositiveRamp(wave,sc,.00625).predict(g,hop,models[selected].x,variants[selected])
        results[str(gate)] = {"scans":len(rows),"scores":score(rows,preds,sc),
             "max_refinement_error_decade":float(np.max(np.abs(np.log10(
                 preds[selected][mask]/refined[mask]))))}
        if gate in PLOT_GATES:
            panels.append((gate,rows,old,preds[selected]))
    output = {"protocol":"Fixed common hopping law from training-only early local-voltage points; initial conductance is informed by each measured SET prefix at local 0.5 V. Fit first 70% of scans at 0.9–1.3 V; leave late lower-gate scans and all 1.4/1.5 V out. Outward scoring starts at terminal 0.55 V and additionally excludes all samples through the two measured points used to infer the input conductance for each scan. Field-activated conductance growth integrated against voltage path, not time. All model parameters shared. Use close-to-ohmic current for newly formed connections. In addition to the previous autonomous model, outward scores include a same-prefix comparator using the previous readout and log-state growth with its background scaled by the same input.",
              "selected":selected,"hopping_slope_per_V":hop,
              "fits":{key:{"log_rate_slope_log_cap_uS":p.x.tolist(),"cost":float(p.cost)} for key,p in models.items()},
              "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    draw(args.figure,panels)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
