"""Compare field-assisted connection loss with the previous RESET rule."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_log_soft as readout
from analysis import validate_entry_anchors as entry
from analysis import validate_log_soft_cycle as cycle


STATE_GRID = np.linspace(-7., 4., 441)
PIVOT_V = .5


class ResetRamp:
    def __init__(self, rows, anchors, shapes, step, current_law=cycle.current_law, *,
                 evolve_return=False):
        _, first, _, self.circuit = rows[0]
        self.wave = np.abs(first.voltage)
        if any(not np.array_equal(np.abs(reset.voltage), self.wave) for _,reset,_,_ in rows):
            raise ValueError("ResetRamp requires matching waveforms")
        self.apex = int(np.argmax(self.wave))
        peak = self.wave[self.apex]
        up = np.linspace(0., peak, int(np.ceil(peak/step))+1)
        self.evolve_return = evolve_return
        if evolve_return:
            ending = self.wave[-1]
            down = np.linspace(peak, ending, int(np.ceil((peak-ending)/step))+1)
            self.grid = np.r_[up, down[1:]]
        else:
            self.grid = up
        self.anchors, self.shapes = anchors, shapes
        self.current_law = current_law
        applied = np.broadcast_to(self.grid[:,None], (len(self.grid),len(STATE_GRID)))
        _,self.internal = self.circuit.operating_point(
            applied,lambda v:self.current_law(v,STATE_GRID,anchors,shapes))

    def predict(self, initial, floor, params, *, rate_law=None, return_trajectory=False):
        """Solve d(G-Gbg)/d|Vapp|=-k(Vh)(G-Gbg), k(Vh)=kref exp(beta*(Vh-Vref))."""
        initial = np.asarray(initial)
        background = 10**floor
        residual = np.maximum(10**initial-background,0.)
        baseline = np.minimum(10**initial,background)
        trajectory = np.empty((len(initial),len(self.grid)))
        trajectory[:,0] = initial
        if rate_law is None:
            log_rate, beta = params
            def rate_law(local):
                return np.exp(log_rate + beta*(local-PIVOT_V))
        for j, delta in enumerate(np.abs(np.diff(self.grid))):
            state = trajectory[:,j]
            local = np.interp(state,STATE_GRID,self.internal[j])
            exponent = rate_law(local)
            mid = np.log10(baseline+residual*np.exp(-.5*delta*exponent))
            local = .5*(np.interp(mid,STATE_GRID,self.internal[j])
                        + np.interp(mid,STATE_GRID,self.internal[j+1]))
            exponent = rate_law(local)
            residual *= np.exp(-delta*exponent)
            trajectory[:,j+1] = np.log10(baseline+residual)
        if self.evolve_return:
            path = np.r_[0., np.cumsum(np.abs(np.diff(self.grid)))]
            samples = np.r_[0., np.cumsum(np.abs(np.diff(self.wave)))]
            states = np.array([np.interp(samples,path,row) for row in trajectory])
        else:
            outward = np.array([np.interp(self.wave[:self.apex+1],self.grid,row)
                                for row in trajectory])
            states = np.c_[outward,np.broadcast_to(trajectory[:,-1,None],
                                                  (len(initial),len(self.wave)-self.apex-1))]
        current,_ = self.circuit.operating_point(
            np.broadcast_to(self.wave,states.shape),
            lambda v:self.current_law(v,states,self.anchors,self.shapes))
        return current, states if return_trajectory else states[:,-1]


def measured_end(row, circuit):
    ref = readout.reference_current(*readout.intrinsic(row,circuit,outward=False))
    return None if ref is None else np.log10(ref/readout.REFERENCE_V/1e-6)


def make_figure(path, panels, old_label="Previous RESET", new_label="Activated RESET"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(9,9), layout="constrained")
    for row, (gate, reset_rows, old, new) in enumerate(panels):
        x = np.abs(reset_rows[0].voltage)
        peak = x.argmax()
        data = np.stack([np.abs(r.current) for r in reset_rows])
        keep = np.stack([r.masks["use_unclamped"] for r in reset_rows])
        for ax, part, label in zip(axes[row],(np.arange(len(x))<=peak,np.arange(len(x))>=peak),
                                   ("outward sweep","return sweep")):
            ids = np.flatnonzero(part & keep.any(axis=0))
            values = np.where(keep[:,ids],data[:,ids],np.nan)
            bottom,median,top = np.nanquantile(values,[.1,.5,.9],axis=0)
            ax.fill_between(x[ids],bottom*1e6,top*1e6,color="0.8",alpha=.6,label="Measured p10–p90")
            ax.plot(x[ids],median*1e6,color="0.16",label="Measured median")
            ax.plot(x[ids],old[ids]*1e6,color="#486d93",ls="--",label=old_label)
            ax.plot(x[ids],new[ids]*1e6,color="#a12f3e",label=new_label)
            ax.set(yscale="log",title=f"{gate:g} V gate | {label}",
                   xlabel="External reverse voltage magnitude (V)",ylabel="Current magnitude (µA)")
            ax.grid(alpha=.22)
    axes[0][0].legend(loc="best",fontsize=8)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figure",type=Path,required=True)
    parser.add_argument("--set-results",type=Path,required=True)
    args = parser.parse_args()
    set_results = json.loads(args.set_results.read_text())["results"]
    pairs = cycle.data()
    training = {.9:pairs[.9][:70],1.1:pairs[1.1][:32]}
    test = {.9:pairs[.9][70:],1.1:pairs[1.1][32:],1.3:pairs[1.3]}
    train = training[.9]+training[1.1]
    low = readout.learn_anchors([readout.readout_record(r,rc) for _,r,_,rc in train])
    entries = [rec for _,r,_,rc in train if (rec:=entry.early_record(r,rc)) is not None]
    anchors, shapes, _ = entry.add_entry_anchors(entries,*low)
    previous = cycle.fit_erase(train,anchors,shapes)
    ends = [s for _,r,_,rc in train if (s:=measured_end(r,rc)) is not None]
    floor = float(np.quantile(ends,.10))

    fit_groups = []
    for rows in training.values():
        records = [(r,s) for _,r,_,rc in rows if (s:=cycle.observed_state(r,rc)) is not None]
        starts = np.array([s for r,s in records])
        ramp = ResetRamp(rows,anchors,shapes,.02)
        masks = np.stack([r.masks["use_unclamped"] for r,s in records])
        target = np.log10(np.abs(np.stack([r.current for r,s in records])[masks]))
        fit_groups.append((ramp,starts,masks,target))

    def residual(params):
        pieces = []
        for ramp,starts,masks,observed in fit_groups:
            current,_ = ramp.predict(starts,floor,params)
            pieces.append((np.log10(current[masks])-observed)/np.sqrt(len(observed)))
        return np.concatenate(pieces)

    fits = [least_squares(residual,start,bounds=([-8.,0.],[9.,30.]),
                         diff_step=1e-4,ftol=1e-5,max_nfev=120)
            for start in ([0.,5.],[4.,12.])]
    best = min((fit for fit in fits if fit.success),key=lambda fit:fit.cost)
    report = {}
    panels = []
    for gate,rows in test.items():
        start = float(np.log10(set_results[str(gate)]["entry_predicted_uS"]))
        ramp = ResetRamp(rows,anchors,shapes,.01)
        current,last = ramp.predict([start],floor,best.x)
        reference,last_refined = ResetRamp(rows,anchors,shapes,.005).predict([start],floor,best.x)
        reset_rows = [r for _,r,_,_ in rows]
        old = cycle.reset_predict(reset_rows[0],ramp.circuit,start,previous,anchors,shapes)
        masks = np.stack([r.masks["use_unclamped"] for r in reset_rows])
        measured = np.stack([r.current for r in reset_rows])
        scores = {}
        for label,branch in (("full",np.ones(len(ramp.wave),dtype=bool)),
                             ("outward",np.arange(len(ramp.wave))<=ramp.apex),
                             ("return",np.arange(len(ramp.wave))>ramp.apex)):
            mask = masks & branch
            scores[label] = {name:readout.rmse(np.log10(
                np.broadcast_to(trace,measured.shape)[mask]/np.abs(measured[mask])))
                             for name,trace in (("previous",old),("activated",current[0]))}
        obs = [s for _,r,_,rc in rows if (s:=measured_end(r,rc)) is not None]
        valid = masks.any(axis=0)
        report[str(gate)] = {"cycles":len(rows),"entry_uS":10**start,
                             "predicted_end_uS":float(10**last[0]),
                             "measured_end_median_uS":float(10**np.median(obs)),
                             "rmse_decade":scores,
                             "refinement_max_current_diff_decade":float(np.max(np.abs(
                                 np.log10(reference[0,valid]/current[0,valid])))),
                             "refinement_end_diff_decade":float(abs(last_refined[0]-last[0]))}
        panels.append((gate,reset_rows,old,current[0]))
    output = {"protocol":"Freeze SET estimates and readout curves. Fit a shared RESET law on the first 70 cycles at 0.9 V and cycles 4–35 at 1.1 V, using measured entry in training. Score against the same points as the previous RESET law with no target current used as input. Integration uses external voltage path, not measured time. Return retains the apex state. Background fixed to the 10th percentile of training RESET-end observations.",
              "background_uS":10**floor,"background_training_samples":len(ends),
              "rate_at_local_0p5V_per_V":float(np.exp(best.x[0])),
              "log_rate_slope_per_V":float(best.x[1]),"results":report}
    for path in (args.output,args.figure):
        path.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    make_figure(args.figure,panels)
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
