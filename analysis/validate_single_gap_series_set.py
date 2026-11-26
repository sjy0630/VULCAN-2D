"""Test one evolving gap in series with a fixed internal resistance."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_iv41_high_conductance as reads
from analysis import validate_compact_set_physics as compact
from analysis import validate_set_current_bottleneck as limit
from analysis import validate_set_retained_links as retained


class GapRamp:
    """The state z increases transmission amplitude of the same nonlinear gap."""

    def __init__(self,terminal,circuit,step):
        self.terminal = terminal
        self.circuit = circuit
        self.apex = int(terminal.argmax())
        self.grid = np.linspace(0.,terminal[self.apex],int(np.ceil(terminal[self.apex]/step))+1)
        if circuit.transistor is not None:
            grid, currents = circuit.transistor
            self.levels,indices = np.unique(currents,return_index=True)
            self.transistor_voltage = grid[indices]

    def transistor_drop(self,current):
        if self.circuit.transistor is None:
            return np.zeros_like(current)
        return np.interp(current,self.levels,self.transistor_voltage)

    def predict(self,initial,parameters, *, iterations=28, evolve_state=True):
        beta,log_resistance,log_rate,sensitivity,capacity,*state_coupling = parameters
        coupling = state_coupling[0] if state_coupling else 0.
        initial_log_conductance = np.log(initial/1e-7)
        resistance = np.exp(log_resistance)
        reference_current = initial*compact.REFERENCE_V
        reference_gap = compact.REFERENCE_V-resistance*reference_current
        if np.any(reference_gap<=0):
            raise ValueError("Series resistance is incompatible with the low-bias reference")
        amplitude = reference_current/np.sinh(beta*reference_gap)

        def solve(applied,state,ampl):
            amp = ampl*np.exp(state)
            hi = np.minimum(self.circuit.supply_bound(applied),
                            np.minimum(applied/resistance,amp*np.sinh(beta*applied)))
            lo = np.zeros_like(hi)
            for _ in range(iterations):
                i = .5*(lo+hi)
                voltage = (np.arcsinh(i/amp)/beta+resistance*i
                           +self.transistor_drop(i))
                below = voltage<applied
                lo,hi = np.where(below,i,lo),np.where(below,hi,i)
            i = .5*(lo+hi)
            gap = np.arcsinh(i/amp)/beta
            return i,gap

        if not evolve_state:
            state = np.zeros((len(initial),len(self.terminal)))
            current,gap = solve(self.terminal,state,amplitude[:,None])
            return current,gap,state

        state = np.zeros_like(initial)
        trace = [state]
        for j,step in enumerate(np.diff(self.grid)):
            a = self.grid[j]
            _,v = solve(np.asarray(a),state,amplitude)
            r = np.exp(log_rate+sensitivity*(v-1.)+coupling*(initial_log_conductance+state))
            mid = state+(capacity-state)*(-np.expm1(-r*step/2))
            _,v = solve(np.asarray(a+step/2),mid,amplitude)
            r = np.exp(log_rate+sensitivity*(v-1.)+coupling*(initial_log_conductance+mid))
            state = state+(capacity-state)*(-np.expm1(-r*step))
            trace.append(state)
        trace = np.asarray(trace).T
        mapped = np.array([np.interp(self.terminal[:self.apex+1],self.grid,z) for z in trace])
        mapped = np.c_[mapped,np.broadcast_to(trace[:,-1,None],(len(initial),len(self.terminal)-self.apex-1))]
        current,gap = solve(self.terminal,mapped,amplitude[:,None])
        return current,gap,mapped


def calibrate(training):
    samples = []
    largest_initial = max(r[2] for _,rows,_ in training.values() for r in rows)
    max_resistance = min(2.5e4,.8/largest_initial)
    for _,rows,circuit in training.values():
        indices = np.linspace(0,len(rows)-1,min(len(rows),11),dtype=int)
        items = [rows[i] for i in indices]
        ramp = GapRamp(items[0][1].voltage,circuit,.065)
        actual,usable,initial = compact.data_matrix(items,circuit)
        windows = compact.branch_windows(ramp.terminal)
        targets = []
        for key,weight in (("low_outward",1),("transition_outward",2),
                           ("transition_1p2_to_1p8",1),("high_outward",.75),("return",.35)):
            keep = usable&windows[key]
            targets.append((keep,np.log10(actual[keep]),np.sqrt(weight/keep.sum())))
        samples.append((ramp,initial,targets))

    def residual(p):
        parts = []
        for ramp,initial,targets in samples:
            current,_,_ = ramp.predict(initial,p,iterations=25)
            for keep,actual,w in targets:
                parts.append(w*(np.log10(current[keep])-actual))
        return np.concatenate(parts)

    bounds = ([.5,np.log(30.),-12.,0.,.1],
              [18.,np.log(max_resistance),8.,35.,24.])
    starts = ([6.,np.log(3500.),0.,8.,7.],[9.,np.log(12000.),-2.,15.,12.])
    fits = []
    for start in starts:
        start = np.clip(start,np.array(bounds[0])+1e-5,np.array(bounds[1])-1e-5)
        fits.append(least_squares(residual,start,bounds=bounds,diff_step=2e-3,
                                  ftol=3e-4,max_nfev=75))
    candidates=[f for f in fits if f.success]
    if not candidates:
        raise RuntimeError("No converged training fit")
    return min(candidates,key=lambda f:f.cost),bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache","baseline","output","figure"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    saved = json.loads(args.baseline.read_text())
    alpha = saved["tested_nonzero_alpha"]
    baseline_params = saved["fits"][str(alpha)]["log_rate_slope_log_cap_uS"]
    blocks=compact.get_cohorts(json.loads((args.cache/"index.json").read_text()),args.cache)
    training,test = {},{}
    for gate,(all_rows,rows,sc) in blocks.items():
        valid = np.isfinite(limit.estimate_slopes(rows,sc,0.))
        rows = [r for r,k in zip(rows,valid) if k]
        cut = int(.7*len(all_rows)) if gate<=1.3 else 0
        if cut:
            training[gate] = (all_rows,[r for r in rows if r[0]<cut],sc)
        test[gate] = (all_rows,[r for r in rows if r[0]>=cut],sc)

    fit,bounds=calibrate(training)
    results,panels = {},[]
    beta,log_R,_,_,_ = fit.x
    R = np.exp(log_R)
    for gate,(_,rows,sc) in test.items():
        v=rows[0][1].voltage
        measured,mask,initial=compact.data_matrix(rows,sc)
        r=GapRamp(v,sc,.0125)
        refined=GapRamp(v,sc,.00625)
        current,gap,z=r.predict(initial,fit.x)
        tight=refined.predict(initial,fit.x)[0]
        slopes=limit.estimate_slopes(rows,sc,alpha)
        baseline=limit.LimitedRamp(v,sc,.0125,slopes,alpha).predict(initial,1.,baseline_params,False)[0]
        assert np.all(np.diff(z[:,:r.apex+1],axis=1)>=-1e-12)
        fitted_voltage = gap+R*current+np.interp(current,r.levels,r.transistor_voltage)
        result = {"n":len(rows),
                  "scores":retained.scored(rows,sc,{"finite_supply":baseline,"single_gap":current}),
                  "max_voltage_balance_error_V":float(np.max(np.abs((fitted_voltage-v)[mask]))),
                  "max_refinement_change_decade":float(np.max(np.abs(np.log10(current[mask]/tight[mask]))))}
        end=np.asarray([reads.terminal_estimate(row,False,.15) for _,row,_ in rows])
        if gate in (.9,1.4):
            panels.append((f"{gate:g} V gate",rows,baseline,current))
        if gate==1.5:
            static,_vg,_z = r.predict(initial,fit.x,evolve_state=False)
            for label,keep in (("lower written conductance",end<=30.),("higher written conductance",end>30.)):
                at=int(np.argmin(np.abs(v[:r.apex+1]-1.4)))
                use=keep&mask[:,at]
                result[label]={"scans":int(keep.sum()),
                    **{f"at_1p4V_{name}_uA":float(np.median(y[use,at])*1e6)
                       for name,y in (("observed",measured),("finite_supply",baseline),("single_gap",current),
                                      ("single_gap_no_growth",static))}}
                panels.append((f"1.5 V gate · {label}",[r for r,k in zip(rows,keep) if k],baseline[keep],current[keep]))
        results[str(gate)] = result
    output={"protocol":"Same held-out scans and masks as the finite-supply comparator. First 70% from gates 0.9–1.3 V used to fit shared parameters. Neither 1.4 nor 1.5 V used for coefficient fitting. Single-gap model conditions only on the observed current at internal 0.5 V; the finite-supply comparator also uses low-bias shape. The gap current is A0 exp(z) sinh(beta Vgap); the same current flows through a fixed series resistance and the transistor. The gap conductance state evolves with actual gap voltage per increment of the applied voltage path, and freezes on return for this SET test. State and resistance are effective quantities. No RESET or stochastic high-state formation fitted here.",
            "fit":{"beta_per_V":float(beta),"resistance_ohm":float(R),
                   "log_growth_rate_field_slope_capacity":fit.x[2:].tolist(),
                   "training_cost":float(fit.cost),"bounds":[list(b) for b in bounds]},
            "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    retained.plot(args.figure,panels,baseline_label="Finite-supply comparator",
                  candidate_label="One evolving gap + series resistance",
                  title="SET outward | single-gap physical model")
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
