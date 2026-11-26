"""Test a shared, sequential SET law for the logarithmic conductance state."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_log_soft as reads
from analysis import validate_entry_anchors as entry
from analysis import validate_log_soft_cycle as cycle
from vulcan2d import model_shared_device as model
from vulcan2d import validate_shared_device as shared


STATE_GRID = np.linspace(-6., 3., 451)
REF = reads.REFERENCE_V


def positive_current(voltage, state, device, prep, initial, ratio):
    background = prep.off_set_a * voltage * np.exp(device.beta_set * np.sqrt(voltage))
    excess = np.maximum(10**state - 10**initial, 0.)
    return background + (ratio * REF * 1e-6 * excess
                         * (voltage/REF)**device.soft_exponent)


class SetRamp:
    def __init__(self, rows, device, prep, initial, ratio, step, *, current_law=None):
        set_row, _, self.circuit, _ = rows[0]
        self.wave = np.abs(set_row.voltage)
        if any(not np.array_equal(np.abs(s.voltage), self.wave) for s,_,_,_ in rows):
            raise ValueError("Expected a common SET waveform at each gate")
        self.apex = int(self.wave.argmax())
        self.grid = np.linspace(0., self.wave[self.apex], int(np.ceil(self.wave[self.apex]/step))+1)
        self.initial = initial
        if current_law is None:
            current_law = lambda v,s: positive_current(v,s,device,prep,initial,ratio)
        self.law = current_law
        _, self.local = self.circuit.operating_point(
            np.broadcast_to(self.grid[:,None], (len(self.grid),len(STATE_GRID))),
            lambda v: self.law(v,STATE_GRID))
        self.ds = np.diff(self.grid)

    def predict(self, parameters):
        """Integrate ds/dVapp=k[max(Vh-Vth,0)]²(smax-s), with midpoint updates."""
        log_k, threshold, upper = parameters
        states = np.empty(len(self.grid))
        states[0] = s = self.initial
        k = np.exp(log_k)
        for j, step in enumerate(self.ds):
            field = np.interp(s, STATE_GRID, self.local[j])
            speed = k * max(field-threshold,0.)**2
            half = upper - (upper-s)*np.exp(-speed*step/2)
            field = .5*(np.interp(half,STATE_GRID,self.local[j])
                        + np.interp(half,STATE_GRID,self.local[j+1]))
            speed = k * max(field-threshold,0.)**2
            states[j+1] = s = upper - (upper-s)*np.exp(-speed*step)
        values = np.r_[np.interp(self.wave[:self.apex+1],self.grid,states),
                       np.full(len(self.wave)-self.apex-1,s)]
        current, local = self.circuit.operating_point(self.wave,lambda v:self.law(v,values))
        return float(s),current,local


def positive_ratio(rows):
    values = []
    for set_row, reset, sc, rc in rows:
        state = cycle.observed_state(reset,rc)
        ref = reads.reference_current(*reads.intrinsic(set_row,sc,outward=False))
        if state is not None and ref is not None:
            values.append(np.log10(ref/REF/1e-6)-state)
    if not values:
        raise ValueError("Could not estimate common positive-to-negative readout ratio")
    return float(10**np.median(values)),len(values)


def entry_targets(rows):
    return np.array([s for _,r,_,rc in rows if (s:=cycle.observed_state(r,rc)) is not None])


def set_metrics(rows, prediction):
    return reads.rmse(np.concatenate([
        cycle.log_errors(s,prediction,s.masks["use_unclamped"]) for s,_,_,_ in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    pairs = cycle.data()
    training = {0.9:pairs[.9][:70],1.1:pairs[1.1][:32]}
    tests = {0.9:pairs[.9][70:],1.1:pairs[1.1][32:],1.3:pairs[1.3]}
    train = training[.9]+training[1.1]
    nominal = training[1.1]
    device, prep = shared.fit_soft_set([s for s,_,_,_ in nominal],nominal[0][2])
    set_end = model.simulate_sweep(nominal[0][0].voltage,"SET",device,prep,nominal[0][2])["final"]
    device, prep = shared.fit_soft_reset([r for _,r,_,_ in nominal],nominal[0][3],device,prep,set_end)
    records = [rec for _,r,_,rc in train if (rec:=reads.readout_record(r,rc)) is not None]
    initial = float(np.median([rec[0] for rec in records]))
    ratio,n_ratio = positive_ratio(train)
    low = reads.learn_anchors(records)
    high_records = [rec for _,r,_,rc in train if (rec:=entry.early_record(r,rc)) is not None]
    anchors,shapes,_ = entry.add_entry_anchors(high_records,*low)
    eraser = cycle.fit_erase(train,anchors,shapes)
    ramps = {g:SetRamp(rows,device,prep,initial,ratio,.025) for g,rows in training.items()}
    targets = {g:float(np.median(entry_targets(rows))) for g,rows in training.items()}
    training_points = {g:shared._training_data([s for s,_,_,_ in rows],use_transition=False)
                       for g,rows in training.items()}

    def residual(x):
        chunks = []
        for g,ramp in ramps.items():
            state, current, _ = ramp.predict(x)
            _, mask, logged, weights = training_points[g]
            log_current = np.log10(np.maximum(np.broadcast_to(current,mask.shape)[mask],1e-18))
            diff = log_current-logged
            weights = weights/np.linalg.norm(weights)
            chunks.extend((weights*diff,np.array([state-targets[g]])))
        return np.concatenate(chunks)

    fits = [least_squares(residual,x,bounds=([-6.,0.,.65],[4.,2.20,3.]),
                         diff_step=1e-4,ftol=1e-5,max_nfev=160)
            for x in ([-1.,1.5,1.5],[-3.,.5,2.5])]
    best = min((fit for fit in fits if fit.success),key=lambda fit:fit.cost)
    hbn = best.x
    scores = {}
    for gate,rows in tests.items():
        ramp = SetRamp(rows,device,prep,initial,ratio,.0125)
        state,predicted_set,_ = ramp.predict(hbn)
        refined,_,_ = SetRamp(rows,device,prep,initial,ratio,.00625).predict(hbn)
        old = model.simulate_sweep(rows[0][0].voltage,"SET",device,prep,rows[0][2])
        original_reset, new_reset = [],[]
        for _,r,_,rc in rows:
            original = model.simulate_sweep(r.voltage,"RESET",device,prep,rc,old["final"])
            new = cycle.reset_predict(r,rc,state,eraser,anchors,shapes)
            mask = r.masks["use_unclamped"]
            original_reset.extend(cycle.log_errors(r,original["current_A"],mask))
            new_reset.extend(cycle.log_errors(r,new,mask))
        observed = entry_targets(rows)
        scores[str(gate)] = {
            "cycles":len(rows),
            "entry_observed_median_uS":float(10**np.median(observed)),
            "entry_predicted_uS":10**state,
            "entry_rmse_decade":reads.rmse(state-observed),
            "set_original_rmse_decade":set_metrics(rows,old["current_A"]),
            "set_field_law_rmse_decade":set_metrics(rows,predicted_set),
            "reset_original_rmse_decade":reads.rmse(original_reset),
            "reset_field_law_rmse_decade":reads.rmse(new_reset),
            "entry_refinement_diff_decade":abs(refined-state)}
    # Development-only sensitivity check: an active constraint cannot identify
    # a threshold or saturation level independently of the chosen bounds.
    wider_fits = [least_squares(residual,start,bounds=([-8.,0.,.65],[5.,2.8,4.]),
                               diff_step=1e-4,ftol=1e-5,max_nfev=160)
                  for start in (hbn,[-2.,1.,2.])]
    wider = min((fit for fit in wider_fits if fit.success),key=lambda fit:fit.cost)
    wider_entry = {str(g):10**SetRamp(rows,device,prep,initial,ratio,.0125).predict(wider.x)[0]
                   for g,rows in tests.items()}
    report = {
        "protocol":"Shared sequential SET. Fit the first 70 cycles at 0.9 V and cycles 4–35 at 1.1 V. Only transistor tables differ by gate. Starting log state is the median training RESET return state and is shared across cycles. This is an assumed initialization, not a per-cycle measured input. All of 1.3 V is held out from this fit. Write path uses Vapp increments, no physical time calibration. RESET solver and anchors follow the previous expanded readout model. Hard path excluded in this candidate.",
        "field_model":{"initial_log_conductance":initial,"readout_positive_negative_ratio":ratio,
                       "ratio_sample_count":n_ratio,"rate_per_V3":float(np.exp(hbn[0])),
                       "threshold_local_v":float(hbn[1]),"maximum_log_conductance":float(hbn[2]),
                       "parameter_fit_bounds":[[-6.,0.,.65],[4.,2.20,3.]],
                       "training_costs":[float(fit.cost) for fit in fits]},
        "wider_bounds_sensitivity":{"threshold_local_v":float(wider.x[1]),
                                    "maximum_log_conductance":float(wider.x[2]),
                                    "training_cost":float(wider.cost),
                                    "entry_predicted_uS":wider_entry},
        "results":scores}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
