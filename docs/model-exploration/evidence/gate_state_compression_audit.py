import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from vulcan2d import model_shared_device as model
from vulcan2d import validate_shared_device as shared

OUTPUT = Path('../outputs/vulcan-handoff')
ENTRY_V = 0.15
GATES = (0.9, 1.1, 1.3)
SCALES_UA = (0, 5, 15, 50, 150, 500)


def indexed(rows):
    result = {int(re.search(r'\d+$', row.label).group()): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate cycle numbers')
    return result


def observed_entry(row, circuit):
    voltage, current = np.abs(row.voltage), np.abs(row.current)
    grid, drain = circuit.transistor
    levels, use = np.unique(drain, return_index=True)
    local = voltage - np.interp(current, levels, grid[use])
    candidates = (row.masks['use_unclamped'] & row.masks['branch_outward']
                  & (current > 0) & (current < .95 * np.interp(voltage, grid, drain))
                  & (local >= .075) & (local <= .225))
    ids = np.flatnonzero(candidates)
    for left, right in zip(ids[:-1], ids[1:]):
        if right == left + 1 and local[left] <= ENTRY_V <= local[right] and local[left] < local[right]:
            return float(np.interp(ENTRY_V, local[[left, right]], current[[left, right]]) / ENTRY_V)
    return None


def quantiles(values):
    values = [v for v in values if v is not None]
    return {'n': len(values),
            'p10_median_p90_uS': (np.quantile(values, [.1, .5, .9])*1e6).tolist() if values else None}


def evaluate(sets, resets, sc, rc, device, prep, details=False):
    runs = [model.simulate_cycle(a.voltage, b.voltage, device, prep, sc, rc)
            for a, b in zip(sets, resets)]
    result = {}
    for label, rows, column in (('SET', sets, 0), ('RESET', resets, 1)):
        residuals = []
        for row, run in zip(rows, runs):
            mask = row.masks['use_unclamped']
            current = np.abs(run[column]['current_A'][mask])
            if np.any(~np.isfinite(current) | (current <= 0)):
                raise ValueError('Invalid scored current')
            residuals.extend(np.log10(current / np.abs(row.current[mask])))
        result[label] = float(np.sqrt(np.mean(np.square(residuals))))
    if details:
        finals = [a['final'] for a,b in runs]
        result['set_soft_end'] = [min(x.soft for x in finals), max(x.soft for x in finals)]
        result['set_hard_formed'] = sum(x.hard == 1 for x in finals)
        result['estimated_intrinsic_reset_entry'] = quantiles([
            float(model.device_current(ENTRY_V, x.soft, x.hard, device, prep, 'RESET')/ENTRY_V)
            for x in finals])
        result['observed_intrinsic_reset_entry'] = quantiles([observed_entry(row, rc) for row in resets])
    return result


groups, circuits = shared.load_groups()
pairs = {}
for gate in GATES:
    s, r = [("1T1M", gate, phase) for phase in ('SET', 'RESET')]
    sets, resets = indexed(groups[s]), indexed(groups[r])
    if sets.keys() != resets.keys():
        raise ValueError(f'Incompatible SET and RESET numbering at {gate} V')
    ids = sorted(sets)
    pairs[gate] = ([sets[i] for i in ids], [resets[i] for i in ids], circuits[s], circuits[r])

sets, resets, sc, rc = pairs[1.1]
training = (sets[:32], resets[:32], sc, rc)
pairs[1.1] = (sets[32:], resets[32:], sc, rc)
device, prep = shared.fit_soft_set(training[0], sc)
initial = model.simulate_sweep(training[0][0].voltage, 'SET', device, prep, sc)['final']
device, prep = shared.fit_soft_reset(training[1], rc, device, prep, initial)

results = {}
for scale in SCALES_UA:
    law = replace(device, soft_growth_current_a=scale*1e-6 if scale else None)
    if scale:
        _, calibrated = shared.fit_soft_set(training[0], sc, law, prep, fit_transitions=False)
        initial = model.simulate_sweep(training[0][0].voltage, 'SET', law, calibrated, sc)['final']
        _, calibrated = shared.fit_soft_reset(training[1], rc, law, calibrated, initial, fit_transitions=False)
    else:
        calibrated = prep
    results[str(scale)] = {
        'train': evaluate(*training, law, calibrated),
        'gates': {str(g): evaluate(*pairs[g], law, calibrated, details=True) for g in GATES}}
    print(scale, results[str(scale)], flush=True)
scores = {scale: np.mean(np.square(list(result['train'].values()))) for scale, result in results.items()}
best = min(scores, key=scores.get)

output = {'entry_voltage_device_v': ENTRY_V,
          'protocol': 'Same-numbered SET/RESET pairs, per-pair zero initial state. Train and select current scale on 1.1 V cycles 4–35 only. All candidate amplitudes are shared across gates; only SET/RESET envelope amplitudes are refitted at 1.1 V. Entry estimate inverts measured reverse transistor curves using adjacent valid outward points.',
          'scales_uA': results, 'training_selected_scale_uA': int(best)}
# Intervention using each target RESET entry. It diagnoses the contribution
# of the starting state and is deliberately not a held-out prediction.
off_g = float(model.device_current(ENTRY_V, 0., 0., device, prep, 'RESET')/ENTRY_V)
on_g = float(model.device_current(ENTRY_V, 1., 0., device, prep, 'RESET')/ENTRY_V)
output['reset_entry_intervention'] = {}
for gate, (set_rows, reset_rows, sc, rc) in pairs.items():
    baseline_errors, conditioned_errors, requested = [], [], []
    for set_row, reset_row in zip(set_rows, reset_rows):
        entry = observed_entry(reset_row, rc)
        if entry is None:
            continue
        q = (entry-off_g)/(on_g-off_g)
        requested.append(q)
        written, baseline = model.simulate_cycle(set_row.voltage, reset_row.voltage, device, prep, sc, rc)
        state = replace(written['final'], soft=float(np.clip(q, 0., 1.)))
        conditioned = model.simulate_sweep(reset_row.voltage, 'RESET', device, prep, rc, state)
        mask = reset_row.masks['use_unclamped'] & (np.abs(reset_row.voltage) >= .3)
        target = np.abs(reset_row.current[mask])
        baseline_errors.extend(np.log10(np.abs(baseline['current_A'][mask])/target))
        conditioned_errors.extend(np.log10(np.abs(conditioned['current_A'][mask])/target))
    output['reset_entry_intervention'][str(gate)] = {
        'cycles':len(requested),'required_soft_below_0':int(np.count_nonzero(np.array(requested)<0)),
        'required_soft_above_1':int(np.count_nonzero(np.array(requested)>1)),
        'baseline_rmse_decade':float(np.sqrt(np.mean(np.square(baseline_errors)))),
        'conditioned_rmse_decade':float(np.sqrt(np.mean(np.square(conditioned_errors))))}
output['reset_entry_intervention_scoring'] = 'Diagnostic using measured RESET entry at internal 0.15 V; predicted RESET points evaluated at |Vapp| >= 0.3 V on the matched subset. State inferred with fixed envelopes and clipped to [0, 1].'
output['fixed_soft_entry_conductance_range_uS'] = [off_g*1e6, on_g*1e6]
(OUTPUT/'gate_state_compression_audit.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
