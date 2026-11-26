"""Compare conditional SET probes on the identical original training mask."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'reset_shape_2026-11-03'
paths = [PREVIOUS / 'soft_field_mobility_probe.npz', HERE / 'sinh_field_probe.npz']
with np.load(paths[0]) as z:
    voltage = z['command_voltage_v'].copy()
    observed = abs(z['observed_current_a'])
    mask = z['original_scoring_mask'].copy()
    linear = z['refined_current_a'].copy()
with np.load(paths[1]) as z:
    np.testing.assert_array_equal(voltage, z['command_voltage_v'])
    np.testing.assert_array_equal(observed, abs(z['observed_current_a']))
    np.testing.assert_array_equal(mask, z['original_scoring_mask'])
    nonlinear = z['refined_current_a'].copy()
assert mask.sum() == 500
assert np.all(linear[mask] > 0) and np.all(nonlinear[mask] > 0)
outward = np.arange(len(voltage)) <= np.argmax(voltage)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .15,
                     'pdf.fonttype': 42, 'savefig.dpi': 300})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, branch, title in zip(axes, [outward, ~outward],
                           ['a  Outward SET: onset and growth', 'b  Return SET: high-current shape']):
    use = branch & mask
    for values, color, style, label in [
        (observed, '#333333', '-', 'Measured cycle 4'),
        (linear, '#0072B2', '--', 'Linear local-field mobility'),
        (nonlinear, '#D55E00', '-', 'Fixed sinh local-field probe')]:
        scale = 1. if ax is axes[0] else 1e6
        ax.plot(voltage[use], values[use] * scale, style,
                color=color, lw=1.8, label=label)
    ax.set(xlabel='Applied command voltage (V)', title=title, xlim=(0, 5))
axes[0].set(yscale='log', ylabel='Current magnitude (A)',
            ylim=(.5 * min(observed[mask].min(), linear[mask].min(), nonlinear[mask].min()), 1e-4))
axes[0].axvline(.2, color='#777777', lw=.8, ls=':')
axes[1].set(ylabel='Current magnitude (microampere)', ylim=(0, 60))
axes[1].invert_xaxis()
fig.suptitle('Single-cycle SET diagnosis and one fixed field-selectivity probe', fontsize=14, y=1.01)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, frameon=False, loc='upper center',
           bbox_to_anchor=(.5, .96), fontsize=9)
fig.text(.5, .02, 'Same 500 original eligible samples; both initial states are conditioned on the measured outward 0.2 V point.\n'
         'One exploratory field scale; no parameter scan. This is not full-history validation of a unified 1R/1T1R model.',
         ha='center', fontsize=8.5)
fig.tight_layout(rect=(0, .12, 1, .90), w_pad=2)
for extension in ['png', 'pdf']:
    fig.savefig(HERE / f'soft_drive_comparison.{extension}', bbox_inches='tight')
metrics = {}
for label, pred in [('linear_local_field', linear), ('fixed_sinh_local_field', nonlinear)]:
    metrics[label] = {
        name: float(np.sqrt(np.mean(np.log10(pred[use] / observed[use]) ** 2)))
        for name, use in [('all500', mask), ('outward250', mask & outward), ('return250', mask & ~outward)]}
metadata = {'scope': 'Conditional cycle4 SET, original500-point mask; not a full model score',
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'raw_RMSE_decade': metrics}
(HERE / 'soft_drive_comparison.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
