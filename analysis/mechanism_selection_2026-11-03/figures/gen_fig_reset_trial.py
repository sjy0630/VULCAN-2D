"""Plot frozen conditional RESET results only; no fits or holdout reads."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'field_rupture_trial.npz'
REPORT = HERE.parent / 'field_rupture_trial.json'
with np.load(SOURCE) as z:
    v = abs(z['command_voltage_v'])
    obs = z['observed_reset_abs_current_a']
    mask = z['original_reset_mask']
    pred = z['prior_SET_conditioned_LOO_abs_current']
    control = z['fold_training_SET_geomean_constant_LOO_abs_current']
    ids = z['record_ids']
r = json.loads(REPORT.read_text())
assert np.array_equal(ids, r['predeclared_plan']['training_ids'])
assert mask.sum() == 2799 and np.isfinite(pred).all()
apex = np.argmax(v)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .15, 'pdf.fonttype': 42,
    'savefig.dpi': 300})
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3))
colors = ['#555555', '#0072B2', '#D55E00']
for ax, sel, title in zip(axes[0],
        [np.arange(len(v)) <= apex, np.arange(len(v)) > apex],
        ['a  Outward RESET', 'b  Return RESET']):
    sel = sel & (v > 0)
    for values, color, style, label in zip([obs, pred, control], colors,
            ['-', '-', '--'], ['Measured', 'Prior-SET conditioned (LOO)', 'Constant-state control (LOO)']):
        # Common observation mask; gaps are retained as gaps, not bridged.
        shown = np.where(mask[:, sel], values[:, sel], np.nan)
        q = np.nanquantile(shown, [.1, .5, .9], axis=0)
        if values is not control:
            ax.fill_between(v[sel], q[0], q[2], color=color, alpha=.13, linewidth=0)
        ax.plot(v[sel], q[1], style, color=color, lw=1.8, label=label)
    ax.set_yscale('log')
    ax.set(xlabel=r'$|V_h|$ (V)', ylabel=r'$|I|$ (A)', title=title,
           xlim=(0, 2.5), ylim=(1e-12, 3e-2))
axes[0, 1].invert_xaxis()

out = np.arange(apex + 1)
obs_peak_indices = np.argmax(obs[:, out], axis=1)
pred_peak_indices = np.argmax(pred[:, out], axis=1)
obs_peak_v = v[obs_peak_indices]
pred_peak_v = v[pred_peak_indices]
obs_peak_i = obs[np.arange(len(ids)), obs_peak_indices]
pred_peak_i = pred[np.arange(len(ids)), pred_peak_indices]
ax = axes[1, 0]
ax.scatter(obs_peak_v, obs_peak_i * 1e3, color=colors[0], marker='o', s=32)
ax.scatter(pred_peak_v, pred_peak_i * 1e3, color=colors[1], marker='x', s=40)
for x, y, xp, yp in zip(obs_peak_v, obs_peak_i, pred_peak_v, pred_peak_i):
    ax.plot([x, xp], [y * 1e3, yp * 1e3], color='#999999', alpha=.4, lw=.7)
ax.set(xlabel='Outward peak voltage (V)', ylabel='Outward peak current (mA)',
       title='c  All 14 cycles: peaks shift too early',
       xlim=(.25, max(obs_peak_v.max(), pred_peak_v.max()) + .05), ylim=(0, 16))

j = np.flatnonzero((np.arange(len(v)) > apex) & np.isclose(v, .2, atol=1e-7))[0]
ax = axes[1, 1]
ax.scatter(obs[:, j], pred[:, j], color=colors[1], s=35)
limits = [1e-11, 3e-9]
ax.plot(limits, limits, ':', color='#555555', lw=1)
ax.set(xscale='log', yscale='log', xlim=limits, ylim=limits,
       xlabel='Measured return current at 0.2 V (A)',
       ylabel='Conditional LOO prediction (A)', title='d  Residual tail current is too high')

handles = [Line2D([0], [0], color=c, ls=s, lw=2, label=l)
           for c, s, l in zip(colors, ['-', '-', '--'],
             ['Measured', 'Prior-SET conditioned (LOO)', 'Constant-state control (LOO)'])]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, .965))
fig.suptitle('Conditional RESET trial: limited gain, 1 of 4 feature gates passed', y=1, fontsize=14)
fig.text(.5, .012,
    'Top: across-cycle medians; shading = 10th–90th percentiles, not confidence intervals.\n'
    'Three-parameter LOO on a backbone selected using all 14 training cycles; preceding SET state supplied.\n'
    'Not end-to-end cross-validation, complete-history prediction, or a 1T1R test.',
    ha='center', fontsize=8.5)
fig.tight_layout(rect=(0, .09, 1, .92), h_pad=2.0)
for suffix in ['png', 'pdf']:
    fig.savefig(HERE / ('reset_trial_comparison.' + suffix), bbox_inches='tight')
metadata = {'purpose': 'Visualization of frozen conditional internal-LOO results; no optimization',
    'record_ids': ids.tolist(), 'original_scoring_mask_points': int(mask.sum()),
    'band': 'Empirical across-record Q10-Q90, not uncertainty of model parameters',
    'LOO_scope': 'Only three new parameters refitted within each fold; frozen Ron and spectrum previously selected using training data, so this is not end-to-end cross-validation',
    'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE, REPORT]},
    'feature_scope': 'Median of model features is the primary feature gate; peak scatter shows each record separately',
    'figure_sources': ['field_rupture_trial.npz', 'field_rupture_trial.json']}
(HERE / 'reset_trial_comparison.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Wrote reset_trial_comparison.png and .pdf')
