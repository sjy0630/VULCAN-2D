"""Descriptive improvement versus prediction and shape failures."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
main = json.loads((ROOT/'analysis/side_charge_readout.json').read_text())
validation = json.loads((ROOT/'analysis/side_charge_validation.json').read_text())
high = json.loads((ROOT/'analysis/side_charge_high_voltage.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
blue, orange, green = '#0072B2', '#D55E00', '#009E73'
plt.rcParams.update({'font.size':10})
modes = ['frozen', 'equilibrium']; names = ['Frozen mixture', 'Equilibrium charge']
values = [main['candidates'][m]['splits']['all_descriptive']['fit']['rmse_decade'] for m in modes]
ax = axes[0, 0]; ax.bar(names, values, color=[orange, blue], width=.5)
for i, value in enumerate(values):ax.text(i, value+.008, f'{value:.3f}', ha='center')
ax.set(ylabel='Weighted descriptive RMSE (decade)', ylim=(0, .38))
ax.set_title('a  Charge readout improves the joint fit', loc='left', fontweight='bold')
ax = axes[0, 1]; x = np.arange(2)
for j, group in enumerate(['all', 'first3']):
    values = [validation['native_RESET_port_forecasts'][m]['groups'][group]['equal_pair_rmse_decade'] for m in modes]
    ax.bar(x+(j-.5)*.32, values, width=.3, color=[blue, orange][j], label=['All 53 pairs', 'First 3 pairs'][j])
    for xx, value in zip(x+(j-.5)*.32, values):ax.text(xx, value+.035, f'{value:.3f}', ha='center', fontsize=9)
ax.set(xticks=x, xticklabels=names, ylabel='Native RESET forecast RMSE (decade)', ylim=(0, 3.55))
ax.legend(frameon=False, loc='upper center', ncol=2, fontsize=9); ax.set_title('b  SET-conditioned prediction still fails', loc='left', fontweight='bold')
ax = axes[1, 0]; r = [r for r in high['rows'] if r['mode'] == 'equilibrium']
ax.scatter([r['cycle_id'] for r in r], [r['observed_slope'] for r in r], label='Observed', s=20, color=green)
ax.plot([r['cycle_id'] for r in r], [r['predicted_slope'] for r in r], label='Charge model', color=blue)
ax.axhline(1, color='0.5', ls='--', lw=1)
ax.set(xlabel='Confirmed paired cycle', ylabel='Local log-log slope at internal 2–3 V', ylim=(.4, 1.13))
ax.legend(frameon=False); ax.set_title('c  Positive-return sublinearity is missed', loc='left', fontweight='bold')
ax = axes[1, 1]; profiles = sorted(validation['SET_energy_information_profiles']['equilibrium'], key=lambda r:r['cycle_id'])
ids = [r['cycle_id'] for r in profiles]
low = np.array([r['negative_prediction_log10_A_min'][0] for r in profiles])
upper = np.array([r['negative_prediction_log10_A_max'][0] for r in profiles])
ax.fill_between(ids, 10**(low+9), 10**(upper+9), color=blue, alpha=.25, label='SET-tolerance prediction range')
v = np.array(main['voltage_V']); index = np.flatnonzero(np.isclose(v, -.1))[0]
observed = np.array(main['observed_log10_current_A'])[:, index]
ax.scatter(ids, 10**(observed+9), s=18, color=orange, label='Observed RESET')
ax.set(yscale='log', xlabel='Confirmed paired cycle', ylabel='Current at internal −0.1 V (nA)')
ax.legend(frameon=False, fontsize=9); ax.set_title('d  SET-compatible states still miss the first 3', loc='left', fontweight='bold')
for ax in axes.ravel():
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Equilibrium charge gating improves fits but fails the SET→RESET test', fontsize=16, y=.985)
fig.text(.075, .025, 'All 53 pairs retained. Panel b: native applied-voltage input, unique load roots, target RESET used only for scoring.\n'
         'Panel d: fixed training physics; SET RMSE within +0.005 decade of its energy-profile minimum. This range is not a confidence interval.', fontsize=9)
fig.tight_layout(rect=(.02, .08, 1, .96), h_pad=2.3, w_pad=2.2)
folder = ROOT/'analysis/figures'; folder.mkdir(exist_ok=True)
for suffix in ('png', 'pdf'):
    fig.savefig(folder/f'36_side_charge_evidence.{suffix}', dpi=180)
print(folder/'36_side_charge_evidence.png')
