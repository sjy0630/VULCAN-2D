"""Spectrum shape, grouped prediction and the conditional relaxation bound."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'analysis'))
from verify_charge_spectrum_artifacts import high_slopes, relaxation_bound
from vulcan2d.coupled_charge import VT
read = lambda p: json.loads((ROOT/'analysis'/p).read_text())
main = read('charge_spectrum_comparison.json'); wide = read('charge_spectrum_expanded_prior.json')
grouped = read('charge_spectrum_expanded_groups.json'); old = read('coupled_charge_comparison.json')
w = np.array(main['weights']); y = np.array(main['observed_log10_current_A'])
blue, orange, green, grey = '#0072B2', '#D55E00', '#009E73', '#777777'
fig, axes = plt.subplots(1, 3, figsize=(15, 5.1))
ax = axes[0]; ids = np.arange(1, 54)
ax.scatter(ids, high_slopes(y, w), s=13, color=green, label='Observed')
for data, name, color, style in [(old['splits']['all_descriptive'], 'Single level', grey, ':'),
                               (main['splits']['all_descriptive'], 'Spectrum: original prior', blue, '--'),
                               (wide, 'Spectrum: wider prior', orange, '-')]:
    ax.plot(ids, high_slopes(np.array(data['predicted_log10_A']), w), color=color, ls=style, label=name)
ax.set(xlabel='Confirmed paired cycle', ylabel='Log-log slope, internal 2–3 V', ylim=(-.03, .9))
ax.legend(frameon=False, fontsize=8, loc='center right')
ax.set_title('a  The energy prior affects shape', loc='left', fontweight='bold')
ax = axes[1]; x = np.arange(3); width = .23
for j, (data, name, color) in enumerate([(old, 'Single level', grey), (main, 'Spectrum', blue), (grouped, 'Wider prior', orange)]):
    values = [data['SET_only_five_fold_scores'][g]['negative_rmse_decade'] for g in ('all', 'first3', 'later')]
    bars = ax.bar(x+(j-1)*width, values, width, color=color, label=name)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.05, f'{value:.2f}', ha='center', fontsize=8, rotation=90)
ax.set(xticks=x, xticklabels=['All 53', 'Initial 3', 'Later 50'], ylabel='SET → RESET RMSE (decade)', ylim=(0, 3.75))
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_title('b  Prediction remains unresolved', loc='left', fontweight='bold')
ax = axes[2]; energy = np.linspace(-.15, .22, 400)
f = np.exp(-np.logaddexp(0., energy/VT)); limit = (1+np.sqrt(f))*np.exp(np.logaddexp(0., -energy/VT))
ax.semilogy(energy, limit, color=grey)
p0 = old['splits']['all_descriptive']['fit']['shared_parameters']; p1 = main['splits']['all_descriptive']['fit']['shared_parameters']; p2 = wide['fit']['shared_parameters']
for p, color in [(np.r_[p0[:5], 0.], grey), (p1, blue), (p2, orange)]:
    ax.scatter(p[0]-p[5], relaxation_bound(p), color=color, s=35, zorder=5)
ax.annotate('Spectrum fits: ≤1.23×', (p2[0]-p2[5], relaxation_bound(p2)), xytext=(-.09, 5), fontsize=9,
            arrowprops={'arrowstyle': '->', 'color': orange})
ax.set(xlabel='Lowest conducting energy at zero bias (eV)', ylabel='Upper bound on slow-rate acceleration', ylim=(.9, 1000))
ax.set_title('c  A conditional memory constraint', loc='left', fontweight='bold')
for ax in axes: ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('A broader charge spectrum helps shape, but does not close state prediction', fontsize=14, y=.99)
fig.text(.035, .035, 'All 53 pairs retained. Wider-prior results are post-hoc sensitivities. Panel c: constant attempt rates, independent channels, 300 K;\n'
         'a mathematical bound at the selected parameters, not measured retention or switching time. No full-cycle or microscopic-species claim.', fontsize=9)
fig.tight_layout(rect=(0, .15, 1, .95), w_pad=2.)
for suffix in ('png', 'pdf'): fig.savefig(ROOT/f'analysis/figures/38_charge_spectrum_evidence.{suffix}', dpi=190)
print(ROOT/'analysis/figures/38_charge_spectrum_evidence.png')
