"""Connect a competitive energy-state candidate to cross-scan evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def run():
    main = json.loads((ROOT / 'analysis/barrier_height_comparison.json').read_text())
    valid = json.loads((ROOT / 'analysis/barrier_height_validation.json').read_text())
    old = json.loads((ROOT / 'analysis/barrier_state_sensitivity.json').read_text())
    bound = json.loads((ROOT / 'analysis/wkb_frozen_polarity_bound.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.7))
    fig.subplots_adjust(left=.09, right=.97, bottom=.15, top=.87, hspace=.48, wspace=.32)
    scores = [old['grouped_higher_voltage_scores'][k] for k in ('power', 'pf_like', 'barrier')]+[valid['grouped_higher_voltage_scores']]
    ax = axes[0, 0]; x = np.arange(4)
    ax.bar(x-.17, [s['rmse_decade'] for s in scores], .33, color='#0072B2', label='All retained test points')
    ax.bar(x+.17, [s['high_signal_capacity_compatible_rmse_decade'] for s in scores], .33, color='#E69F00', label='Signal / capacity sensitivity')
    for j, score in enumerate(scores):
        ax.text(j-.17, score['rmse_decade']+.005, f"{score['rmse_decade']:.3f}", ha='center', fontsize=8)
    ax.set(xticks=x, xticklabels=['Power', 'PF-like', 'Width\nstate', 'Height\nstate'], ylim=(0, .25),
           ylabel='Voltage-extrapolation RMSE (decade)')
    ax.set_title('a  Height modulation is competitive at gate 1.3 V', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    ax = axes[0, 1]; fits = main['splits']['all_descriptive']['fits']; widths = main['fixed_width_profile_nm']
    ax.plot(widths, [fits[f'width_{g:g}']['rmse_decade'] for g in widths], '-o', color='#009E73', ms=5)
    ax.set(xscale='log', yscale='log', xticks=[.25, .5, 1., 2., 4., 6.], xticklabels=['0.25', '0.5', '1', '2', '4', '6'],
           xlabel='Fixed width in the conditional profile (nm)', ylabel='All-branch descriptive RMSE (decade)')
    ax.set_title('b  The profile selects 4 nm under the stated priors', loc='left', fontsize=10, fontweight='bold')
    ax.text(.97, .83, '300 K; effective mass 0.5 mₑ\nSupply coefficient ≤ 0.1 S\n4 nm is not an identified physical gap',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax = axes[1, 0]; rows = bound['first3_primary_at_0p1V']; x = np.arange(1, 4)
    ax.plot(x, [r['observed_ratio'] for r in rows], 'o', color='#0072B2', label='Measured ratio')
    ax.plot(x, [r['favorable_zero_reference_ratio_upper'] for r in rows], 's', color='#D55E00', label='Favorable zero-reference upper ratio')
    ax.axhline(rows[0]['frozen_passive_wkb_ratio_lower'], color='black', ls='--', label='Frozen passive WKB lower bound')
    ax.set(yscale='log', xticks=x, xlabel='Confirmed SET / RESET pair at gate 1.1 V', ylabel='|I(−0.1 V)| / I(+0.1 V)', ylim=(3e-5, .06))
    ax.set_title('c  Frozen passive WKB fails for the first 3 pairs', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    ax = axes[1, 1]; minima = [1000*r['minimum_extra_landscape_difference_lower_ev_over_8_processing_methods']
                              for r in bound['robust_first3_summary']]
    ax.bar(x, minima, width=.5, color='#D55E00')
    for j, val in zip(x, minima):
        ax.text(j, val+2., f'≥ {val:.1f}', ha='center', fontsize=9)
    ax.set(xticks=x, ylim=(0, 102), xlabel='Confirmed SET / RESET pair at gate 1.1 V',
           ylabel='Conditional extra landscape difference (meV)')
    ax.set_title('d  The required departure can be quantified', loc='left', fontsize=10, fontweight='bold')
    ax.text(.04, .96, 'Only if the other WKB / supply assumptions remain.\nNot a measured trap energy or contact potential.',
            transform=ax.transAxes, va='top', fontsize=8)
    fig.suptitle('Barrier-height modulation is competitive; a common frozen state fails the polarity test', y=.965, fontsize=13)
    fig.text(.5, .914, 'Local constitutive evidence (gate 1.3 V) and independently derived polarity constraints (gate 1.1 V)', ha='center', fontsize=9)
    fig.text(.09, .057, 'Panel a: same five grouped folds, target 0.08–0.16 V identifies a state, 0.17–0.30 V is withheld. This is not temporal prediction.\n'
             'Panels c–d assume the same frozen state, passive potential division, equal temperature and common energy-independent supply weights. Measurement systematics remain uncalibrated.', fontsize=7.8)
    path = ROOT / 'analysis/figures/35_barrier_height_polarity'
    fig.savefig(path.with_suffix('.png'), dpi=180); fig.savefig(path.with_suffix('.pdf'))
    print(path)


if __name__ == '__main__':
    run()
