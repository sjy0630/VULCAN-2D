"""Physical transport consistency, predictive comparison and parameter ambiguity."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def run():
    main = json.loads((ROOT / 'analysis/barrier_state_comparison.json').read_text())
    sensitivity = json.loads((ROOT / 'analysis/barrier_state_sensitivity.json').read_text())
    full = main['splits']['all_descriptive']['fits']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
    fig.subplots_adjust(left=.06, right=.98, bottom=.25, top=.77, wspace=.31)
    ax = axes[0]; heights = main['height_profile_ev']
    ax.plot(heights, [full[f'barrier_{h:g}']['rmse_decade'] for h in heights], '-o', color='#0072B2', ms=4,
            label='Supply ceiling 0.1 S (main comparison)')
    for upper, color in ((2., '#E69F00'), (6., '#009E73')):
        rows = [r for r in sensitivity['posthoc_descriptive_parameter_sensitivities']
                if r['kind'] == 'supply_prefactor_upper_bound' and r['log10_prefactor_upper_S'] == upper]
        ax.plot([r['height_ev'] for r in rows], [r['fit']['rmse_decade'] for r in rows], '--o', color=color,
                ms=4, label=f'Supply ceiling 10$^{upper:g}$ S (sensitivity)')
    ax.axhline(full['pf_like']['rmse_decade'], color='black', ls=':', label='PF-like amplitude control')
    ax.set(xscale='log', yscale='log', xlabel='Shared barrier height (eV)', ylabel='All-branch descriptive RMSE (decade)',
           xticks=[.1, .2, .5, 1., 2., 4.], xticklabels=['0.1', '0.2', '0.5', '1', '2', '4'])
    ax.set_title('a  Barrier selection depends on the supply prior', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax = axes[1]; x = np.arange(3); names = ('power', 'pf_like', 'barrier')
    all_scores = [sensitivity['grouped_higher_voltage_scores'][k]['rmse_decade'] for k in names]
    robust = [sensitivity['grouped_higher_voltage_scores'][k]['high_signal_capacity_compatible_rmse_decade'] for k in names]
    ax.bar(x-.18, all_scores, .35, color='#0072B2', label='All held-out voltage points')
    ax.bar(x+.18, robust, .35, color='#E69F00', label='Signal / capacity sensitivity')
    for j, (a, b) in enumerate(zip(all_scores, robust)):
        ax.text(j-.18, a+.005, f'{a:.3f}', ha='center', fontsize=8)
        ax.text(j+.18, b+.005, f'{b:.3f}', ha='center', fontsize=8)
    ax.set(xticks=x, xticklabels=['Power', 'PF-like', 'Width-state\nbarrier'], ylim=(0, .25),
           ylabel='Grouped voltage-extrapolation RMSE (decade)')
    ax.set_title('b  The physical candidate has not won prediction', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax = axes[2]
    default = np.array(full['barrier_0.5']['local_parameters'])
    relaxed = next(r for r in sensitivity['posthoc_descriptive_parameter_sensitivities']
                   if r['kind'] == 'supply_prefactor_upper_bound' and r['log10_prefactor_upper_S'] == 6. and r['height_ev'] == 1.)
    alternate = np.array(relaxed['fit']['local_parameters'])
    for phase, color in (('outward', '#0072B2'), ('return', '#D55E00')):
        chosen = np.array([r['branch'] == phase for r in main['branches']])
        ax.scatter(default[chosen], alternate[chosen], s=12, alpha=.65, color=color, label=phase.capitalize())
    ax.plot([0, 6], [0, 6], ':', color='gray', lw=1.)
    ax.set(xlim=(-.1, 6.1), ylim=(-.1, 6.1), xlabel='Conditional width at 0.5 eV, 2.22 mS (nm)',
           ylabel='Conditional width at 1 eV, 3.58×10$^5$ S (nm)')
    ax.set_title('c  Similar fits do not identify a unique gap', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('A finite-temperature barrier state is testable, but its geometry is not identified', fontsize=14, y=.96)
    fig.text(.5, .875, '101 RESET traces · 202 low-bias branches · forward transistor coupling · all observations retained', ha='center', fontsize=10)
    fig.text(.06, .09, 'Panel b: five cycle-grouped folds; target 0.08–0.16 V identifies one state, then 0.17–0.30 V is predicted. This is constitutive voltage extrapolation, not time prediction.\n'
             'The sensitivity subset requires current > 3× zero reference and compatibility with transistor capacity. Panel c uses descriptive fits; widths assume effective mass 0.5 mₑ.', fontsize=8)
    path = ROOT / 'analysis/figures/34_barrier_state_evidence'
    fig.savefig(path.with_suffix('.png'), dpi=180); fig.savefig(path.with_suffix('.pdf'))
    print(path)


if __name__ == '__main__':
    run()
