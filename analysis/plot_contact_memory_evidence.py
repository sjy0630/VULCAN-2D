"""Display the selective-change constraint and both counterfactual directions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]


def run():
    result = json.loads((ROOT/'analysis/contact_memory_counterfactual.json').read_text())
    d = np.load(ROOT/'analysis/contact_memory_counterfactual.npz')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(2, 2, figsize=(11., 7.5), constrained_layout=True)
    ax = axes[0, 0]
    diagonal = np.logspace(-3, 3, 101); ax.loglog(diagonal, diagonal, ':', color='.65', label='Equal relative change')
    for split, marker, short in [('all_descriptive', 'o', 'All 53'), ('chronological', '^', 'First 35')]:
        for label, color, direction in [('negative_to_positive', '#0072B2', '-g to +g'),
                                        ('positive_to_negative', '#D55E00', '+g to -g')]:
            r = result['candidates'][f'{split}_{label}']
            ax.scatter(r['SET_return_4_over_3'], r['RESET_outward_4_over_3'], color=color,
                       marker=marker, s=45, label=f'{short}: {direction}')
    ax.scatter(result['observed_SET_return_4_over_3'], result['observed_RESET_outward_4_over_3'],
               marker='*', color='black', s=150, label='Measured')
    ax.set(title='a  Required: selective RESET change', xlabel='SET return: cycle 4 / cycle 3',
           ylabel='RESET outward: cycle 4 / cycle 3', xlim=(.001, 1e3), ylim=(.001, 1e3))
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    for ax, window, phase, letter in [(axes[0, 1], 2, 'SET return', 'b'), (axes[1, 0], 4, 'RESET outward', 'c')]:
        for cycle, color in [(3, '#0072B2'), (4, '#D55E00')]:
            mask = (d['cycle'] == cycle) & (d['window'] == window) & (abs(d['voltage']) >= .1-1e-10)
            x = abs(d['voltage'][mask]); sort = np.argsort(x)
            ax.semilogy(x[sort], d['observed'][mask][sort], 'o', ms=4, color=color, label=f'Measured {cycle}')
            for split, style, short in [('all_descriptive', '--', 'All 53'), ('chronological', '-', 'First 35')]:
                y = d[f'{split}_negative_to_positive_current_A'][mask]
                ax.semilogy(x[sort], y[sort], style, color=color, lw=1.2, label=f'{short} parameters, {cycle}')
        ax.set(title=f'{letter}  {phase}: imposed -g to +g switch', xlabel='Applied voltage magnitude (V)',
               ylabel='Current magnitude (A)')
        ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 1]; cycles = np.arange(1, 54)
    for label, color, text in [('fixed_positive', '.5', 'Fixed +g benchmark'),
                               ('negative_to_positive', '#0072B2', '-g to +g'),
                               ('positive_to_negative', '#D55E00', '+g to -g')]:
        r = result['candidates'][f'chronological_{label}']['full_scores']['per_cycle']
        ax.plot(cycles, [r[str(c)] for c in cycles], color=color, label=text, lw=1.1)
    ax.axvspan(.5, 3.5, color='.6', alpha=.12); ax.axvline(35.5, color='.5', ls=':', lw=.8)
    ax.set(title='d  All native curves remain in the assessment', xlabel='Cycle index',
           ylabel='Eight-window RMSE (decade)', xlim=(.5, 53.5))
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Contact-ratio exchange changes both polarities; selective experimental jump remains unexplained', fontsize=11)
    fig.supxlabel('No parameter refit. Switch imposed after pair 3 at zero bias; full state carried continuously.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT/'analysis/figures'/f'42_contact_memory_counterfactual.{ext}')
    plt.close(fig); d.close()


if __name__ == '__main__': run()
