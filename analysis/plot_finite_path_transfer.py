"""Show why reproducing rare jumps alone does not validate a state model."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'constant': '#777777', 'single_hop': '#0072B2', 'polarizable_saddle': '#D55E00'}
LABELS = {'constant': 'Constant rates', 'single_hop': 'Field-tilted rates', 'polarizable_saddle': 'Common saddle lowering'}
RATIO = 'log10_return_outward_ratio_0p5_0p8'
CURRENT = 'log10_return_current_A_0p1_0p3'


def run():
    report = json.loads((ROOT / 'analysis/finite_path_transfer.json').read_text())
    matching = json.loads((ROOT / 'analysis/finite_path_initial_match.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(left=.09, right=.975, bottom=.15, top=.87, hspace=.46, wspace=.3)
    for candidate in report['candidates']:
        rows = [r for r in report['results'] if r['candidate'] == candidate]
        n = [r['effective_count'] for r in rows]
        axes[0, 0].plot(n, [100*r['zero_reference_event_sensitivity']['source']['drop_then_rise']['fraction'] for r in rows],
                        '-o', color=COLORS[candidate], ms=4, lw=1.5, label=LABELS[candidate])
        for ax, key in ((axes[0, 1], RATIO), (axes[1, 0], CURRENT)):
            ax.plot(n, [r['readouts'][key]['quantiles']['0.5'] for r in rows], '-o',
                    color=COLORS[candidate], ms=4, lw=1.5, label=LABELS[candidate])
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xscale('log', base=2); ax.set_xticks(report['effective_counts'], labels=report['effective_counts'])
        ax.set_xlabel('Effective count N (total transport held fixed)')
        ax.grid(axis='y', alpha=.16)
    ax = axes[0, 0]
    ax.axhline(2., color='black', ls='--', lw=1.2)
    ax.text(1.02, 3.2, 'Observed: 2 of 100 target traces', fontsize=8)
    ax.set(ylim=(-2, 63), ylabel='Persistent drop-then-rise paths (%)')
    ax.set_title('a  Increasing N suppresses large observable steps', loc='left', fontweight='bold', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    for ax, key, title, ylabel in (
            (axes[0, 1], RATIO, 'b  The typical return weakening remains mismatched', 'Median log$_{10}$(return / outward current)'),
            (axes[1, 0], CURRENT, 'c  Low-bias return current adds a separate constraint', 'Median log$_{10}$(return current / A)')):
        observed = report['target_summary']['readouts'][key]['quantiles']
        ax.axhspan(observed['0.1'], observed['0.9'], color='black', alpha=.08, zorder=0)
        ax.axhline(observed['0.5'], color='black', ls='--', lw=1.2)
        ax.set(ylabel=ylabel)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=10)
        if key == RATIO:
            ax.text(.98, .06, 'Measured at matched |V| = 0.5–0.8 V', ha='right', transform=ax.transAxes, fontsize=8)
        else:
            ax.text(.98, .055, 'Read window: |V| = 0.1–0.3 V on return', ha='right', transform=ax.transAxes, fontsize=8)
    axes[0, 1].set_ylim(-3.5, 3.8); axes[1, 0].set_ylim(-11.1, -4.7)
    ax = axes[1, 1]
    x = np.arange(len(matching['cases']))
    labels, values, upper = [], [], []
    for row in matching['cases']:
        factor = row['maximum_initial_current_factor_from_source']
        labels.append(('All targets' if factor is None else f'Within {factor:g}×')+f"\n(n={row['n_targets']})")
        values.append(row['observed_readouts'][RATIO]['median'])
        upper.append(row['zero_reference_sensitivity']['ratio_upper_median_log10_valid_only'])
    ax.plot(x, values, 'o-', color='black', lw=1.4, label='Observed median')
    ax.plot(x, upper, 's--', color='#009E73', lw=1.3, ms=4, label='Favorable zero-reference perturbation')
    for candidate, count in (('single_hop', 8), ('polarizable_saddle', 32)):
        row = next(r for r in report['results'] if r['candidate'] == candidate and r['effective_count'] == count)
        ax.axhline(row['readouts'][RATIO]['quantiles']['0.5'], color=COLORS[candidate], ls=':', lw=1.6,
                   label=LABELS[candidate]+f', N={count}')
    ax.set(xticks=x, xticklabels=labels, ylim=(-3., .4),
           xlabel='Initial 0.1–0.3 V outward current relative to source',
           ylabel='Median log$_{10}$(return / outward current)')
    ax.set_title('d  Similar starting currents do not close the gap', loc='left', fontweight='bold', fontsize=10)
    ax.legend(frameon=False, fontsize=7.4, loc='center right'); ax.grid(axis='y', alpha=.16)
    fig.suptitle('Frozen anomaly-derived rates do not explain the ordinary RESET ensemble', fontsize=14, y=.968)
    fig.text(.5, .916, 'Gate 1.3 V: trace 23 outward only supplies parameters; the other 100 traces are prediction targets.', ha='center', fontsize=9)
    fig.text(.09, .06, 'Model frequencies: 20,000 simulated paths per setting, not experimental trials. Panels b–c: dashed = target median; gray = target 10–90% range.\n'
             'N changes granularity, not a measured atom count. Panel d is a post-hoc sensitivity check. Device/history identity remains unconfirmed.', fontsize=8, va='center')
    output = ROOT / 'analysis/figures/33_finite_path_transfer'
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    print(output)


if __name__ == '__main__':
    run()
