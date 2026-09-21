"""Distinguish a deterministic mean trajectory from stochastic switching paths."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def run():
    stochastic = json.loads((ROOT / 'analysis/stochastic_reentry_probe.json').read_text())
    deterministic = json.loads((ROOT / 'analysis/bidirectional_1.3_23_single_hop.json').read_text())
    va = np.array(stochastic['applied_abs_voltage_V']); obs = np.array(stochastic['observed_current_A'])
    apex = stochastic['outward_apex_raw_index']; candidates = stochastic['candidates']
    names = ['constant', 'single_hop', 'polarizable_saddle']
    labels = ['Constant\nrates', 'Field-tilted\nrates', 'Common saddle\nlowering']
    colors = ['#999999', '#0072B2', '#D55E00']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.7))
    fig.subplots_adjust(left=.085, right=.96, bottom=.16, top=.875, hspace=.5, wspace=.3)
    ax = axs[0, 0]
    ax.plot(va[:apex+1], obs[:apex+1], color='#0072B2', label='Measured outward')
    ax.plot(va[apex:], obs[apex:], color='#D55E00', label='Measured return')
    ax.plot(va, deterministic['fit']['predicted_current_A'], '--', color='black', lw=1.2,
            label='Best deterministic single-hop fit')
    ax.set(yscale='log', xlim=(.08, 1.5), ylim=(1e-9, 3e-4), xlabel='Applied negative-voltage magnitude (V)', ylabel='Current magnitude (A)')
    ax.set_title('a  The deterministic mean misses the sharp observed path', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    ax.text(.12, 1.3e-4, 'Gate 1.3 V, trace 23\nFull-trace fit: 0.879 decade RMSE', fontsize=8)

    ax = axs[0, 1]
    for name, color, label in zip(names, colors, labels):
        ax.plot(va[apex:], candidates[name]['conditional_return_high_probability_by_point'], color=color,
                lw=1.6, label=label.replace('\n', ' '))
    ax.set(xlim=(1.5, 0.), ylim=(0, 1.04), xlabel='Return applied-voltage magnitude (V)', ylabel='Probability of high configuration at each point')
    ax.set_title('b  Return-state probabilities use outward-fitted rates', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.text(.05, .31, 'All three begin with the observed high state\nat the outward peak; return currents are withheld.',
            transform=ax.transAxes, fontsize=8)

    ax = axs[1, 0]
    row = candidates['single_hop']
    for y, key, actual, color in [(1, 'drop_voltage_q10_median_q90_conditioned_on_pair', .39, '#0072B2'),
                                  (0, 'rise_voltage_q10_median_q90_conditioned_on_pair', 1.09, '#D55E00')]:
        q10, median, q90 = row[key]
        ax.errorbar(median, y, xerr=[[median-q10], [q90-median]], fmt='o', color=color, capsize=6, label='Simulated 10–90% range' if y == 1 else None)
        ax.scatter([actual], [y], marker='x', color='black', s=65, zorder=5, label='Observed native after-point' if y == 1 else None)
    ax.set(yticks=[0, 1], yticklabels=['Recovery', 'Drop'], xlim=(0, 1.5), ylim=(-.55, 1.65),
           xlabel='Applied negative-voltage magnitude at observed state change (V)')
    ax.set_title('c  Random event positions remain broadly distributed', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.text(.03, .04, 'Ranges conditioned on one observed drop and one recovery\nbefore the peak; 20,000 model paths, not additional experiments.', transform=ax.transAxes, fontsize=8)

    ax = axs[1, 1]
    values = [candidates[n]['conditional_all_sampled_return_high_probability'] for n in names]
    ax.bar(range(3), values, color=colors, width=.65)
    for j, value in enumerate(values): ax.text(j, value+.012, f'{value:.3f}', ha='center', fontsize=9)
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set(ylim=(0, .43), ylabel='Probability of high at every sampled return point')
    ax.set_title('d  One return observation cannot select a mechanism', loc='left', fontsize=10, fontweight='bold')
    ax.text(.02, .88, 'The high-current return envelope still has\n0.328 decade prediction error.', transform=ax.transAxes, fontsize=8)
    fig.suptitle('Bidirectional kinetics: a useful stochastic explanation remains quantitatively incomplete', fontsize=13, fontweight='bold', y=.98)
    fig.text(.085, .92, 'Two deterministic laws: 11 selected complete traces each. Stochastic probe: one retrospectively selected trace.', fontsize=9)
    fig.text(.085, .035,
             'a: all trace currents used in fitting. b–d: outward jump labels are supplied, then the return is withheld. No cross-cycle prediction is established.\n'
             'The transport envelopes and state labels do not identify atomic species. Absolute rates and retention require missing timing/state information.\n'
             'No unique mechanism is selected; ordinary cycles, stochastic event counts and SET-to-RESET history still require joint testing.', fontsize=8)
    for ext in ('png', 'pdf'): fig.savefig(ROOT / f'analysis/figures/32_bidirectional_stochastic_evidence.{ext}', dpi=220)
    plt.close(fig)


if __name__ == '__main__': run()
