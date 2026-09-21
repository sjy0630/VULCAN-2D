"""Separate sequential data evidence from optimistic current-support fits."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_cycle_order_evidence import cusum
ROOT = Path(__file__).resolve().parents[1]


def run():
    order = json.loads((ROOT/'analysis/cycle_order_evidence.json').read_text())['analyses']['raw']
    cert = json.loads((ROOT/'analysis/cycle_order_exact_certificate.json').read_text())['analyses']['raw']
    audit = json.loads((ROOT/'analysis/stochastic_domain_refinement.json').read_text())
    source = json.loads((ROOT/'analysis/stochastic_readout_support_validation.json').read_text())
    data = np.load(ROOT/'analysis/stochastic_domain_refinement.npz')
    fixed = np.load(ROOT/'analysis/stochastic_readout_support_validation.npz')
    joint = np.load(ROOT/'analysis/stochastic_domain_support_all_descriptive_N4.npz')
    x = np.asarray(order['features']); n = len(x); cycle = np.arange(1, n+1)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(2, 3, figsize=(13., 7.5), constrained_layout=True)
    ax = axes[0, 0]; ax.semilogy(cycle, 10**x[:, 2], 'o-', color='#0072B2', lw=.7, ms=3)
    ax.axvspan(.5, 3.5, color='#D55E00', alpha=.12)
    ax.set(title='a  Measured low-bias polarity history', xlabel='Pair index', ylabel='RESET outward / SET return', xlim=(.5, 53.5))
    ax = axes[0, 1]; values = cusum(x)
    for j, color, label in [(0, '#999999', 'SET return'), (1, '#D55E00', 'RESET outward'), (2, '#0072B2', 'Log ratio')]:
        ax.plot(range(1, n), values[:, j], color=color, label=label)
    ax.set(title=f'b  Joint search: exact p = {cert["exact_probability"]:.2g}', xlabel='Cut after pair', ylabel='Standardized CUSUM')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 2]; j = int(np.flatnonzero(np.isclose(joint['unique_voltage'], -.2, atol=1e-10, rtol=0))[0])
    levels = joint['configuration_current'][:, j]
    pure = np.count_nonzero(joint['fractions'] > 0, axis=1) == 1
    for value in levels[~pure]: ax.axhline(value, color='.8', lw=.5)
    for value in levels[pure]: ax.axhline(value, color='#D55E00', lw=1.1, ls='--')
    mask = (data['window'] == 4) & np.isclose(data['voltage'], -.2, atol=1e-10, rtol=0)
    ax.semilogy(data['cycle'][mask], data['observed'][mask], 'o', color='#0072B2', ms=3)
    ax.set(title='c  Possible current levels at -0.20 V', xlabel='Pair index', ylabel='Current (A)', ylim=(1e-10, 3e-6))
    ax.text(.03, .96, 'Orange: pure-state levels\nGray: mixtures of four domains', transform=ax.transAxes, va='top', fontsize=8)
    ax = axes[1, 0]; domain_counts = [1, 2, 4, 8, 16]
    values = [audit['candidates'][f'all_descriptive_fixed_N{n}']['full_support_scores']['rmse_decade'] for n in domain_counts]
    ax.plot(domain_counts, values, 'o-', color='#0072B2', label='Fixed N=1-fit readout')
    ax.plot(4, audit['candidates']['all_descriptive_joint_N4']['full_support_scores']['rmse_decade'], '*', color='#D55E00', ms=12, label='Readout refitted at N=4')
    infinite = source['domain_support']['all_descriptive']['continuous_arbitrary_population_support']['full_support_scores']['rmse_decade']
    ax.axhline(infinite, color='.5', ls=':', label='Fixed-readout continuous interval')
    ax.set_xscale('log', base=2); ax.set_xticks(domain_counts, [str(n) for n in domain_counts])
    ax.set(title='d  Oracle support error; no path prediction', xlabel='Effective domain count', ylabel='Eight-window RMSE (decade)')
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 1]
    for key, color, label in [('all_descriptive_fixed_N1', '#999999', 'N=1, four states'),
                               ('all_descriptive_fixed_N4', '#0072B2', 'N=4, fixed readout'),
                               ('all_descriptive_joint_N4', '#D55E00', 'N=4, refitted readout')]:
        per = audit['candidates'][key]['full_support_scores']['per_cycle']
        ax.plot(cycle, [per[str(c)] for c in cycle], color=color, label=label)
    ax.set(title='e  Optimistic residuals across all 53 pairs', xlabel='Pair index', ylabel='Eight-window RMSE (decade)')
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 2]
    for c, color in [(3, '#0072B2'), (4, '#D55E00')]:
        m = (data['cycle'] == c) & (data['window'] == 4) & (abs(data['voltage']) >= .1-1e-10)
        ax.semilogy(abs(data['voltage'][m]), data['observed'][m], 'o-', ms=3, color=color, label=f'Measured {c}')
        ax.semilogy(abs(data['voltage'][m]), data['all_descriptive_current'][m], '--', color=color, label=f'Best state at each point {c}')
    ax.set(title='f  Current values fit; path not yet validated', xlabel='Applied voltage magnitude (V)', ylabel='Current (A)')
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Cycle order constrains stochastic explanations; current support alone is not a mechanism', fontsize=12)
    for ext in ('png', 'pdf'): fig.savefig(ROOT/'analysis/figures'/f'41_stochastic_readout_evidence.{ext}')
    plt.close(fig); data.close(); fixed.close(); joint.close()


if __name__ == '__main__': run()
