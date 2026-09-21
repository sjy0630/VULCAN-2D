"""Render full-history prediction, state ambiguity and opposite loop orientation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def run():
    order = json.loads((ROOT/'analysis/reset_order_constraint.json').read_text())
    profile = json.loads((ROOT/'analysis/reset_history_transport_profiles.json').read_text())
    terminal = json.loads((ROOT/'analysis/reset_history_terminal.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig.subplots_adjust(left=.085, right=.94, top=.87, bottom=.15, wspace=.37, hspace=.45)
    ax = axs[0, 0]
    example = next(r for r in order['raw_strong_examples'] if r['gate_V'] == 1.3 and r['cycle_id'] == 23)
    v = np.abs(np.array(example['voltage_v'])); i = np.abs(np.array(example['current_a'])); apex = int(np.argmax(v))
    ax.plot(v[:apex+1], i[:apex+1], color='#0072B2', lw=1.5, label='Outward negative sweep')
    ax.plot(v[apex+1:], i[apex+1:], color='#D55E00', lw=1.5, label='Return sweep')
    ax.axvspan(.5, .8, color='#DDDDDD', alpha=.5)
    ax.set(yscale='log', xlim=(.08, 1.5), ylim=(1e-10, 4e-4), xlabel='Applied negative-voltage magnitude (V)', ylabel='Stored current magnitude (A)')
    ax.set_title('a  One negative ramp shows a drop, followed by recovery', loc='left', fontsize=10.2, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='center right')
    ax.text(.13, 1.6e-4, 'Gate 1.3 V, trace 23\nReturn / outward: 1,397x in shaded window', fontsize=8)
    for location, text in [(.385, 'Drop'), (1.085, 'Recovery')]:
        ax.axvline(location, color='#777777', ls=':', lw=.7)
        ax.text(location+.02, 2e-10, text, fontsize=8)

    ax = axs[0, 1]
    for gate, color, marker in [(0.9, '#009E73', 'o'), (1.1, '#0072B2', 's'), (1.3, '#D55E00', '^')]:
        rows = [r for r in order['cycles'] if r['gate_V'] == gate]
        ax.scatter([r['cycle_id'] for r in rows], [r['geometric_return_outward_ratio'] for r in rows],
                   color=color, marker=marker, s=18, alpha=.65, label=f'Gate {gate:.1f} V')
        strong = [r for r in rows if r['strong_order_reversal']]
        ax.scatter([r['cycle_id'] for r in strong], [r['geometric_return_outward_ratio'] for r in strong],
                   s=70, facecolor='none', edgecolor='black', lw=.9)
    ax.axhline(1, color='black', ls='--', lw=1); ax.axhline(3, color='#777777', ls=':', lw=.8)
    ax.set(yscale='log', xlim=(0, 103), ylim=(3e-5, 5000), xlabel='Numeric trace identifier', ylabel='Geometric return / outward current ratio')
    ax.set_title('b  Strong reversals are also present beyond the first trace', loc='left', fontsize=10.2, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.text(38, 2e-4, 'Circles: five robust order reversals\nMatched applied |V| = 0.50–0.80 V', fontsize=8)

    ax = axs[1, 0]
    rows = [r for r in profile['profiles'] if r['candidate'] == 'thermal_rate']
    x = [r['fixed_connected_alpha_per_V'] for r in rows]
    p1, = ax.plot(x, [r['rmse_decade'] for r in rows], 'o-', color='#0072B2', lw=1.5, label='Joint current RMSE')
    ax.set(xlabel='Fixed connected-envelope curvature (1/V)', ylabel='Joint current RMSE (decade)', ylim=(.15, .20), xlim=(1, 21))
    ax.tick_params(axis='y', labelcolor='#0072B2'); ax.yaxis.label.set_color('#0072B2')
    other = ax.twinx(); other.spines['right'].set_visible(True)
    p2, = other.plot(x, [r['eligible_half_survival_q10_median_q90_V'][1] for r in rows], 's--', color='#CC79A7', lw=1.5, label='Inferred half-survival voltage')
    other.set(ylabel='Median inferred half-survival voltage (V)', ylim=(0, 1.4)); other.tick_params(axis='y', labelcolor='#CC79A7'); other.yaxis.label.set_color('#CC79A7')
    ax.set_title('c  Similar current fits permit very different state trajectories', loc='left', fontsize=10.1, fontweight='bold')
    ax.legend([p1, p2], [p1.get_label(), p2.get_label()], frameon=False, fontsize=7.5, loc='upper right')

    ax = axs[1, 1]
    names = ['field_weibull', 'thermal_rate', 'power_damage']; labels = ['Maximum-field\nthresholds', 'Thermal-rate\naccumulation', 'Power-driven\nloss']
    x = np.arange(3)
    for dx, prefix, color, label in [(-.18, 'five_fold', '#0072B2', 'Five held-pair folds'),
                                   (.18, 'chronological', '#D55E00', 'Train 1–35; predict 36–53')]:
        values = [terminal['groups'][n]['all53'][prefix+'__RESET_outward_conditioned']['return_rmse_decade'] for n in names]
        ax.bar(x+dx, values, .34, color=color, label=label)
        for xi, value in zip(x+dx, values):ax.text(xi, value+.012, f'{value:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(ylabel='Withheld RESET return RMSE (decade)', ylim=(0, .73))
    ax.set_title('d  Forward circuit tests do not identify one winning mechanism', loc='left', fontsize=10.0, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('State-history evidence: one-way loss is not a universal RESET description', fontsize=14, fontweight='bold', y=.975)
    fig.text(.085, .915, '254 measured negative loops; history-model fits use the 53 loops at gate 1.1 V. Production M13 is unchanged.', fontsize=9)
    fig.text(.085, .035,
             'a/b: terminal comparisons, no transistor inverse. Reversal rule includes a conservative zero-reference perturbation; this is not an error model.\n'
             'c: descriptive parameter profiles, not confidence intervals. d: target outward RESET initializes state; target return is withheld.\n'
             'The three histories share effective transport envelopes. Device identities/order across gates and dwell/gap times remain unconfirmed.', fontsize=8, color='#444444')
    out = ROOT/'analysis/figures'; out.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):fig.savefig(out/f'30_reset_history_and_order.{ext}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':run()
