"""Show the raw recovery and a transport-form-independent circuit bound."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def run():
    order = json.loads((ROOT / 'analysis/reset_order_constraint.json').read_text())
    bounds = json.loads((ROOT / 'analysis/negative_reentry_load_bound.json').read_text())
    raw = next(r for r in order['raw_strong_examples'] if r['gate_V'] == 1.3 and r['cycle_id'] == 23)
    row = next(r for r in bounds['rows'] if r['gate_V'] == 1.3 and r['cycle_id'] == 23)
    event, bound = row['event'], row['nominal']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={'width_ratios': [1.5, 1]})
    fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.26, wspace=.28)
    v, i = np.abs(raw['voltage_v']), np.abs(raw['current_a']); apex = int(np.argmax(v))
    a.plot(v[:apex+1], i[:apex+1]*1e6, color='#0072B2', lw=1.6, label='Outward negative sweep')
    a.plot(v[apex+1:], i[apex+1:]*1e6, color='#D55E00', lw=1.6, label='Return sweep')
    a.scatter([event['applied_abs_voltage_after_V']], [bound['current_upper_without_enhancement_A']*1e6],
              marker='x', color='black', s=50, zorder=5)
    a.annotate('1.392 µA circuit bound', xy=(1.09, 1.392), xytext=(.64, .09), fontsize=9,
               arrowprops={'arrowstyle': '->', 'color': '#444444'})
    a.set(yscale='log', xlim=(.08, 1.5), ylim=(1e-4, 300), xlabel='Applied negative-voltage magnitude (V)',
          ylabel='Stored current magnitude (µA)')
    a.set_title('a  Raw measurement: drop, then recovery', loc='left', fontsize=11, fontweight='bold')
    a.legend(loc='lower right', frameon=False, fontsize=8)
    values = [event['current_before_A']*1e6, bound['current_upper_without_enhancement_A']*1e6,
              event['current_after_A']*1e6]
    b.bar(np.arange(3), values, color=['#999999', '#009E73', '#D55E00'], width=.65)
    for x, value in enumerate(values):
        b.text(x, value*1.16, f'{value:.3f}', ha='center', fontsize=10)
    b.set_xticks(np.arange(3), ['Before\n1.08 V', 'Upper bound\n1.09 V', 'Measured\n1.09 V'], fontsize=9)
    b.set(yscale='log', ylim=(.1, 300), ylabel='Current magnitude (µA)')
    b.set_title('b  Only a 0.01 V increase in bias', loc='left', fontsize=11, fontweight='bold')
    b.text(.05, .86, 'Measured / upper bound = 43.94', transform=b.transAxes, fontsize=9)
    fig.suptitle('A fixed nonlinear conductor plus the measured transistor cannot explain this recovery',
                 fontsize=12.5, fontweight='bold', y=.97)
    fig.text(.085, .865, 'Gate 1.3 V, trace 23  |  Bound uses only the before-point current and independent transistor data.', fontsize=10)
    fig.text(.085, .12, r'$I_{\mathrm{after}}\leq I_T\left[I_T^{-1}(I_{\mathrm{before}})+\Delta V_{\mathrm{app}}\right]$'
             '   if conducting ability does not increase.', fontsize=12)
    fig.text(.085, .035,
             'Conditional on quasistatic conduction, fixed transistor and nonnegative frozen-state differential conductance.\n'
             'The violation survives all 12 specified sensitivity checks (minimum 27.61×); these are not calibrated error bars.\n'
             'Structural changes, contact/charge-state changes and measurement/control history are not distinguished by this test.', fontsize=8.5, color='#444444')
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT / f'analysis/figures/31_recovery_load_bound.{ext}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    run()
