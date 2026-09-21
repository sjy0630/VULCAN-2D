"""Show fit gains together with the unresolved native cycle transition."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_activated_configuration import assemble

ROOT = Path(__file__).resolve().parents[1]


def run():
    d = assemble(); records = {}
    for split in ('all_descriptive', 'chronological'):
        for model in ('single_reconciled', 'single_wide', 'single_contact'):
            p = ROOT/'analysis'/f'retained_charge_configuration_{split}_{model}.json'
            r = json.loads(p.read_text()); assert r['status'] == 'complete'
            records[split, model] = (r, np.load(p.with_suffix('.npz')))
    # This descriptive selection never chooses the chronological parameters.
    r, z = min((records['all_descriptive', model] for model in ('single_reconciled', 'single_wide', 'single_contact')),
               key=lambda pair: pair[0]['training_scores']['rmse_decade'])
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 220})
    fig, axes = plt.subplots(2, 3, figsize=(13., 7.5), constrained_layout=True)
    colors = ['#0072B2', '#D55E00']
    for ax, window, title in [(axes[0, 0], 2, 'a  SET return: cycles 3 and 4'),
                               (axes[0, 1], 4, 'b  RESET outward: cycles 3 and 4')]:
        for c, color in zip((3, 4), colors):
            m = (d['cycle'] == c) & (d['window'] == window) & (abs(d['voltage']) >= .1-1e-10)
            x = abs(d['voltage'][m]); y = d['observed'][m]
            ax.semilogy(x, y, 'o-', color=color, ms=3, lw=.8, label=f'Measured {c}')
            ax.semilogy(x, z['current'][m], '--', color=color, lw=1.5, label=f'Model {c}')
        ax.set(title=title, xlabel='Applied voltage magnitude (V)', ylabel='Current magnitude (A)')
        ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[0, 2]
    for model, color in [('single_reconciled', '#009E73'), ('single_contact', '#CC79A7')]:
        _, zz = records['all_descriptive', model]; state = zz['states']; w = zz['quadrature_weights']
        end = [np.flatnonzero(d['cycle'] == c)[-1] for c in range(1, 54)]
        name = 'Fixed contact' if model == 'single_reconciled' else 'Refitted contact'
        ax.plot(range(1, 54), ((state[:, 2]+state[:, 3])@w)[end], color=color, label=name+' q')
        ax.plot(range(1, 54), ((state[:, 1]+state[:, 3])@w)[end], color=color, ls='--', label=name+' m')
    ax.set(title='c  Retained states after each pair', xlabel='Pair index', ylabel='Mean state probability', ylim=(-.02, 1.02))
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 0]
    for c, color in zip((4, 27), colors):
        m = (d['cycle'] == c) & (d['voltage'] < 0)
        ax.semilogy(abs(d['voltage'][m]), d['observed'][m], color=color, lw=.9, label=f'Measured {c}')
        ax.semilogy(abs(d['voltage'][m]), z['current'][m], color=color, ls='--', lw=1.3, label=f'Model {c}')
    ax.set(title='d  Complete ordinary RESET scans', xlabel='Applied voltage magnitude (V)', ylabel='Current magnitude (A)', ylim=(1e-11, 2e-4))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[1, 1]; x = np.arange(2); width = .19
    for j, (model, label, color) in enumerate([('previous', 'Fast side charge', '#999999'),
                                             ('single_reconciled', 'Retained side charge', '#0072B2'),
                                             ('single_wide', 'Wider energy bounds', '#009E73'),
                                             ('single_contact', 'Refitted contact', '#CC79A7')]):
        values = []
        for split in ('all_descriptive', 'chronological'):
            if model == 'previous':
                rr = json.loads((ROOT/'analysis'/f'activated_configuration_{split}_single_reconciled.json').read_text())
            else: rr = records[split, model][0]
            values.append(rr['training_scores' if split == 'all_descriptive' else 'target_scores']['rmse_decade'])
        ax.bar(x+(j-1.5)*width, values, width, label=label, color=color)
    ax.set(xticks=x, xticklabels=['All 53: descriptive', 'Last 18: predicted'],
           ylabel='Eight-window RMSE (decade)', title='e  Matched prediction tasks')
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 2]; contact = json.loads((ROOT/'analysis/retained_charge_configuration_contact_sensitivity.json').read_text())
    rows = contact['candidates']['all_descriptive']; gl = [row['ln_left_over_right'] for row in rows]
    ax.semilogy(gl, [row['RESET_outward_4_over_3'] for row in rows], 'o-', color='#0072B2', label='Model, amplitude compensated')
    ax.axhline(contact['observed_RESET_outward_4_over_3'], color='#D55E00', ls='--', label='Measured R4/R3')
    ax.set(title='f  Contact sweep: fixed-state I unchanged', xlabel='ln(left / right contact attempt rate)', ylabel='R4/R3 at 0.10–0.30 V', ylim=(.6, 500))
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Retained side charge: fit gains do not resolve the cycle-3 to cycle-4 transition', fontsize=12)
    folder = ROOT/'analysis/figures'; folder.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'): fig.savefig(folder/f'40_retained_charge_evidence.{suffix}')
    plt.close(fig)
    for _, zz in records.values(): zz.close()


if __name__ == '__main__': run()
