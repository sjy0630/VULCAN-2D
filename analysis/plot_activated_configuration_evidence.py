"""Native full-cycle predictions, retained state, and an honest reference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_activated_configuration import assemble

ROOT = Path(__file__).resolve().parents[1]


def run():
    data = assemble(); records = {}
    for split in ('all_descriptive', 'chronological'):
        for saddle in ('single', 'polarizable'):
            path = ROOT/'analysis'/f'activated_configuration_{split}_{saddle}_reconciled.json'
            r = json.loads(path.read_text()); assert r['status'] == 'complete'
            records[split, saddle] = (r, np.load(path.with_suffix('.npz')))
    best = min((records['all_descriptive', s] for s in ('single', 'polarizable')),
               key=lambda x: x[0]['training_scores']['rmse_decade'])
    record, z = best
    ref = json.loads((ROOT/'analysis/activated_configuration_reference.json').read_text())
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 240})
    fig, ax = plt.subplots(2, 3, figsize=(13., 7.4), constrained_layout=True)
    colors = ['#4477AA', '#CC6677', '#228833', '#AA3377']
    for j, phase in enumerate(('SET', 'RESET')):
        a = ax[0, j]
        for c, color in zip((1, 3, 4, 36), colors):
            mask = (data['cycle'] == c) & (data['voltage'] > 0 if j == 0 else data['voltage'] < 0)
            v = abs(data['voltage'][mask]); y = data['observed'][mask]; pred = z['current'][mask]
            a.semilogy(v, y, color=color, lw=.85, alpha=.7, label=f'Measured {c}')
            a.semilogy(v, pred, color=color, lw=1.3, ls='--')
        a.set(xlabel='Applied voltage magnitude (V)', ylabel='Current magnitude (A)',
              title=f'{chr(97+j)}  {phase}: solid measured, dashed model', ylim=(1e-12, 2e-4))
        if j == 0: a.legend(frameon=False, fontsize=8, ncol=2)
    def low_ratio(current):
        values = []
        for c in range(1, 54):
            logs = []
            for window in (2, 4):
                m = (data['cycle'] == c) & (data['window'] == window)
                logs.append(np.mean(np.log10(current[m])))
            values.append(10**(logs[1]-logs[0]))
        return values
    a = ax[0, 2]
    a.semilogy(range(1, 54), low_ratio(data['observed']), 'o', color='#222222', ms=3, label='Measured')
    a.semilogy(range(1, 54), low_ratio(z['current']), '-', color='#EE7733', label='Continuous prediction')
    a.set(xlabel='Pair index', ylabel='RESET outward / SET return current',
          title='c  Low-bias history: 0.08–0.30 V')
    a.axvline(3.5, ls=':', color='.7'); a.legend(frameon=False, fontsize=8)
    a = ax[1, 0]; mean = z['states'][:, 1]@z['quadrature_weights']; n = len(mean)//53
    coordinate = np.arange(len(mean))/n+1
    m = data['cycle'] <= 5
    a.plot(coordinate[m], mean[m], lw=1.2, color='#4477AA')
    for i in range(1, 6): a.axvline(i+503/n, color='.85', lw=.7)
    a.set(xlabel='Pair index + scan fraction (first five pairs)', ylabel='Mean q=1 probability',
          title='d  Configuration evolves without per-scan resets', ylim=(-.02, 1.02))
    a = ax[1, 1]
    cycle = np.arange(1, 54)
    for saddle, color in [('single', '#4477AA'), ('polarizable', '#EE7733')]:
        r, _ = records['all_descriptive', saddle]
        a.plot(cycle, [r['full_scores']['per_cycle'][str(c)] for c in cycle], color=color,
               label='Linear field' if saddle == 'single' else 'Quadratic saddle term')
    a.plot(cycle, [ref['all_descriptive_exact_floor']['per_cycle'][str(c)] for c in cycle],
           color='.35', ls=':', label='Repeating waveform optimum')
    a.set(xlabel='Pair index', ylabel='Eight-window RMSE (decade)', title='e  All-data descriptive errors')
    a.legend(frameon=False, fontsize=8)
    a = ax[1, 2]; positions = np.arange(2); width = .24
    for shift, saddle, color in [(-width, 'single', '#4477AA'), (0., 'polarizable', '#EE7733')]:
        values = [records['all_descriptive', saddle][0]['training_scores']['rmse_decade'],
                  records['chronological', saddle][0]['target_scores']['rmse_decade']]
        a.bar(positions+shift, values, width, color=color)
    a.bar(positions+width, [ref['all_descriptive_exact_floor']['rmse_decade'],
                           ref['first35_template_target_last18']['rmse_decade']], width, color='.55')
    a.set(xticks=positions, xticklabels=['All 53: descriptive', 'Last 18: predicted'],
          ylabel='Eight-window RMSE (decade)', title='f  Same native-voltage task; gray = template')
    fig.suptitle('Activated configurations: thermodynamic consistency does not establish predictive closure', fontsize=12)
    folder = ROOT/'analysis/figures'; folder.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'): fig.savefig(folder/f'39_activated_configuration_evidence.{suffix}')
    plt.close(fig)


if __name__ == '__main__': run()
