"""Plot the exact class constraint, held-pair failure and measured scan contrast."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
from diagnose_cross_polarity_physics import read_pairs


def run():
    fit = json.loads((ROOT/'analysis/parallel_path_comparison.json').read_text())
    bound = json.loads((ROOT/'analysis/parallel_rectification_bound.json').read_text())
    memory = json.loads((ROOT/'analysis/scan_memory_contrast.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))
    fig.subplots_adjust(left=.075, right=.97, top=.88, bottom=.13, wspace=.29, hspace=.45)
    colors = ['#D55E00', '#CC79A7', '#0072B2']
    ax = axs[0, 0]
    for cid, color in zip((1, 2, 3), colors):
        rows = [r for r in bound['first3_primary_curve'] if r['cycle_id'] == cid]
        v = [r['hbn_voltage_V'] for r in rows]
        ax.plot(v, [r['ratio'] for r in rows], color=color, lw=1.7, label=f'Cycle {cid}')
        ax.plot(v, [r['zero_reference_ratio_upper'] for r in rows], color=color, lw=1, ls='--')
    lower = [r['bound_lower'] for r in rows]
    ax.plot(v, lower, color='black', lw=1.6, label='300 K class lower bound')
    ax.fill_between(v, 1e-5, lower, color='#DDDDDD', alpha=.7)
    ax.set(xlabel='Matched internal |V| (V)', ylabel='Reverse / forward current magnitude',
           yscale='log', xlim=(.1, .3), ylim=(1e-5, .1))
    ax.set_title('a  A spectrum of independent fixed-rate levels is insufficient', loc='left', fontsize=10.4, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.text(.102, 1.45e-5, 'Dashed: upper zero-reference perturbation; shaded: excluded', fontsize=7.5)

    ax = axs[0, 1]
    for phase, color, label in [('set', '#0072B2', 'SET: return / outward'),
                               ('reset', '#D55E00', 'RESET: outward / return')]:
        rows = [r for r in memory['cycles'] if r['phase'] == phase and r['window'] == 'middle']
        ax.plot([r['cycle_id'] for r in rows], 10**np.array([r['mean_log10_history_contrast'] for r in rows]),
                '.-', color=color, ms=4, lw=.8, label=label)
    ax.axhline(1, color='black', ls=':', lw=1); ax.axvspan(.5, 3.5, color='#DDDDDD', alpha=.6)
    ax.set(yscale='log', xlim=(.5, 53.5), ylim=(.7, 2000), xlabel='Paired cycle index',
           ylabel='Geometric mean same-polarity scan contrast')
    ax.set_title('b  The early reverse sweeps carry little hysteresis', loc='left', fontsize=10.4, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.text(20, 1.35, 'Matched applied |V| = 0.50–0.80 V', fontsize=8)

    ax = axs[1, 0]
    names = ['band_amplitude', 'two_band_fixed', 'two_band_conserved', 'two_band_free', 'two_level_free']
    labels = ['One band\namplitude', 'Two bands\nfixed ratio', 'Two bands\none state', 'Two bands\ntwo states', 'Two levels\ntwo states']
    x = np.arange(len(names))
    for dx, key, color, label in [(-.18, 'all53_joint_descriptive', '#999999', 'Joint target-data fit'),
                                 (.18, 'five_fold_SET_only_forecast', '#0072B2', 'Held pair; state from SET only')]:
        vals = [fit['candidates'][name]['groups']['all53'][key]['negative_rmse_decade'] for name in names]
        ax.bar(x+dx, vals, .34, color=color, label=label)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(ylabel='RESET-initial RMSE (decade), all 53 pairs', ylim=(0, .86))
    ax.set_title('c  Better joint fits do not give better SET-to-RESET forecasts', loc='left', fontsize=10.2, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = axs[1, 1]
    pairs = read_pairs('1.1')
    for cid, color in [(3, '#0072B2'), (4, '#D55E00')]:
        pair = pairs[cid-1]
        for phase, ls in [('set', '-'), ('reset', '--')]:
            raw = pair[phase]; v = np.abs(raw['voltage']); i = np.abs(raw['current'])
            if phase == 'set':
                start = len(v)-1-int(np.argmax(v[::-1])); mask = np.arange(len(v)) >= start
            else:mask = np.arange(len(v)) <= int(np.argmax(v))
            mask &= (v >= .08) & (v <= .6)
            ax.plot(v[mask], i[mask], color=color, ls=ls, lw=1.5, label=f'Cycle {cid} '+('SET return' if phase == 'set' else 'RESET outward'))
    ax.set(yscale='log', xlabel='Applied |V| (V)', ylabel='Measured terminal |I| (A)', xlim=(.08, .6), ylim=(2e-10, 1e-5))
    ax.set_title('d  Similar positive returns can precede different reverse reads', loc='left', fontsize=10.2, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    fig.suptitle('Low-bias constraints on parallel-path transport and shared state', fontsize=14, fontweight='bold', y=.975)
    fig.text(.075, .925, 'Target: 1.1 V gate, all 53 ordered pairs. Conditional model-class test and descriptive scan audit; no microscopic identification.', fontsize=9)
    fig.text(.075, .035,
             'Bound assumes one frozen state, independent levels, voltage-independent positive weights/rates and passive linear voltage division.\n'
             'Zero-reference perturbations are sensitivity checks, not error bars. Panels b/d use terminal voltages; a/c use de-embedded coordinates.\n'
             'Held-pair forecasts use the measured target SET return; these are not autonomous full-cycle predictions.', fontsize=8, color='#444444')
    out = ROOT/'analysis/figures'; out.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):fig.savefig(out/f'29_parallel_path_state_evidence.{ext}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':run()
