#!/usr/bin/env python3
"""Plot saved refined calibration arrays only; no forward model or fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PHASES = ('set4', 'reset4', 'set5')
STEM = 'fig_calibration_iv_old_vs_final'
SOURCES = {
    'old': (STAGE/'verify_old_refined_chain.npz', STAGE/'requests/002_verify_old_refined.json'),
    'request57': (STAGE/'verify_request57_refined_chain.npz', STAGE/'requests/004_verify_request57_refined.json'),
    'final': (STAGE/'final_refined_chain.npz', STAGE/'requests/102_final_refined.json'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    traces, summaries, hashes, decoded, metrics = {}, {}, {}, {}, {}
    for label, (array_path, summary_path) in SOURCES.items():
        hashes[str(array_path)] = sha256(array_path)
        hashes[str(summary_path)] = sha256(summary_path)
        summary = json.loads(summary_path.read_text())
        assert summary['status'] == 'completed_chain'
        assert summary['precision'] == {'name': 'refined', 'rtol': 1e-8, 'atol': 1e-9, 'max_step': .01}
        assert summary['method'] == 'Radau'
        traces[label], metrics[label], decoded[label] = {}, {}, []
        summaries[label] = summary
        with np.load(array_path, allow_pickle=False) as archive:
            for phase in PHASES:
                a = {}
                for key in ('command_voltage_v', 'observed_current_a', 'original_scoring_mask', 'current_a'):
                    full_key = f'{phase}__{key}'
                    a[key] = archive[full_key].copy()
                    decoded[label].append(full_key)
                v, observed, mask = a['command_voltage_v'], abs(a['observed_current_a']), a['original_scoring_mask']
                prediction = abs(a['current_a'])
                expected_n = 170 if phase == 'reset4' else 500
                floor = 1e-12 if phase == 'reset4' else 1e-11
                original = np.isfinite(observed) & (observed >= floor) & (abs(v) > 1e-12)
                assert mask.dtype == np.bool_
                np.testing.assert_array_equal(mask, original)
                assert int(mask.sum()) == expected_n
                low = mask & (abs(v) <= .2+1e-12)
                assert int(low.sum()) == 20
                assert np.all(np.isfinite(prediction[mask])) and np.all(prediction[mask] > 0.)
                residual = np.log10(prediction[mask]/observed[mask])
                rmse = float(np.sqrt(np.mean(residual**2)))
                saved = summary['scores']['phase_metrics'][phase]['all_eligible']['RMSE_decade']
                assert abs(rmse-saved) < 1e-12
                metrics[label][phase] = {
                    'n_scored': expected_n, 'low_scored_n': 20, 'RMSE_decade': rmse,
                    'low20_RMSE_decade': float(np.sqrt(np.mean(np.log10(prediction[low]/observed[low])**2))),
                    'saved_refined_RMSE_decade': saved,
                }
                traces[label][phase] = a
                if label != 'old':
                    for key in ('command_voltage_v', 'observed_current_a', 'original_scoring_mask'):
                        np.testing.assert_array_equal(a[key], traces['old'][phase][key])

    result_path = STAGE/'bounded_coordinate_check.json'
    hashes[str(result_path)] = sha256(result_path)
    terminal = json.loads(result_path.read_text())
    assert terminal['search_requests'] == 96
    assert terminal['search_outcome']['status'] == 'request_budget_exhausted'
    assert terminal['search_outcome']['grid_stopped'] is False
    assert terminal['source_files_unchanged'] is True
    assert terminal['numerical_validation_passed'] is True
    assert terminal['all_three_goals_passed'] is True

    # Okabe--Ito orange/blue and line styles distinguish the two saved models.
    orange, blue, observed_color, ink = '#E69F00', '#0072B2', '#747D85', '#243240'
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
        'axes.labelsize': 10.5, 'axes.titlesize': 11.5, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#78838D',
        'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 5.4), sharey=True)
    fig.subplots_adjust(left=.082, right=.982, bottom=.29, top=.70, wspace=.17)
    fig.suptitle('Continuous three-phase calibration', x=.082, y=.97,
                 ha='left', fontsize=15, fontweight='bold')
    old_total = summaries['old']['scores']['all1170_RMSE_decade']
    new_total = summaries['final']['scores']['all1170_RMSE_decade']
    fig.text(.082, .913,
             f'Shared material law and one history initial state; pooled RMSE: {old_total:.4f} → {new_total:.4f} decade',
             ha='left', fontsize=10.4)
    handles = [
        Line2D([], [], color=observed_color, marker='o', linestyle='none', markersize=4,
               markeredgewidth=0, label='Experiment: all scored points'),
        Line2D([], [], color=orange, linestyle=(0, (4, 2)), linewidth=1.7, label='Old control (refined)'),
        Line2D([], [], color=blue, linestyle='-', linewidth=1.7, label='Selected model (refined)'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.075, .87), ncol=3,
               frameon=False, fontsize=10, columnspacing=2.3, handlelength=2.3)

    for ax, phase in zip(axes, PHASES):
        base, selected = traces['old'][phase], traces['final'][phase]
        v, mask = base['command_voltage_v'], base['original_scoring_mask']
        # Preserve native temporal ordering through outward and return branches;
        # zeros are excluded by the original score mask, never by plot clipping.
        ax.scatter(v[mask], abs(base['observed_current_a'][mask]), s=6,
                   color=observed_color, alpha=.58, linewidths=0, zorder=2)
        ax.plot(v[mask], abs(base['current_a'][mask]), color=orange,
                linestyle=(0, (4, 2)), linewidth=1.65, zorder=3)
        ax.plot(v[mask], abs(selected['current_a'][mask]), color=blue,
                linewidth=1.55, zorder=4)
        ax.set_yscale('log')
        ax.set_ylim(1e-11, 1.1e-4)
        ax.grid(axis='y', which='major', color='#DFE3E6', linewidth=.6, alpha=.8)
        ax.yaxis.set_major_locator(LogLocator(base=10., numticks=9))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='y', which='minor', length=2, color='#ADB4BB')
        if phase == 'reset4':
            ax.set_xlim(-1.75, .05)
            ax.set_xticks([-1.5, -1., -.5, 0.])
        else:
            ax.set_xlim(-.1, 5.1)
            ax.set_xticks([0., 1., 2., 3., 4., 5.])
        count = metrics['old'][phase]['n_scored']
        ax.set_title(f'{phase.upper()} · {count} calibration points', pad=28)
        old_rmse = metrics['old'][phase]['RMSE_decade']
        new_rmse = metrics['final'][phase]['RMSE_decade']
        ax.text(.5, 1.04, f'RMSE: {old_rmse:.4f} → {new_rmse:.4f} decade',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Command voltage (V)', labelpad=7)
    axes[0].set_ylabel(r'Current magnitude $|I|$ (A)', labelpad=8)

    fig.text(.082, .181, 'Search budget exhausted (96/96 requests): convergence was not established.',
             fontsize=10, fontweight='bold', ha='left')
    fig.text(.082, .123,
             r'All original 500 / 170 / 500 scored points are retained, including 20 points at $0<|V|\leq0.2$ V per phase.',
             fontsize=9.5, ha='left')
    fig.text(.082, .065,
             'The three-phase non-degradation requirement is a modeling goal, not a noise band or confidence interval.',
             fontsize=9.5, ha='left', color='#536171')

    outputs = {}
    for suffix in ('png', 'pdf'):
        path = HERE/f'{STEM}.{suffix}'
        fig.savefig(path, dpi=300, facecolor='white')
        outputs[path.name] = sha256(path)
    plt.close(fig)
    for path, expected in hashes.items():
        assert sha256(path) == expected
    metadata = {
        'figure': 'Three original calibration I-V phases, saved old vs final refined models',
        'source_sha256': hashes, 'script_sha256': sha256(Path(__file__)), 'output_sha256': outputs,
        'software': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        'size_inches': [11.3, 5.4], 'png_dpi': 300, 'phase_order': list(PHASES),
        'array_keys_decoded': decoded, 'plotted_model_labels': ['old', 'final'],
        'request57_retained_in_metadata_only': True, 'metrics_from_saved_arrays': metrics,
        'pooled1170_RMSE_decade': {label:s['scores']['all1170_RMSE_decade'] for label,s in summaries.items()},
        'mean_phase_MSE_decade2': {label:s['scores']['mean_phase_MSE_decade2'] for label,s in summaries.items()},
        'search_requests': 96, 'grid_stopped': False, 'search_status': terminal['search_outcome']['status'],
        'refined_validation_passed': terminal['numerical_validation_passed'],
        'all_three_modeling_goals_passed': terminal['all_three_goals_passed'],
        'scoring_formula': 'sqrt(mean(log10(abs(I_model)/abs(I_observed))**2)) over each original scoring mask',
        'points_plotted': {'set4':500,'reset4':170,'set5':500},
        'low_voltage_points_retained': {'set4':20,'reset4':20,'set5':20},
        'observations': 'Identical saved original native arrays across all three models; positive stored magnitudes; signed command voltage retained.',
        'display': 'Observed points and model samples are not downsampled. Original zero-voltage exclusions are preserved. No confidence or uncertainty band.',
        'interpretation': [
            'All three phases are calibration; none is a holdout or an independent prediction.',
            'Selected final model meets all three non-degradation goals relative to old control.',
            'Final does not dominate request57 in every phase; request57 scores are retained explicitly.',
            'Search request budget exhausted; coordinate-grid stop was not attained.',
            'Modeling non-degradation bounds do not represent experimental noise or confidence intervals.',
            'No extension data/model/score, original holdout, new forward solve, inverse, ODE or fit was used.',
        ],
    }
    (HERE/f'{STEM}.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'outputs':outputs,'points':[500,170,500],'low_points':[20,20,20]},indent=2))


if __name__ == '__main__':
    main()
