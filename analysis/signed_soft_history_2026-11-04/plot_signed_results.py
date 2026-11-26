"""Plot only terminal, validated saved RESET4 -> SET5 results; never solve physics.

Preparation compiles this source without importing it. Rendering requires root's
explicit terminal confirmation via --terminal-authorized. Incomplete/unvalidated
results stop before matplotlib import or figure creation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_artifact(item, expected_name):
    path = Path(item['path']).resolve()
    if path != HERE / expected_name:
        raise ValueError(f'Unexpected final artifact path: {path}')
    if sha(path) != item['sha256']:
        raise ValueError(f'Final artifact hash changed: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-authorized', action='store_true')
    args = parser.parse_args()
    if not args.terminal_authorized:
        raise SystemExit('Prepared only; wait for root terminal confirmation.')
    report_path = HERE / 'bounded_signed_trial.json'
    report_hash = sha(report_path)
    report = json.loads(report_path.read_text())
    if report['status'] != 'completed_bounded_signed_attempt_no_full_model_claim':
        raise SystemExit('No figure: the signed attempt is incomplete or failed.')
    if report.get('numerical_validation_passed') is not True:
        raise SystemExit('No figure: the signed attempt did not pass numerical validation.')
    if report.get('full_unified_model_achieved') is not False:
        raise ValueError('This plot must preserve the explicit limited model claim.')
    if report.get('original_holdout_accessed') is not False or report.get('SET5_used_for_b_selection') is not False:
        raise ValueError('Training-only, RESET4-only selection scope changed.')
    unchanged = report.get('source_files_unchanged', {})
    if not unchanged or not all(unchanged.values()):
        raise ValueError('Final source immutability check missing or failed.')
    pair = next(p for p in report['final_pairs'] if p['precision']['name'] == 'refined')
    if pair.get('validation_passed') is not True or pair['status'] != 'completed_pair':
        raise ValueError('The final refined pair is not validated.')

    import numpy as np
    data, summaries, sources, phase_metrics = {}, {}, {}, {}
    for phase, count, scored in (('reset4', 173, 170), ('set5', 503, 500)):
        saved = pair['phases'][phase]
        if saved.get('validation_passed') is not True:
            raise ValueError(f'{phase} final validation missing.')
        for kind, extension in (('summary', 'json'), ('arrays', 'npz')):
            item = saved['artifacts'][kind]
            path = checked_artifact(item, f'final_refined_{phase}.{extension}')
            sources[str(path)] = item['sha256']
            if kind == 'summary':
                summaries[phase] = json.loads(path.read_text())
            else:
                with np.load(path, allow_pickle=False) as archive:
                    data[phase] = {key: archive[key].copy() for key in (
                        'command_voltage_v', 'path_voltage_s', 'current_a', 'state_u',
                        'observed_current_a', 'original_scoring_mask')}
        a, s = data[phase], summaries[phase]
        v, mask = a['command_voltage_v'], a['original_scoring_mask']
        if mask.dtype != np.bool_ or v.shape != (count,) or int(mask.sum()) != scored:
            raise ValueError('Native/scored point identity or mask datatype changed.')
        if any(a[key].shape != (count,) for key in a):
            raise ValueError('Inconsistent native array shapes.')
        if not all(np.all(np.isfinite(a[key])) for key in a if key != 'original_scoring_mask'):
            raise ValueError('Final arrays contain nonfinite values.')
        if np.any(a['current_a'] * v < 0):
            raise ValueError('Signed model current fails passivity.')
        if np.any(abs(a['current_a'][mask]) <= 0) or np.any(abs(a['observed_current_a'][mask]) <= 0):
            raise ValueError('Log-current scoring requires positive amplitudes.')
        outward = np.arange(count) <= np.argmax(abs(v))
        a['outward_mask'], a['return_mask'] = mask & outward, mask & ~outward
        if np.any(a['outward_mask'] & a['return_mask']) or not np.array_equal(
                a['outward_mask'] | a['return_mask'], mask):
            raise ValueError('The two figure panels must retain every scored point exactly once.')
        phase_metrics[phase] = {}
        for name, use in (('all_eligible', mask), ('outward', a['outward_mask']), ('return', a['return_mask'])):
            rmse = float(np.sqrt(np.mean(np.log10(abs(a['current_a'][use]) / abs(a['observed_current_a'][use])) ** 2)))
            recorded = s['original_mask_diagnostics'][name]
            if recorded['n'] != int(use.sum()) or not math.isclose(rmse, recorded['RMSE_decade'], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'{phase}/{name} saved diagnostics mismatch.')
            phase_metrics[phase][name] = {'n': int(use.sum()), 'RMSE_decade': rmse}
        if s.get('final_phase_validation_passed') is not True or not s.get('no_conditioned_anchor') or not s.get('no_hidden_state_reset'):
            raise ValueError('Validated continuous no-anchor phase metadata missing.')
    reset, follow = data['reset4'], data['set5']
    join_error = abs(float(reset['state_u'][-1]) - float(follow['state_u'][0]))
    entry_error = abs(float(reset['state_u'][0]) - report['plan']['RESET4_incoming_u'])
    if max(join_error, entry_error) > 1e-12:
        raise ValueError('Saved continuous state handoff differs from frozen entry.')
    if sha(report_path) != report_hash:
        raise ValueError('Main report changed during reading; terminal state not stable.')

    output = HERE / 'figures'
    output.mkdir(exist_ok=True)
    outputs = [output / f'{name}.{ext}' for name in ('signed_reset4_set5_comparison', 'continuous_state_handoff')
               for ext in ('png', 'pdf')]
    metadata_path = output / 'signed_figures_metadata.json'
    if any(path.exists() for path in [*outputs, metadata_path]):
        raise FileExistsError('Preserve existing figures; do not silently overwrite.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, MaxNLocator
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'legend.fontsize': 9, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 300, 'figure.dpi': 120,
    })
    observed_color, model_color = '#444444', '#0072B2'
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7), sharey='row')
    fig.subplots_adjust(left=.10, right=.985, bottom=.15, top=.82, hspace=.58, wspace=.16)
    fig.suptitle('Signed candidate: RESET4 → SET5 curve errors remain', x=.10, y=.975, ha='left', fontsize=17, weight='bold')
    b = report['selected_b']['b']
    fig.text(.10, .932, f'Frozen positive model; b = {b:.6g} from RESET4 only. Final refined trajectories.', fontsize=10)
    fig.text(.10, .903, 'Numerically stable; accurate unified explanation not achieved. Conditional training-chain check.', fontsize=10, color='#9E3B00')
    panel_meta = []
    for row, phase in enumerate(('reset4', 'set5')):
        a = data[phase]
        eligible = a['original_scoring_mask']
        all_y = np.r_[abs(a['observed_current_a'][eligible]), abs(a['current_a'][eligible])]
        low, high = float(all_y.min()), float(all_y.max())
        yrange = (10 ** (math.floor(math.log10(low)) - .10), 10 ** (math.ceil(math.log10(high)) + .10))
        for col, branch in enumerate(('outward', 'return')):
            ax, use = axes[row, col], a[branch + '_mask']
            x = abs(a['command_voltage_v'][use])
            ax.plot(x, abs(a['current_a'][use]), color=model_color, linewidth=1.8, label='Model', zorder=2)
            ax.scatter(x, abs(a['observed_current_a'][use]), s=11, marker='o', facecolors='none',
                       edgecolors=observed_color, linewidths=.6, label='Observed', zorder=3)
            ax.set_yscale('log'); ax.set_ylim(*yrange)
            ax.set_xlim(0, float(max(abs(a['command_voltage_v']))) * 1.025)
            ax.axvspan(0, .2, color='#E69F00', alpha=.075, zorder=0)
            ax.grid(axis='y', which='major', alpha=.16, linewidth=.6)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            metric = phase_metrics[phase][branch]
            role = 'calibration' if phase == 'reset4' else 'next-phase check'
            letter = chr(ord('a') + 2 * row + col)
            ax.set_title(f'({letter}) {phase.upper()} {branch} · {role}', loc='left', pad=11)
            ax.text(.97, .04, f'n = {metric["n"]}  |  RMSE = {metric["RMSE_decade"]:.3f} decade',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                    bbox={'boxstyle': 'round,pad=.25', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88})
            ax.set_xlabel(r'$|V_{\mathrm{command}}|$ (V)')
            if col == 0:
                ax.set_ylabel(r'$|I|$ (A)')
            panel_meta.append({'panel': letter, 'phase': phase, 'branch': branch,
                'native_scored_indices': np.flatnonzero(use).tolist(), **metric, 'y_limits_a': list(yrange)})
    features = {row['identifier']: row for row in summaries['reset4']['single_record_feature_references']}
    peak = features['1t1r.reset.outward.peak_current']
    tail = features['1t1r.reset.return.I_0p2V']
    onset = next(row for row in summaries['set5']['outward_current_crossings'] if row['threshold_a'] == 1e-6)
    notes = [(axes[0, 0], f'Peak: {peak["predicted_value"] * 1e6:.2f} vs {peak["observed_value"] * 1e6:.2f} µA'),
             (axes[0, 1], f'At 0.2 V: {tail["predicted_value"] * 1e9:.2f} vs {tail["observed_value"] * 1e9:.2f} nA'),
             (axes[1, 0], f'1 µA onset: {onset["predicted"]["first_at_or_above_voltage_v"]:.2f} vs {onset["observed"]["first_at_or_above_voltage_v"]:.2f} V')]
    for ax, note in notes:
        ax.set_title(ax.get_title(loc='left') + '\n' + note + ' (model / observed)',
                     loc='left', fontsize=9.7, pad=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, .96), ncol=2)
    fig.text(.10, .065, 'All 170 RESET4 and 500 SET5 eligible observations are shown; no point subsampling. Shading: |V| ≤ 0.2 V.', fontsize=9)
    fig.text(.10, .035, 'SET5 inherits the simulated RESET4 terminal state; no 0.2 V re-anchor. Not an original holdout or full unified-model validation.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'signed_reset4_set5_comparison.{ext}', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    fig.subplots_adjust(left=.10, right=.985, bottom=.26, top=.75)
    fig.suptitle('Continuous state handoff in the inaccurate candidate', x=.10, y=.96, ha='left', fontsize=15, weight='bold')
    first_s = reset['path_voltage_s']
    offset = float(first_s[-1])
    second_s = offset + follow['path_voltage_s']
    ax.plot(first_s, reset['state_u'], color='#D55E00', linewidth=1.9, label='RESET4')
    ax.plot(second_s, follow['state_u'], color='#0072B2', linewidth=1.9, label='SET5')
    ax.scatter([0, offset, second_s[-1]], [reset['state_u'][0], reset['state_u'][-1], follow['state_u'][-1]],
               s=28, color='#222222', zorder=4)
    ax.axvline(offset, color='#888888', linewidth=.8, linestyle='--')
    ax.text(.01, 1.10, f'SET4 saved terminal u = {reset["state_u"][0]:.6g}', transform=ax.transAxes, fontsize=9)
    ax.text(.99, 1.10, f'RESET4 → SET5 state mismatch: {join_error:.1e}', transform=ax.transAxes, ha='right', fontsize=9)
    ax.set_xlabel('Cumulative command-voltage distance from RESET4 start (V)')
    ax.set_ylabel(r'$u=-\ln q$')
    ax.set_xlim(-.1, second_s[-1] * 1.015)
    ax.grid(axis='y', alpha=.16); ax.legend(loc='best', ncol=2)
    fig.text(.10, .09, 'q is a transport-state coordinate, not measured trap occupancy. Voltage distance is not elapsed physical time.', fontsize=9)
    fig.text(.10, .035, 'Training-internal continuous history check; no inferred wait time, state reset or SET5 fitting.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'continuous_state_handoff.{ext}', dpi=300)
    plt.close(fig)

    metadata = {
        'status': 'rendered_from_terminal_numerically_validated_saved_arrays',
        'report_path': str(report_path), 'report_sha256': report_hash,
        'plot_source_sha256': sha(Path(__file__)), 'final_sources_sha256': sources,
        'numerical_validation_passed': True, 'new_roots_ODEs_fits_or_inversions': 0,
        'original_holdout_accessed': False, 'bare_records_accessed': False,
        'SET5_used_for_selection': False, 'full_unified_model_achieved': False,
        'scope': 'RESET4 calibration and training-internal SET5 continuous next-phase check',
        'accuracy_statement': 'Substantial RESET4 peak/tail and SET5 onset errors remain; accurate unified explanation not achieved',
        'displayed_diagnostics': {'RESET4_peak_current': peak, 'RESET4_return_0p2_current': tail, 'SET5_1uA_first_crossing': onset},
        'b': b, 'all_original_scored_points_retained': {'reset4': 170, 'set5': 500},
        'panels': panel_meta, 'phase_metrics': phase_metrics,
        'continuity': {'frozen_SET4_terminal_u': report['plan']['RESET4_incoming_u'],
            'RESET4_initial_u': float(reset['state_u'][0]), 'RESET4_terminal_u': float(reset['state_u'][-1]),
            'SET5_initial_u': float(follow['state_u'][0]), 'SET5_terminal_u': float(follow['state_u'][-1]),
            'phase_join_error_u': join_error, 'no_conditioned_anchor': True},
        'rendering': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
            'PNG_dpi': 300, 'PDF_fonttype': 42, 'model_color': model_color,
            'current_display': 'absolute amplitudes; physical signs checked before rendering'},
        'outputs': [{'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in outputs],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': metadata['status'], 'outputs': [str(p) for p in outputs],
                      'metadata': str(metadata_path)}, indent=2))


if __name__ == '__main__':
    main()
