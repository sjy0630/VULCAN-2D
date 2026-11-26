"""Plot only saved, independently reviewed training results; no model calls."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HELPER = ROOT/'analysis/coupled_training_calibration_2026-11-04/figures/gen_fig_calibration.py'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(HELPER) == '9d72303aed0b6a4e2a02ccf28db9ba07de802c49e974c72dcc4ca4970a7d0417'
spec = importlib.util.spec_from_file_location('saved_plot_helpers', HELPER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    stage = HERE.parent
    report_path = stage/'rate_controls_run/RESULT.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'finite_rate_controls_and_final_checks_finished'
    assert report['numerical_final_candidate_verified'] and report['sources_unchanged']
    row = next(row for row in report['requests'] if row['request_id'] == 6)
    assert row['status'] == 'complete_valid' and row['parameters'] == report['selected_coarse_candidate']['parameters']
    cases = {
        'disabled': helpers.load_case(ROOT/'analysis/coupled_native_training_2026-11-04/baseline_run_v2/disabled_refined'),
        'q2': helpers.load_case(ROOT/'analysis/coupled_training_calibration_2026-11-04/calibration_run/cases/035_best_refined'),
        'selected': helpers.load_case(Path(row['artifact_prefix'])),
    }
    helpers.same_observations(cases)
    metrics = {key: helpers.weighted_metrics(case)[0] for key, case in cases.items()}
    helpers.publication_style(plt)
    outputs = []
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5))
    panels = [('1r', 'set'), ('1r', 'reset'), ('1t1r', 'set'), ('1t1r', 'reset')]
    counts = []
    for k, (ax, panel) in enumerate(zip(axes.flat, panels)):
        count = nrecords = 0
        for (top, cid, phase), arrays in cases['selected']['curves'].items():
            if (top, phase) != panel:
                continue
            nrecords += 1
            v, mask = arrays['command_voltage_v'], arrays['original_scoring_mask']
            count += int(mask.sum())
            ax.scatter(v[mask], abs(arrays['observed_current_a'][mask]), s=3, c='#444444', alpha=.22, linewidths=0, rasterized=True)
            current = abs(arrays['current_a'])
            ax.plot(v, np.where(current > 0, current, np.nan), color='#0072B2', alpha=.36, linewidth=.8)
        ax.set_yscale('log')
        ax.set_title(f'({chr(97+k)}) {panel[0].upper()} {panel[1].upper()}', loc='left')
        ax.set_xlabel('Command voltage (V)')
        ax.set_ylabel(r'$|I|$ (A)')
        ax.text(.02, .025, f'{nrecords} records; {count:,} scored points\nWeighted RMSE = {metrics["selected"][k]:.3f} decade',
                transform=ax.transAxes, va='bottom', fontsize=8, bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        counts.append(count)
    assert sum(counts) == 27023
    fig.suptitle('First-order recovery: continuous trajectories on all training records', y=.99, fontsize=12)
    fig.legend(handles=[Line2D([],[],color='#444444',marker='o',linestyle='None',markersize=3,label='Experiment: original scored points'),
                        Line2D([],[],color='#0072B2',label='Selected q1 model: all records')],
               loc='upper center', bbox_to_anchor=(.5,.95), ncol=2)
    fig.text(.5,.04,'1R cycles 1–3 and 5–15; 1T1R cycles 4–35. Three fixed segment entries; no held-out evaluation.',ha='center',fontsize=8)
    fig.text(.5,.015,'No smoothing, selected best cycle or current floor. Bare SET retains its three equally weighted subgroups.',ha='center',fontsize=7.5)
    fig.tight_layout(rect=(0,.075,1,.89))
    outputs += helpers.export_figure(fig,HERE,'fig_q1_all_training')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0,3.7))
    methods = [('disabled','a-disabled control','#777777','//'),('q2','Selected q2 calibration','#E69F00','\\'),('selected','Selected q1 controls','#0072B2','')]
    x, width = np.arange(4), .24
    peak = max(float(values.max()) for values in metrics.values())
    for j,(key,label,color,hatch) in enumerate(methods):
        bars = ax.bar(x+(j-1)*width,metrics[key],width*.94,label=label,color=color,hatch=hatch,edgecolor='white',zorder=3)
        for bar,value in zip(bars,metrics[key]):
            ax.text(bar.get_x()+bar.get_width()/2,value+.02*peak,f'{value:.3f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,['1R SET','1R RESET','1T1R SET','1T1R RESET'])
    ax.set_ylabel('Original weighted RMSE (decade)')
    ax.set_ylim(0,peak*1.16)
    ax.grid(False,axis='x')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3)
    fig.suptitle('Identical training data, masks and original block weights',y=.99,fontsize=12)
    fig.text(.5,.035,'All cases use saved refined trajectories. The q1 comparison changes recovery order, A and kappa; not an order-only ablation.',ha='center',fontsize=7.5)
    fig.tight_layout(rect=(0,.07,1,.86))
    outputs += helpers.export_figure(fig,HERE,'fig_q1_weighted_errors')
    plt.close(fig)
    inputs = [report_path,HELPER,Path(__file__)]+[case[name] for case in cases.values() for name in ('json_path','npz_path')]
    metadata = dict(status='saved_training_figures_only',input_sha256={str(p):sha(p) for p in inputs},
        output_sha256={str(p):sha(p) for p in outputs},original_scored_points=sum(counts),
        block_RMSE_decade={key:value.tolist() for key,value in metrics.items()},ODEs=0,holdout_read=False)
    (HERE/'FIGURE_METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
