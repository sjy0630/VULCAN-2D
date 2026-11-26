"""Reproduce training-only calibration figures from terminal saved artifacts.

No model/kernel import, ODE, fitting, raw-data archive, held-out record or
current floor is used. All original scoring points and all supplied training
records are retained; model lines connect native samples without smoothing.
Requires only the project's existing NumPy and Matplotlib installations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
ROOT=STAGE.parents[1]
DEFAULT_BASELINES=ROOT/'analysis/coupled_native_training_2026-11-04/baseline_run_v2'
TRAIN={'1r':(1,2,3,5,6,7,8,9,10,11,12,13,14,15),'1t1r':tuple(range(4,36))}
PANELS=(('1r','set'),('1r','reset'),('1t1r','set'),('1t1r','reset'))
COUNTS={('1r','set'):2784,('1r','reset'):2799,('1t1r','set'):16000,('1t1r','reset'):5440}
GROUP_COUNTS=(730,27,2027,2799,16000,5440)
FIELDS=('command_voltage_v','observed_current_a','original_scoring_mask','current_a')
COLORS={'data':'#444444','disabled':'#777777','seed':'#E69F00','selected':'#0072B2'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def terminal_selected(run):
    report_path=run/'RESULT.json'
    report=read_json(report_path)
    # These fields are emitted by the runner's finalization, not its running
    # checkpoints. A partial search can qualify if its retained center has
    # actually completed and passed both declared final precisions.
    if ('source_sha256_after' not in report or 'n_completed_rounds' not in report
            or not report.get('sources_unchanged')
            or not report.get('numerical_final_candidate_verified')):
        raise RuntimeError('Refusing to plot: calibration is not terminal with a verified final candidate')
    matches=[row for row in report['requests'] if row.get('kind')=='final'
             and row.get('precision',{}).get('name')=='refined'
             and row.get('status')=='complete_valid' and row.get('eligible_for_selection')]
    if len(matches)!=1:
        raise ValueError('Terminal report must identify exactly one completed selected refined case')
    chosen=matches[0]
    if chosen['parameters']!=report['last_accepted_center']['parameters']:
        raise ValueError('Selected refined parameters differ from the last accepted search center')
    prefix=Path(chosen['artifact_prefix']).resolve()
    try:
        prefix.relative_to(run.resolve())
    except ValueError as exc:
        raise ValueError('Selected artifact prefix is not inside this saved calibration run') from exc
    return report,chosen,prefix,report_path


def load_case(prefix):
    json_path=Path(str(prefix)+'.json')
    npz_path=Path(str(prefix)+'.npz')
    payload=read_json(json_path)
    if payload['summary'].get('status')!='complete_original_training':
        raise ValueError('A plotted case is not a completed original-training evaluation')
    if payload['summary'].get('n_completed_phases')!=92 or payload['score']['n_residuals']!=27023:
        raise ValueError('A plotted case has changed the original training scope')
    if not payload.get('numerical_verification',{}).get('passed'):
        raise ValueError('A plotted case did not pass its saved native numerical checks')
    curves={}
    with np.load(npz_path,allow_pickle=False) as saved:
        for top,ids in TRAIN.items():
            for record in ids:
                for phase in ('set','reset'):
                    key=top,record,phase
                    tag=f'{top}_cycle_{record}_{phase}__'
                    arrays={name:saved[tag+name].copy() for name in FIELDS}
                    voltage=arrays['command_voltage_v']
                    observation=arrays['observed_current_a']
                    predicted=arrays['current_a']
                    mask=arrays['original_scoring_mask']
                    if (voltage.ndim!=1 or any(value.shape!=voltage.shape for value in arrays.values())
                            or mask.dtype!=np.bool_ or not np.all(np.isfinite(voltage))
                            or not np.all(np.isfinite(predicted))):
                        raise ValueError('Invalid native curve/mask in saved case '+str(key))
                    expected=(np.isfinite(abs(observation))&(abs(voltage)>1e-12)
                              &(abs(observation)>=(1e-11 if phase=='set' else 1e-12)))
                    if not np.array_equal(mask,expected) or np.any(abs(predicted[mask])<=0):
                        raise ValueError('Original observation-based mask or positive scored current changed')
                    curves[key]=arrays
        if saved['weighted_residual'].shape!=(27023,) or saved['raw_residual'].shape!=(27023,):
            raise ValueError('Saved original residual vectors do not have length 27023')
    return dict(payload=payload,curves=curves,json_path=json_path,npz_path=npz_path)


def same_observations(cases):
    reference=cases['selected']['curves']
    for name,case in cases.items():
        for key,values in reference.items():
            for field in ('command_voltage_v','observed_current_a','original_scoring_mask'):
                if not np.array_equal(values[field],case['curves'][key][field],equal_nan=True):
                    raise ValueError(f'{name} changes original {field}: {key}')


def weighted_metrics(case):
    """Recheck the archived four-phase objective from saved native arrays only."""
    values=[]
    counts=[]
    panels=[]
    for top,phase in PANELS:
        matrices=[case['curves'][top,record,phase] for record in TRAIN[top]]
        observed=abs(np.array([row['observed_current_a'] for row in matrices]))
        predicted=abs(np.array([row['current_a'] for row in matrices]))
        mask=np.array([row['original_scoring_mask'] for row in matrices])
        voltage=matrices[0]['command_voltage_v']
        if int(mask.sum())!=COUNTS[top,phase]:
            raise ValueError('Original phase scoring count differs')
        if (top,phase)==('1r','set'):
            outward=np.arange(len(voltage))<=np.argmax(abs(voltage))
            masks=(mask&(observed<.995e-3)&outward,
                   mask&(observed<.995e-3)&~outward,
                   mask&(observed>=.995e-3))
        else:
            masks=(mask,)
        mse=0.
        for selected in masks:
            counts.append(int(selected.sum()))
            errors=np.log10(predicted[selected])-np.log10(observed[selected])
            mse+=float(np.mean(errors**2))/len(masks)
        stored=float(case['payload']['score']['blocks'][f'{top}_{phase}']['weighted_total_mse'])
        if not math.isclose(mse,stored,rel_tol=2e-12,abs_tol=2e-14):
            raise ValueError('Saved phase-weighted MSE differs from original-mask recomputation')
        values.append(math.sqrt(stored))
        panels.append(dict(topology=top,phase=phase,n_records=len(TRAIN[top]),
            n_scored_points=int(mask.sum()),weighted_mse=stored,weighted_RMSE_decade=math.sqrt(stored)))
    if tuple(counts)!=GROUP_COUNTS:
        raise ValueError('Original six objective block counts changed')
    cost=.5*sum(value['weighted_mse'] for value in panels)
    if not math.isclose(cost,case['payload']['score']['verified_cost'],rel_tol=2e-12,abs_tol=2e-14):
        raise ValueError('Four block MSEs do not reproduce the archived original objective cost')
    return np.array(values),panels


def publication_style(plt):
    # academic-plotting data-chart workflow: colorblind-safe Okabe-Ito colors,
    # clear units, embedded vector fonts and 300-DPI raster companion exports.
    plt.rcParams.update({
        'font.family':'serif','font.serif':['DejaVu Serif'],
        'mathtext.fontset':'dejavuserif','font.size':9,
        'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9,
        'legend.fontsize':8.5,'legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.16,'grid.linewidth':.6,
        'axes.axisbelow':True,'lines.linewidth':.85,
        'pdf.fonttype':42,'ps.fonttype':42,
        'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    })


def export_figure(fig,output,stem):
    pdf=output/(stem+'.pdf')
    png=output/(stem+'.png')
    fig.savefig(pdf,metadata={'Creator':'gen_fig_calibration.py','CreationDate':None,'ModDate':None})
    fig.savefig(png,dpi=300,metadata={'Software':'gen_fig_calibration.py'})
    return [pdf,png]


def draw_all_training(plt,cases,metrics,report,output):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator
    fig,axes=plt.subplots(2,2,figsize=(8.0,6.5))
    plotting_counts=[]
    for index,((top,phase),ax) in enumerate(zip(PANELS,axes.flat)):
        records=TRAIN[top]
        point_alpha=.30 if top=='1r' else .16
        line_alpha=.42 if top=='1r' else .24
        total_points=0
        model_points=0
        model_zero_points=0
        all_y=[]
        for record in records:
            curve=cases['selected']['curves'][top,record,phase]
            v=curve['command_voltage_v']
            observed=abs(curve['observed_current_a'])
            predicted=abs(curve['current_a'])
            mask=curve['original_scoring_mask']
            # Exactly every original scoring observation, with no downsampling.
            ax.scatter(v[mask],observed[mask],s=5.0,color=COLORS['data'],
                       alpha=point_alpha,linewidths=0,zorder=2,rasterized=False)
            # Only exact-zero model currents cannot appear on a logarithmic
            # ordinate. NaN breaks those zero endpoints; no floor is inserted.
            line=np.where(predicted>0,predicted,np.nan)
            ax.plot(v,line,color=COLORS['selected'],alpha=line_alpha,
                    linewidth=.85,zorder=3,rasterized=False)
            total_points+=int(mask.sum())
            model_points+=int((predicted>0).sum())
            model_zero_points+=int((predicted==0).sum())
            all_y.extend((observed[mask],predicted[predicted>0]))
        ax.set_yscale('log')
        positive=np.concatenate(all_y)
        ax.set_ylim(float(positive.min())/1.5,float(positive.max())*1.7)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=7))
        ax.grid(True,which='major')
        label='1R' if top=='1r' else '1T1R'
        ax.set_title(f'({chr(97+index)}) {label} {phase.upper()}',loc='left',pad=8)
        ax.set_xlabel('Command voltage (V)')
        ax.set_ylabel(r'$|I|$ (A)')
        text=f'{len(records)} records; {total_points:,} scored points\nWeighted RMSE = {metrics["selected"][index]:.3f} decade'
        ax.text(.02,.03,text,transform=ax.transAxes,fontsize=7.5,va='bottom',ha='left',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2.5),zorder=5)
        plotting_counts.append(dict(topology=top,phase=phase,n_records=len(records),
            experiment_scored_points=total_points,model_nonzero_native_points=model_points,
            model_exact_zero_points_not_log_plotted=model_zero_points,
            experiment_alpha=point_alpha,model_alpha=line_alpha))
    handles=[Line2D([],[],marker='o',linestyle='None',markersize=4,color=COLORS['data'],label='Experiment: every original scoring point'),
             Line2D([],[],color=COLORS['selected'],linewidth=1.4,label='Selected refined model: every training record')]
    fig.suptitle('Continuous coupled-model trajectories on all original training records',fontsize=12,y=.995)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.963),ncol=2)
    fig.text(.5,.065,'Training only: 1R cycles 1–3 and 5–15; 1T1R cycles 4–35. No held-out evaluation.',
             ha='center',fontsize=8)
    fig.text(.5,.038,f'All 46 pairs / 92 phases / 27,023 scored points; {report["n_completed_rounds"]}/4 coordinate rounds completed. '
             'No curve smoothing or selected best cycle.',ha='center',fontsize=7.5)
    fig.text(.5,.012,'1R SET weighted RMSE gives equal weight to nonlimited outward, nonlimited return, and limited points.',
             ha='center',fontsize=7.3)
    fig.tight_layout(rect=(0,.095,1,.91),h_pad=1.6,w_pad=1.8)
    paths=export_figure(fig,output,'fig_calibration_all_training')
    plt.close(fig)
    return paths,plotting_counts


def draw_weighted_rmse(plt,metrics,output):
    fig,ax=plt.subplots(figsize=(8.0,3.9))
    x=np.arange(4)
    width=.235
    methods=(('disabled',r'$a$ disabled baseline','//'),
             ('seed','Coupled seed baseline','\\'),
             ('selected','Selected coupled calibration',''))
    maximum=max(float(values.max()) for values in metrics.values())
    for index,(key,label,hatch) in enumerate(methods):
        bars=ax.bar(x+(index-1)*width,metrics[key],width*.93,
            color=COLORS[key],label=label,hatch=hatch,edgecolor='white',linewidth=.7,zorder=3)
        for bar,value in zip(bars,metrics[key]):
            ax.text(bar.get_x()+bar.get_width()/2.,value+.015*maximum,f'{value:.3f}',
                    ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,('1R SET','1R RESET','1T1R SET','1T1R RESET'))
    ax.set_ylabel('Original phase-weighted log10 RMSE (decade)')
    ax.set_ylim(0,maximum*1.18)
    ax.grid(False,axis='x')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.21),ncol=3)
    fig.suptitle('Three models compared with the unchanged original training objective',fontsize=11.5,y=.995)
    fig.text(.5,.08,'All models: refined saved trajectories; 46 training pairs / 27,023 original residuals. No held-out results.',
             ha='center',fontsize=8)
    fig.text(.5,.047,r'Each bar is $\sqrt{\mathrm{weighted\ phase\ MSE}}$; 1R SET has three equally weighted subgroups.',
             ha='center',fontsize=8)
    fig.text(.5,.014,'The other three blocks use their original eligible points. No averages of per-cycle RMSE and no scatter subtraction.',
             ha='center',fontsize=7.5)
    fig.tight_layout(rect=(0,.13,1,.84))
    paths=export_figure(fig,output,'fig_calibration_weighted_rmse')
    plt.close(fig)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=STAGE/'calibration_run')
    parser.add_argument('--baselines',type=Path,default=DEFAULT_BASELINES)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args(argv)
    run,baseline,output=args.run.resolve(),args.baselines.resolve(),args.output.resolve()
    report,selected,prefix,report_path=terminal_selected(run)
    cases={'disabled':load_case(baseline/'disabled_refined'),
           'seed':load_case(baseline/'enabled_refined'),
           'selected':load_case(prefix)}
    same_observations(cases)
    metrics={}
    metric_metadata={}
    for name,case in cases.items():
        metrics[name],metric_metadata[name]=weighted_metrics(case)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    publication_style(plt)
    output.mkdir(parents=True,exist_ok=True)
    paths,counts=draw_all_training(plt,cases,metrics,report,output)
    paths+=draw_weighted_rmse(plt,metrics,output)
    inputs=[report_path]+[case[key] for case in cases.values() for key in ('json_path','npz_path')]
    metadata=dict(status='figures_generated_from_terminal_saved_training_only',
        script_path=str(Path(__file__).resolve()),script_sha256=sha(Path(__file__)),
        numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        inputs_sha256={str(path):sha(path) for path in inputs},
        outputs_sha256={str(path):sha(path) for path in paths},
        selected_request_id=selected['request_id'],selected_parameters=selected['parameters'],
        selected_original_cost=cases['selected']['payload']['score']['verified_cost'],
        calibration_terminal_status=report['status'],completed_coordinate_rounds=report['n_completed_rounds'],
        training_records={key:list(values) for key,values in TRAIN.items()},
        curve_plot_counts=counts,phase_weighted_metrics=metric_metadata,
        all_original_scored_points_plotted=sum(row['experiment_scored_points'] for row in counts)==27023,
        no_downsampling=True,no_smoothing=True,no_best_cycle_selection=True,
        model_exact_zero_points_are_line_breaks_not_current_floors=True,
        experimental_mask_unchanged=True,holdout_data_read=False,model_imports=0,ODE_calls=0,
        baseline_labels=['a disabled','coupled seed'],all_comparisons_use_refined_saved_cases=True,
        interpretation='training fit comparison only; bounded search is not full convergence or microscopic proof')
    (output/'fig_calibration_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for path in paths:
        print(path)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
