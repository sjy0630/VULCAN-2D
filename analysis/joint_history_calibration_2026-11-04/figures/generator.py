"""Render terminal saved joint-history results only; no physical model imports."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/vulcan-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/vulcan-fontcache')
import numpy as np

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
PHASES=('set4','reset4','set5','reset5','set6')
CALIBRATION=PHASES[:3]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    outputs={ext:HERE/f'fig_continuous_joint_calibration.{ext}' for ext in ('png','pdf')}
    meta_path=HERE/'fig_continuous_joint_calibration_metadata.json'
    table_path=HERE/'RMSE_TABLE.md'
    if any(p.exists() for p in [*outputs.values(),meta_path,table_path]):
        raise FileExistsError('Preserve previous outputs; no silent figure overwrite')
    report_path=STAGE/'bounded_joint_calibration.json'
    report_hash=sha(report_path);report=json.loads(report_path.read_text())
    if not (report.get('terminal_result') is True and report['status']=='completed_bounded_joint_diagnostic'
            and report['numerical_validation_passed'] is True and report['source_files_unchanged'] is True):
        raise ValueError('Require terminal numerically validated joint result')
    if not (report['full_unified_model_achieved'] is False and report['full_joint_contract_evaluated'] is False
            and report['original_holdout_accessed'] is False and report['continuation_used_for_selection'] is False):
        raise ValueError('Limited training scope changed')
    control=next(r for r in report['requests'] if r['label']=='control_Radau')
    final=report['final_chains']['refined']
    if control['status']!='completed_chain' or final['status']!='completed_chain':
        raise ValueError('Saved comparison chains incomplete')
    sources={str(report_path):report_hash};data={};summaries={};metrics={};panels=[];handoffs={}
    for label,record,phases in (('old',control,CALIBRATION),('new',final,PHASES)):
        data[label]={};summaries[label]={};metrics[label]={};handoffs[label]=[]
        previous=None
        for phase in phases:
            entry=next(r for r in record['phase_results'] if r['phase']==phase)
            loaded={}
            for kind,extension in (('arrays','npz'),('summary','json')):
                item=entry['artifacts'][kind];path=Path(item['path']).resolve()
                if path.parent!=STAGE/'phase_trials' or path.suffix!='.'+extension or sha(path)!=item['sha256']:
                    raise ValueError('Saved artifact identity/hash changed: '+str(path))
                sources[str(path)]=item['sha256']
                if kind=='arrays':
                    with np.load(path,allow_pickle=False) as archive:
                        loaded={key:archive[key].copy() for key in ('command_voltage_v','current_a','state_u',
                            'path_voltage_s','observed_current_a','original_scoring_mask')}
                else:summary=json.loads(path.read_text())
            if summary['status']!='completed' or not summary['phase_integrity']['passed']:
                raise ValueError('Saved phase integrity not passed')
            if label=='new' and not (summary['phase_integrity']['complete_integrity_passed']
                    and summary['independent_native_readout']['passed']):
                raise ValueError('Refined phase lacks complete independent validation')
            a=loaded;v=a['command_voltage_v'];mask=a['original_scoring_mask']
            expected=(173,170) if phase.startswith('reset') else (503,500)
            if mask.dtype!=np.bool_ or (len(v),int(mask.sum()))!=expected:
                raise ValueError('Original scored point counts changed')
            if any(a[k].shape!=v.shape for k in a) or not all(np.all(np.isfinite(a[k])) for k in a):
                raise ValueError('Nonfinite or mismatched saved native arrays')
            if np.any(v*a['current_a']<0.) or np.any(abs(a['current_a'][mask])<=0.):
                raise ValueError('Invalid signed model current')
            if v[0]!=0. or v[-1]!=0. or a['path_voltage_s'][0]!=0.:
                raise ValueError('Actual phase command/path endpoints changed')
            if abs(a['state_u'][0]-summary['initial_u'])>1e-12 or abs(a['state_u'][-1]-summary['terminal_u'])>1e-12:
                raise ValueError('Saved state endpoint mismatch')
            if previous is not None:
                join=abs(summary['initial_u']-previous['terminal_u'])
                if join>1e-12:raise ValueError('Saved continuous phase state mismatch')
                handoffs[label].append({'to_phase':phase,'incoming_u':summary['initial_u'],'join_error_u':join})
            previous=summary
            outward=np.arange(len(v))<=np.argmax(abs(v))
            a['outward_mask']=mask&outward;a['return_mask']=mask&~outward
            if not np.array_equal(a['outward_mask']|a['return_mask'],mask) or np.any(a['outward_mask']&a['return_mask']):
                raise ValueError('Every scored point must appear in exactly one branch')
            computed={}
            for group,use in (('all_eligible',mask),('outward',a['outward_mask']),('return',a['return_mask']),
                              ('low_abs_le_0p2',mask&(abs(v)<=.2+1e-12))):
                rmse=float(np.sqrt(np.mean(np.log10(abs(a['current_a'][use])/abs(a['observed_current_a'][use]))**2)))
                saved=summary['metrics'][group]
                if saved['n']!=int(use.sum()) or not math.isclose(rmse,saved['RMSE_decade'],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Saved phase score does not match original arrays')
                computed[group]={'n':int(use.sum()),'RMSE_decade':rmse}
            data[label][phase]=a;summaries[label][phase]=summary;metrics[label][phase]=computed
    for phase in CALIBRATION:
        for key in ('command_voltage_v','observed_current_a','original_scoring_mask','path_voltage_s'):
            if not np.array_equal(data['old'][phase][key],data['new'][phase][key]):
                raise ValueError('Old/new data identity differs: '+phase+'/'+key)
    if sha(report_path)!=report_hash:raise ValueError('Terminal report changed while loading')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator,MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'axes.titlesize':11,'legend.fontsize':9,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'savefig.dpi':300})
    old_color,new_color,obs_color='#D55E00','#0072B2','#555555'
    fig,axes=plt.subplots(2,3,figsize=(14.4,8.8))
    fig.subplots_adjust(left=.065,right=.988,top=.79,bottom=.18,hspace=.57,wspace=.28)
    fig.suptitle('Limited training diagnostic: continuous joint calibration',x=.065,y=.975,ha='left',fontsize=19,weight='bold')
    fig.text(.065,.929,'One shared signed equation and one initial state. Curve errors remain; full 1R/1T1R validation is outstanding.',fontsize=11)
    fig.text(.065,.895,'Calibration: SET4 → RESET4 → SET5. Continuation checks: RESET5 → SET6, excluded from parameter selection.',fontsize=10.5)
    handles=[Line2D([],[],color=old_color,lw=1.8,label='Old control (first 3 phases)'),
             Line2D([],[],color=new_color,lw=1.8,label='Joint fit, refined'),
             Line2D([],[],color=obs_color,marker='o',mfc='none',mew=.7,ls='none',ms=4,label='Observed outward'),
             Line2D([],[],color=obs_color,marker='^',mfc='none',mew=.7,ls='none',ms=4,label='Observed return')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.06,.872),ncol=4,handlelength=2.6,columnspacing=1.8)
    ranges={}
    for family in ('set','reset'):
        vals=[]
        for label in ('old','new'):
            for phase,a in data[label].items():
                if phase.startswith(family):
                    m=a['original_scoring_mask'];vals.extend([abs(a['current_a'][m]),abs(a['observed_current_a'][m])])
        values=np.concatenate(vals)
        ranges[family]=[10**(math.floor(math.log10(float(min(values))))-.08),
                        10**(math.ceil(math.log10(float(max(values))))+.08)]
    locations={'set4':(0,0),'reset4':(0,1),'set5':(0,2),'reset5':(1,0),'set6':(1,1)}
    for index,phase in enumerate(PHASES):
        ax=axes[locations[phase]];a=data['new'][phase];family='reset' if phase.startswith('reset') else 'set'
        for branch,style,marker in (('outward','-','o'),('return','--','^')):
            use=a[branch+'_mask'];x=abs(a['command_voltage_v'][use])
            if phase in CALIBRATION:
                old=data['old'][phase]
                ax.plot(x,abs(old['current_a'][use]),color=old_color,ls=style,lw=1.45,zorder=2)
            ax.plot(x,abs(a['current_a'][use]),color=new_color,ls=style,lw=1.7,zorder=3)
            ax.scatter(x,abs(a['observed_current_a'][use]),s=10.5,marker=marker,facecolors='none',
                       edgecolors=obs_color,linewidths=.55,zorder=4)
            panels.append({'phase':phase,'branch':branch,'native_scored_indices':np.flatnonzero(use).tolist(),
                'n':int(use.sum()),'display_current':'absolute magnitude; source sign preserved and checked'})
        ax.set_yscale('log');ax.set_ylim(*ranges[family]);ax.set_xlim(0,1.025*float(max(abs(a['command_voltage_v']))))
        ax.axvspan(0,.2,color='#E69F00',alpha=.08,zorder=0)
        ax.grid(axis='y',which='major',alpha=.16,lw=.6)
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=8));ax.xaxis.set_major_locator(MaxNLocator(4))
        role='calibration' if phase in CALIBRATION else 'training continuation'
        title=f'({chr(97+index)}) {phase.upper()} · {role}'
        new_rmse=metrics['new'][phase]['all_eligible']['RMSE_decade']
        subtitle=(f'RMSE: {metrics["old"][phase]["all_eligible"]["RMSE_decade"]:.3f} → {new_rmse:.3f} decade'
                  if phase in CALIBRATION else f'RMSE: {new_rmse:.3f} decade; fixed parameters')
        ax.set_title(title+'\n'+subtitle,loc='left',fontsize=10.7,pad=10)
        ax.set_xlabel(r'$|V_{\mathrm{command}}|$ (V)');ax.set_ylabel(r'$|I|$ (A)')
    ax=axes[1,2];offset=0.;segments=[]
    for index,phase in enumerate(PHASES):
        a=data['new'][phase];x=offset+a['path_voltage_s'];end=float(x[-1]);center=(offset+end)/2.
        if index%2==0:ax.axvspan(offset,end,color='#777777',alpha=.045,zorder=0)
        if index>0:ax.axvline(offset,color='#999999',lw=.7,ls=':',zorder=0)
        ax.plot(x,a['state_u'],color=new_color,lw=1.7,zorder=3)
        ax.scatter([x[0],x[-1]],[a['state_u'][0],a['state_u'][-1]],s=15,color=new_color,zorder=4)
        ax.text(center,.98,phase.upper(),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        segments.append({'phase':phase,'global_path_start_v':offset,'global_path_stop_v':end,
                         'initial_u':float(a['state_u'][0]),'terminal_u':float(a['state_u'][-1])})
        offset=end
    old_offset=0.
    for phase in CALIBRATION:
        a=data['old'][phase];x=old_offset+a['path_voltage_s']
        ax.plot(x,a['state_u'],color=old_color,lw=1.4,alpha=.9,zorder=2);old_offset=float(x[-1])
    ax.set_xlim(-.4,offset+.4);ax.set_ylim(bottom=0.)
    ax.set_title('(f) Continuous state across all five phases\nNo phase or cycle re-anchoring',loc='left',fontsize=10.7,pad=10)
    ax.set_xlabel('Cumulative command-voltage distance (V)',fontsize=9)
    ax.set_ylabel(r'$u=-\ln q$');ax.grid(axis='y',alpha=.16,lw=.6);ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.text(.065,.105,'Every original scored observation is shown: 500 / 170 / 500 calibration points; 170 / 500 continuation points. No subsampling.',fontsize=10)
    fig.text(.065,.074,'I–V lines: solid outward / dashed return. Shading: |V| ≤ 0.2 V. Low-current errors retained; continuation is not an original holdout.',fontsize=10)
    fig.text(.065,.043,'q is a transport-state coordinate, not measured occupancy. Voltage distance is not physical time. No combined five-phase objective is reported.',fontsize=9.5)
    for path in outputs.values():fig.savefig(path,dpi=300)
    plt.close(fig)
    rows=['# Saved-result RMSE comparison','',
          'All values are log10-current RMSE in decades, on the original eligible masks. Each phase uses its full native waveform and actual incoming state.','',
          '| Phase | Role | Scored points | Old Radau control | Joint refined | Joint low-bias RMSE (20 points) |',
          '|---|---|---:|---:|---:|---:|']
    for phase in PHASES:
        n=metrics['new'][phase]['all_eligible']['n'];new=metrics['new'][phase]['all_eligible']['RMSE_decade']
        old=f'{metrics["old"][phase]["all_eligible"]["RMSE_decade"]:.9f}' if phase in CALIBRATION else 'Not simulated'
        low=metrics['new'][phase]['low_abs_le_0p2']['RMSE_decade']
        rows.append(f'| {phase.upper()} | '+('Calibration' if phase in CALIBRATION else 'Training continuation')+f' | {n} | {old} | {new:.9f} | {low:.9f} |')
    rows.extend(['','Calibration only: equal-phase mean squared error (the selection objective): '
        f'{control["scores"]["mean_phase_MSE_decade2"]:.9f} → {final["scores"]["mean_phase_MSE_decade2"]:.9f} decade².',
        'Calibration only: point-weighted 1170-point RMSE: '
        f'{control["scores"]["all1170_RMSE_decade"]:.9f} → {final["scores"]["all1170_RMSE_decade"]:.9f} decade.',
        '', 'SET4 accuracy worsens while RESET4 and SET5 improve. This is a limited compromise from bounded calibration, not completion of the unified 1R/1T1R model or identification of a microscopic mechanism.',
        'RESET5 and SET6 did not enter candidate selection. No old continuation baseline or combined five-phase objective is invented.'])
    table_path.write_text('\n'.join(rows)+'\n')
    metadata={'status':'rendered_from_terminal_saved_control_and_refined_results',
        'sources_sha256':sources,'generator_sha256':sha(Path(__file__)),
        'numerical_validation_in_source_report':report['numerical_validation_passed'],
        'source_report_integrity_flag':report['source_files_unchanged'],
        'provenance_scope':'Figure hashes verify the saved report and arrays; any execution-source timing audit is separate.',
        'new_KCL_solves':0,'new_state_inversions':0,'new_ODEs':0,'new_fits':0,
        'original_holdout_accessed':False,'bare_arrays_read':False,'full_unified_model_achieved':False,
        'data_roles':{'calibration':list(CALIBRATION),'training_continuation':list(PHASES[3:]),
                      'continuation_used_for_selection':False,'combined_five_phase_objective_reported':False},
        'native_scored_points_shown':{p:metrics['new'][p]['all_eligible']['n'] for p in PHASES},
        'branch_point_membership':panels,'metrics':metrics,'actual_state_handoffs':handoffs,
        'five_phase_state_segments':segments,'y_limits_a':ranges,
        'rendering':{'PNG_dpi':300,'PDF_vector':True,'PDF_fonttype':42,'font':'DejaVu Sans',
                     'size_inches':[14.4,8.8],'matplotlib':matplotlib.__version__,'numpy':np.__version__,
                     'old_control_color':old_color,'joint_refined_color':new_color,
                     'outward_line':'solid','return_line':'dashed','downsampling':False},
        'outputs':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in [*outputs.values(),table_path]]}
    meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':metadata['status'],'outputs':[str(p) for p in [*outputs.values(),meta_path,table_path]],
                     'refined_phase_RMSE':{p:metrics['new'][p]['all_eligible']['RMSE_decade'] for p in PHASES}},indent=2))

if __name__=='__main__':main()
