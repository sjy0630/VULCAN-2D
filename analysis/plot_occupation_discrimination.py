"""Plot saved occupation comparisons, without refitting/selecting models."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_occupation_transport import extract_curves
from vulcan2d.occupation_transport import log10_shape


def run():
    r=json.loads((ROOT/'analysis/occupation_comparison.json').read_text())
    c=json.loads((ROOT/'analysis/occupation_circuit_profiles.json').read_text())
    fits=r['groups']['linear_current']['3.0']['candidates']
    curves,_,_=extract_curves('linear_current',3.)
    names=['power','occupation','full_bias_level','rollover']
    colors={'power':'#777777','occupation':'#0072B2','full_bias_level':'#009E73','rollover':'#D55E00'}
    labels={'power':'Power (1 shape parameter)','occupation':'Local partial-bias level (2)',
            'full_bias_level':'Full-bias asymmetric level (2)','rollover':'Empirical rollover (3)'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    fig.subplots_adjust(left=.09,right=.96,top=.86,bottom=.20,wspace=.3,hspace=.43)
    ax=axs[0,0]
    for curve in curves:
        anchor=np.interp(.3,curve.voltage_v,curve.log10_current_a)
        ax.plot(curve.voltage_v,10**(curve.log10_current_a-anchor),color='#CCCCCC',alpha=.28,lw=.6)
    v=np.geomspace(.15,3,300)
    for name in names:
        pars=fits[name]['all_data_fit']['parameters']
        shape=log10_shape(v,name,pars)-log10_shape(.3,name,pars)
        ax.plot(v,10**shape,color=colors[name],lw=2,label=labels[name])
    ax.set(xscale='log',yscale='log',xlabel='Internal voltage (V)',ylabel=r'$I(v)/I(0.3\,V)$',xlim=(.15,3))
    ax.set_title('a  Common shape across 53 SET returns',loc='left',fontweight='bold')
    ax.legend(fontsize=8,frameon=False)
    ax=axs[0,1]
    for name in names:
        pars=fits[name]['all_data_fit']['parameters']
        slope=np.gradient(log10_shape(v,name,pars),np.log10(v))
        ax.plot(v,slope,color=colors[name],lw=2,label=labels[name])
    audit=json.loads((ROOT/'analysis/transistor_gate_interpolation_audit.json').read_text())
    ws=audit['return_shape_sensitivity']['linear_current']['window_results'][:5]
    ax.scatter([np.mean(x['window_V']) for x in ws],[x['shared_exponent'] for x in ws],
               color='black',s=28,zorder=5,label='Observed window slopes')
    ax.axhline(1,color='#999999',ls='--',lw=1)
    ax.set(xlabel='Internal voltage (V)',ylabel=r'$d\log I/d\log v$',xlim=(.15,3),ylim=(0,2.5))
    ax.set_title('b  A single level misses part of the curvature',loc='left',fontweight='bold')
    ax.legend(fontsize=8,frameon=False,loc='upper right')
    ax=axs[1,0]
    # Consistent full-target fitting protocol, then contrasting extrapolation.
    protocols=['loco_autonomous_branch','loco_voltage_extrapolation']
    x=np.arange(4)
    for j,name in enumerate(names):
        internal=fits[name]['evaluation']
        terminal=c['terminal_groups']['3.0'][name]
        values=[internal[protocols[0]]['equal_cycle_rmse_decade'],terminal[protocols[0]]['equal_cycle_rmse_decade'],
                internal[protocols[1]]['equal_cycle_rmse_decade'],terminal[protocols[1]]['equal_cycle_rmse_decade']]
        ax.bar(x+(j-1.5)*.19,values,width=.18,color=colors[name],label=labels[name])
    ax.set_xticks(x,['Held cycle\ninternal','Held cycle\nterminal','Low-only\ninternal','Low-only\nterminal'])
    ax.set(ylabel='Current RMSE (decade)',ylim=(0,.31))
    ax.axvline(1.5,color='#BBBBBB',lw=1)
    ax.set_title('c  Terminal limits can hide extrapolation errors',loc='left',fontweight='bold')
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    ax=axs[1,1]
    for upper,key,label,color,style in [('3.0','full','0.15–3 V','#0072B2','-'),
        ('4.2','full','0.15–4.2 V','#D55E00','-'),('3.0','low','0.15–1.2 V','#777777','--')]:
        ps=c['background_profiles'][upper][key]
        ax.plot([10**p['log10_gamma'] for p in ps],[p['rmse_decade'] for p in ps],color=color,
                ls=style,marker='o',ms=3,lw=1.5,label=label)
    ax.set(xscale='log',xlabel=r'Background ratio at reference, $\gamma$',ylabel='Profiled shape RMSE (decade)',ylim=(.057,.115))
    ax.set_title('d  Background evidence depends on the window',loc='left',fontweight='bold')
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    fig.suptitle('SET return: a stationary channel can bend the curve, but is not yet identified',
                 fontsize=14,fontweight='bold',y=.97)
    fig.text(.5,.915,'Fixed slow structure | explicit carrier rate balance | held-cycle and voltage-range tests',ha='center',fontsize=11,color='#444444')
    lines=['a–b: Full-window shared shapes; grey traces are observed curves normalized at 0.3 V. These panels are descriptive fits.',
           'c: Held-cycle amplitudes use training data only; low-only fits use ≤1.2 V and a target low-voltage amplitude anchor.',
           'Terminal predictions solve the measured transistor series circuit. All scores use the same observed validity masks.',
           'd: Partial-bias kernel: profiles re-optimize the other two parameters. These are not statistical confidence intervals.']
    for k,line in enumerate(lines):fig.text(.075,.102-k*.023,line,fontsize=8.6,color='#444444')
    out=ROOT/'analysis/figures'
    for suffix in ('png','pdf'):
        dest=out/f'25_occupation_discrimination.{suffix}';fig.savefig(dest,dpi=250,bbox_inches='tight',pad_inches=.15);print(dest)
    plt.close(fig)


if __name__=='__main__':run()
