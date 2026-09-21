"""Render saved transport comparisons; performs no fitting or model selection."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    report=json.loads((HERE/"transport_candidate_comparison.json").read_text())
    audit=json.loads((HERE/"transistor_gate_interpolation_audit.json").read_text())
    groups=report['groups']['primary']['signal_guard']
    candidates=['ohmic','power','pf_free','pf_constrained','schottky_free','schottky_constrained','hopping']
    branches=['set_pre','set_connected','reset_initial','reset_hrs']
    values=np.array([[groups[b]['candidates'][m]['evaluation']['loco_low_read_conditioned_high_voltage']['equal_cycle_rmse_decade']
                      for m in candidates] for b in branches])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(12,8.5))
    grid=fig.add_gridspec(2,2,height_ratios=[1,1.15],left=.16,right=.92,top=.86,bottom=.22,
                         hspace=.52,wspace=.36)
    ax=fig.add_subplot(grid[0,:])
    heat=ax.imshow(values,cmap='magma_r',norm=LogNorm(vmin=.05,vmax=1.2),aspect='auto')
    ax.set_xticks(range(7),['Ohmic','Power','PF free','PF bounded','Contact free','Contact bounded','Hopping'])
    ax.set_yticks(range(4),['SET pre (53)','SET return (53)','RESET initial (51)','RESET HRS (53)'])
    for row in range(4):
        for col in range(7):
            v=values[row,col]
            ax.text(col,row,f'{v:.3f}',ha='center',va='center',fontsize=10,
                    color='white' if v>.24 else 'black')
    cbar=fig.colorbar(heat,ax=ax,pad=.015,fraction=.03)
    cbar.set_label('RMSE (decade)');cbar.set_ticks([.05,.1,.2,.5,1]);cbar.set_ticklabels(['0.05','0.1','0.2','0.5','1'])
    ax.set_title('a  Predict high-voltage current from low-voltage reads',loc='left',fontweight='bold',pad=12)
    ax=fig.add_subplot(grid[1,0])
    labels=['SET pre','RESET HRS']
    for candidate,color,dx,label in [('pf_free','#D55E00',-.08,'PF free'),('schottky_free','#0072B2',.08,'Contact free')]:
        eps=[groups[b]['candidates'][candidate]['all_data_shape_fit']['conditional_physics']['apparent_epsilon_r']
             for b in ('set_pre','reset_hrs')]
        ax.scatter(np.arange(2)+dx,eps,s=60,color=color,label=label,zorder=4)
        for i,v in enumerate(eps):
            ax.annotate(f'{v:.2f}',(i+dx,v),xytext=(0,8),textcoords='offset points',ha='center',color=color,fontsize=10)
    ax.axhspan(2,8,color='#A1A7AE',alpha=.20,label='Assumed bound 2–8')
    ax.axhline(3.1,color='#767D85',lw=1,ls=':')
    ax.set(yscale='log',ylim=(1.4,80),xlim=(-.5,1.5),ylabel=r'Conditional apparent $\epsilon_r$')
    ax.set_xticks([0,1],labels);ax.set_yticks([2,4,8,20,40]);ax.set_yticklabels(['2','4','8','20','40'])
    ax.set_title('b  Parameter consistency at 300 K, 6 nm',loc='left',fontweight='bold',pad=12)
    ax.legend(fontsize=8,loc='upper left',frameon=False)
    ax=fig.add_subplot(grid[1,1])
    styles=[('linear_current','#222222','o','Current linear'),('sqrt_current','#009E73','s','Square root'),
            ('log_current','#0072B2','^','Log current'),('pchip_current','#CC79A7','D','Cubic (PCHIP)')]
    for method,color,marker,label in styles:
        rows=audit['return_shape_sensitivity'][method]['window_results']
        x=np.array([np.mean(r['window_V']) for r in rows]); y=np.array([r['shared_exponent'] for r in rows])
        ax.plot(x,y,color=color,marker=marker,ms=4.5,lw=1.3,label=label)
    ax.axhline(1,color='#777777',lw=1,ls='--')
    ax.axvspan(2,3,color='#A1A7AE',alpha=.17)
    ax.set(xlim=(.1,3.85),ylim=(.4,2.15),xlabel='Internal voltage window midpoint (V)',
           ylabel=r'Local power exponent, $d\log I/d\log v$')
    ax.set_title('c  SET return is not a fixed power law',loc='left',fontweight='bold',pad=12)
    ax.legend(fontsize=8,frameon=False,loc='upper right')
    ax.grid(axis='y',alpha=.18)
    fig.suptitle('Transport evidence: plausible parameters are necessary, but not sufficient',y=.975,
                 fontsize=15,fontweight='bold')
    fig.text(.5,.92,'1.1-V gate | measured transistor constraints | internal, retrospective discrimination',ha='center',fontsize=11,color='#444444')
    footnotes=[
        'a: Shape is trained on other cycles at low voltage; the target low reads set amplitude. This is not an autonomous full-cycle score.',
        'b: Full-window fits; uniform field and temperature are assumed. The dotted line (3.1) is a different h-BN sample, not this device.',
        'c: Four positive-gate interpolation rules; all 53 cycles have local exponent below 1 in the 2–3 V window under every rule.',
        'Bounded kernels assume dielectric constant 2–8. Windows and signal guards are shared; no microscopic species is identified.'
    ]
    for i,line in enumerate(footnotes):fig.text(.10,.122-i*.028,line,fontsize=8.8,color='#444444')
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):
        path=out/f'24_transport_discrimination.{suffix}';fig.savefig(path,dpi=260,bbox_inches='tight',pad_inches=.15);print(path)
    plt.close(fig)


if __name__=='__main__':main()
