"""Render saved joint-network, charge-identifiability and state-information evidence."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_contact_networks import assemble_pairs
from vulcan2d.contact_network import log10_shape


def run():
    report=json.loads((ROOT/'analysis/contact_network_comparison.json').read_text())
    charge=json.loads((ROOT/'analysis/contact_charging_sensitivity.json').read_text())
    state=json.loads((ROOT/'analysis/contact_state_information.json').read_text())
    bound=json.loads((ROOT/'analysis/shared_amplitude_bound.json').read_text())
    pairs,_=assemble_pairs()
    with (ROOT/'analysis/cross_polarity_physics_eta_grid.csv').open(encoding='utf-8-sig') as f:
        reference={int(r['cycle_id']):float(r['set_current_A']) for r in csv.DictReader(f)
                   if r['gate_V']=='1.1' and r['lookup_branch']=='mean' and abs(float(r['hbn_voltage_V'])-.3)<1e-10}
    names=['fixed_level','coulomb_right','empirical_branches']
    colors={'fixed_level':'#888888','coulomb_right':'#0072B2','empirical_branches':'#D55E00'}
    labels={'fixed_level':'Fixed single level (2 parameters)','coulomb_right':'Finite-U two-site limit (5)','empirical_branches':'Empirical two branches (5)'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    fig.subplots_adjust(left=.09,right=.96,top=.86,bottom=.19,wspace=.3,hspace=.44)
    for branch,ax in zip(['positive','negative'],axs[0]):
        for pair in pairs:
            v=getattr(pair,branch+'_voltage_v');y=getattr(pair,branch+'_log10_current_a')
            ax.plot(abs(v),10**y/reference[pair.cycle_id],color='#D55E00' if pair.cycle_id<=3 and branch=='negative' else '#BBBBBB',
                    alpha=.7 if pair.cycle_id<=3 and branch=='negative' else .28,lw=.8)
        v=np.geomspace(.15,3,250) if branch=='positive' else -np.linspace(.1,.3,150)
        for name in names:
            p=report['groups']['all53']['candidates'][name]['all_data_fit']['parameters']
            y=log10_shape(v,name,p)-log10_shape(.3,name,p)
            ax.plot(abs(v),10**y,color=colors[name],lw=2,label=labels[name])
        ax.set(yscale='log',xlabel='Internal voltage magnitude (V)',ylabel=r'$|I(v)|/I_{SET}(0.3\,V)$')
    axs[0,0].set(xscale='log',xlim=(.15,3));axs[0,0].legend(fontsize=8,frameon=False)
    axs[0,0].set_title('a  Joint fit retains the positive return turnover',loc='left',fontweight='bold')
    axs[0,1].set(xlim=(.1,.3),ylim=(5e-5,2))
    axs[0,1].set_title('b  Mean polarity is captured; early cycles are not',loc='left',fontweight='bold')
    ax=axs[1,0];x=np.arange(3)
    protocols=['retrospective_shared_amplitude','five_fold_autonomous_branch','chronological_autonomous_branch']
    for j,name in enumerate(names):
        values=[report['groups']['all53']['candidates'][name]['evaluation'][p]['equal_pair_branch_rmse_decade'] for p in protocols]
        ax.bar(x+(j-1)*.23,values,width=.22,color=colors[name],label=labels[name])
    ax.set_xticks(x,['Descriptive\nall 53 pairs','Held pairs\n5 folds','Chronological\n36–53'])
    ax.set(ylabel='Pair / branch balanced RMSE (decade)',ylim=(0,.6))
    ax.set_title('c  More parameters do not remove pairwise discrepancies',loc='left',fontweight='bold',fontsize=10.7)
    ax.legend(fontsize=8,frameon=False,loc='upper right')
    ax=axs[1,1]
    for tag,color,label in [('all53','#0072B2','All 53 pairs'),('cycles4to53','#009E73','Cycles 4–53 sensitivity')]:
        ps=charge['groups'][tag]['right_slow_U_profile'];cost=np.array([r['rmse_decade'] for r in ps])
        ax.plot([r['U_eV'] for r in ps],cost-cost.min(),color=color,lw=1.6,marker='o',ms=4,label=label)
    ax.set(xlabel='Fixed mutual interaction U (eV)',ylabel='RMSE above best profile value (decade)',xlim=(0,3),ylim=(-.005,.18))
    ax.set_title('d  Current curves do not determine a unique U',loc='left',fontweight='bold')
    ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Coupled contact / localized-state transport: physical consistency improves, history remains',fontsize=14,fontweight='bold',y=.97)
    fig.text(.5,.915,'Full bias | detailed balance | common current at each transition | finite double occupancy',ha='center',fontsize=11)
    notes=['a–b: Native measured traces normalized by their own SET current at 0.3 V; model shapes use the all-pair joint fit.',
           'c: Held-pair predictions exclude both target branches. These are frozen branches in internal voltage, not full-cycle simulations.',
           'd: Other parameters are refitted at each U. Flat profiles are identifiability diagnostics, not confidence intervals.',
           'The finite-U example uses a conditional point-charge bound (300 K, 6 nm, dielectric constant 3.1); no microscopic site is identified.']
    for k,s in enumerate(notes):fig.text(.075,.108-k*.024,s,fontsize=8.6,color='#444444')
    save(fig,'27_contact_network_evidence')

    fig,axs=plt.subplots(1,2,figsize=(12,6))
    fig.subplots_adjust(left=.09,right=.94,top=.79,bottom=.28,wspace=.42)
    ax=axs[0];g=bound['groups']['all53'];cycles=g['cycle_ids'];d=np.array(g['cycle_mean_log_ratio'])
    ax.scatter(cycles,10**d,c=['#D55E00' if i<=3 else '#444444' for i in cycles],s=22,label='Observed paired ratio')
    predicted=[]
    for detail in state['details']:
        p=detail['target_fits']['target_SET_E1_forecast']['parameters'];v=np.array(g['voltage_grid_V'])
        predicted.append(10**float(np.mean(log10_shape(-v,'coulomb_right',p)-log10_shape(v,'coulomb_right',p))))
    ax.plot(cycles,predicted,color='#0072B2',lw=1.5,label='Network initialized from target SET')
    ax.set(yscale='log',xlabel='Paired cycle index',ylabel='Geometric-mean negative / positive ratio',xlim=(0,54),ylim=(3e-4,3))
    ax.set_title('a  SET shape does not predict early reverse suppression',loc='left',fontweight='bold',fontsize=10.8)
    ax.legend(fontsize=8,frameon=False,loc='lower right')
    ax=axs[1]
    protocols=['target_SET_all_points_amplitude_only','target_SET_E1_forecast','joint_target_E1_E2_descriptive','independent_target_amplitudes_descriptive']
    values=[]
    for p in protocols:
        normal=state['groups']['cycles4to53'][p];early=state['groups']['first3'][p]
        values.append([normal['negative_rmse_decade'],early['negative_rmse_decade'],early['positive_rmse_decade'],early['positive_0p10_0p14_holdout_rmse_decade']])
    values=np.array(values);im=ax.imshow(values,cmap='magma_r',norm=LogNorm(vmin=.05,vmax=2.5),aspect='auto')
    ax.set_xticks(range(4),['RESET\n4–53','RESET\n1–3','SET fit\n1–3','SET low\nheld out'])
    ax.set_yticks(range(4),['SET only:\namplitude','SET only:\none energy','Both branches:\ntwo energies','Both branches:\ntwo amplitudes'])
    ax.tick_params(axis='both',labelsize=8)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if values[i,j]>.35 else 'black')
    ax.axhline(1.5,color='white',lw=2)
    ax.set_title('b  A descriptive repair fails the withheld low-bias test',loc='left',fontweight='bold',fontsize=10.8)
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.04);cb.set_label('RMSE (decade)',fontsize=9);cb.set_ticks([.05,.1,.3,1,2]);cb.set_ticklabels(['.05','.1','.3','1','2'])
    fig.suptitle('A shared amplitude is insufficient; fitting a hidden state is not predicting it',fontsize=14,fontweight='bold',y=.96)
    fig.text(.5,.875,'Exact common-shape floor on ±0.10–0.30 V: 0.315 decade (all 53), 0.129 decade (cycles 4–53)',ha='center',fontsize=10.5)
    notes=['a: Geometric means use matched internal-voltage grids; target RESET is excluded from the SET-initialized prediction.',
           'b: Top rows forecast RESET using measured SET. Bottom rows consume target RESET and are descriptive only.',
           'The last column scores positive 0.10–0.14 V points withheld from these fits (cycles 1–3); joint fitting does not resolve them.',
           'The error floor rejects amplitude-only fixed shapes at this residual level; it does not prove a unique state count, ion mechanism or relaxation.']
    for k,s in enumerate(notes):fig.text(.07,.172-k*.036,s,fontsize=8.6,color='#444444')
    save(fig,'28_contact_state_information')


def save(fig,name):
    for suffix in ('png','pdf'):
        p=ROOT/'analysis/figures'/f'{name}.{suffix}';fig.savefig(p,dpi=250,bbox_inches='tight',pad_inches=.16);print(p)
    plt.close(fig)


if __name__=='__main__':run()
