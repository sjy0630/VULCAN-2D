"""Visualize saved, unadjusted reverse/forward predictions versus paired data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def run():
    report=json.loads((ROOT/'analysis/occupation_polarity_check.json').read_text())
    points=report['points'];names=('full_bias_level','full_bias_level_background')
    colors=('#009E73','#0072B2');labels=('One level, fixed rates','One level + Ohmic background')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(11.5,5.8))
    fig.subplots_adjust(left=.09,right=.97,top=.77,bottom=.27,wspace=.28)
    ax=axs[0]
    for cid in range(1,54):
        rows=[r for r in points if r['candidate']==names[0] and r['cycle_id']==cid]
        ax.plot([r['hbn_voltage_V'] for r in rows],[r['observed_reset_set_ratio'] for r in rows],
                color='#D55E00' if cid<=3 else '#AAAAAA',alpha=.6 if cid<=3 else .35,lw=1 if cid<=3 else .65)
    for name,color,label in zip(names,colors,labels):
        rows=[r for r in points if r['candidate']==name]
        volts=sorted({r['hbn_voltage_V'] for r in rows})
        median=[np.median([r['predicted_negative_positive_ratio'] for r in rows if r['hbn_voltage_V']==v]) for v in volts]
        ax.plot(volts,median,color=color,lw=2.5,label=label)
    ax.axhline(1,color='#888888',ls='--',lw=1)
    ax.set(yscale='log',xlabel='Internal voltage magnitude (V)',ylabel=r'Negative / positive current ratio',xlim=(.1,.3),ylim=(1e-4,80))
    ax.set_title('a  Polarity prediction has the wrong scale',loc='left',fontweight='bold')
    ax.legend(fontsize=8,frameon=False,loc='center left',bbox_to_anchor=(0,.34))
    ax=axs[1]
    for name,color,label in zip(names,colors,labels):
        rows=[r for r in points if r['candidate']==name and abs(r['hbn_voltage_V']-.3)<1e-10]
        ax.plot([r['cycle_id'] for r in rows],[r['predicted_negative_positive_ratio'] for r in rows],color=color,lw=1.6,label=label)
    rows=[r for r in points if r['candidate']==names[0] and abs(r['hbn_voltage_V']-.3)<1e-10]
    ax.scatter([r['cycle_id'] for r in rows],[r['observed_reset_set_ratio'] for r in rows],
               c=['#D55E00' if r['cycle_id']<=3 else '#444444' for r in rows],s=18,label='Measured paired ratio')
    ax.axhline(1,color='#888888',ls='--',lw=1)
    ax.set(yscale='log',xlabel='Paired cycle index',ylabel='Current ratio at |v| = 0.30 V',ylim=(1e-4,80),xlim=(0,54))
    ax.set_title('b  The discrepancy persists beyond cycles 1–3',loc='left',fontweight='bold')
    ax.legend(fontsize=8,frameon=False,loc='lower right')
    fig.suptitle('A good positive-return fit does not establish a frozen single-level mechanism',
                 fontsize=13,fontweight='bold',y=.97)
    fig.text(.5,.88,'Full reservoir voltage difference | no reverse-bias fit or polarity multiplier | all 53 pairs',ha='center',fontsize=10.5)
    notes=['Observed ratio: RESET initial / paired SET return after measured-transistor de-embedding. Orange: first three cycles.',
           'Each prediction excludes the target SET cycle from parameter fitting; no RESET data enter the candidate parameters.',
           'The measurements span two scans. Failure constrains fixed-rate, frozen-state single-level models; it does not prove relaxation.',
           'The comparison retains zero-reference-sensitive points; separate signal-guard and cycles 4–53 summaries preserve the discrepancy.']
    for k,s in enumerate(notes):fig.text(.075,.17-k*.038,s,fontsize=8.6,color='#444444')
    for suffix in ('png','pdf'):
        p=ROOT/'analysis/figures'/f'26_occupation_polarity.{suffix}';fig.savefig(p,dpi=250,bbox_inches='tight',pad_inches=.15);print(p)
    plt.close(fig)


if __name__=='__main__':run()
