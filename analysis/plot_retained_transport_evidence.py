"""Display limited refit gains and the unresolved transition out of rectification."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def run():
    old=json.loads((ROOT/'analysis/retained_charge_configuration_chronological_single_contact.json').read_text())
    new=json.loads((ROOT/'analysis/retained_transport_chronological.json').read_text())
    alternate=json.loads((ROOT/'analysis/retained_transport_chronological_fast_start.json').read_text())
    d=np.load(ROOT/'analysis/retained_transport_chronological.npz')
    fast=np.load(ROOT/'analysis/retained_fast_side_evidence.npz')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.4),constrained_layout=True)
    ax=axes[0];cycles=np.arange(1,54)
    for record,color,label in [(old,'.55','Inherited transport'),(new,'#0072B2','Joint refit, training selected'),
                               (alternate,'#D55E00','Fast-start alternative')]:
        per=record['full_scores']['per_cycle'];ax.plot(cycles,[per[str(c)] for c in cycles],color=color,lw=1.1,label=label)
    ax.axvline(35.5,color='.5',ls=':',lw=.8)
    ax.set(title='a  More freedom: limited overall gain',xlabel='Cycle index',ylabel='Eight-window RMSE (decade)',xlim=(.5,53.5))
    ax.legend(frameon=False,fontsize=7)
    for ax,c,letter in [(axes[1],3,'b'),(axes[2],4,'c')]:
        for wi,color,label in [(2,'#0072B2','SET return'),(4,'#D55E00','RESET outward')]:
            m=(d['cycle']==c)&(d['window']==wi)&(abs(d['voltage'])>=.1-1e-10)
            xx=abs(d['voltage'][m]);order=np.argsort(xx)
            ax.semilogy(xx[order],d['observed'][m][order],'o',color=color,ms=4,label=f'Measured {label}')
            ax.semilogy(xx[order],fast['chronological_current_A'][m][order],'-',color=color,lw=1.2,label=f'Fixed fast exchange: {label}')
        ax.set(title=f'{letter}  Cycle {c}: '+('polarity contrast appears' if c==3 else 'recovery is still missing'),
               xlabel='Applied voltage magnitude (V)',ylabel='Current magnitude (A)',ylim=(2e-10,3e-6))
        ax.legend(frameon=False,fontsize=7)
    fig.suptitle('Fast trap exchange can block reverse current; a single fixed rate does not explain the history',fontsize=11)
    fig.supxlabel('Panels b/c: unchanged first-35 source parameters except ell_m=6; no refit or imposed switch; refined path step 0.000625 V.',fontsize=8)
    for ext in ('png','pdf'):fig.savefig(ROOT/'analysis/figures'/f'43_retained_transport_and_capture.{ext}')
    plt.close(fig);d.close();fast.close()


if __name__=='__main__':run()
