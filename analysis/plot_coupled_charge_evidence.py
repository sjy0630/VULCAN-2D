"""Four-state network: shape tradeoff, exact rate degeneracy, SET information."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vulcan2d.coupled_charge import stationary
main=json.loads((ROOT/'analysis/coupled_charge_comparison.json').read_text())
validation=json.loads((ROOT/'analysis/coupled_charge_validation.json').read_text())
tradeoff=json.loads((ROOT/'analysis/coupled_charge_slope_tradeoff.json').read_text())
info=json.loads((ROOT/'analysis/full_set_information_audit.json').read_text())
blue,orange,green='#0072B2','#D55E00','#009E73'
fig,axes=plt.subplots(2,2,figsize=(13,8.6));rows=validation['positive_high_voltage_slopes']
ids=[r['cycle_id'] for r in rows];ax=axes[0,0]
ax.scatter(ids,[r['observed_slope'] for r in rows],s=18,color=green,label='Observed')
ax.plot(ids,[r['predicted_slope'] for r in rows],color=blue,label='Current-only fit')
ax.plot(ids,tradeoff['fits'][-1]['predicted_slopes'],color=orange,ls='--',label='Strong slope penalty')
ax.set(xlabel='Confirmed paired cycle',ylabel='Local log-log slope, internal 2–3 V',ylim=(-.03,.83))
ax.legend(frameon=False,fontsize=9,loc='upper right');ax.set_title('a  Saturation timing is a competing constraint',loc='left',fontweight='bold')
ax=axes[0,1]
error=[main['splits']['all_descriptive']['fit']['rmse_decade']]+[r['unpenalized_scores']['all']['weighted_rmse_decade'] for r in tradeoff['fits']]
slope=[np.median([r['predicted_slope'] for r in rows])]+[np.median(r['predicted_slopes']) for r in tradeoff['fits']]
ax.plot(error,slope,'o-',color=blue);ax.axhline(np.median([r['observed_slope'] for r in rows]),color=green,ls='--',label='Observed median')
for j,label in [(0,'No slope penalty'),(2,'Penalty 10'),(3,'Penalty 100')]:
    ax.annotate(label,(error[j],slope[j]),xytext=(8,6 if j!=2 else -20),textcoords='offset points',fontsize=9)
ax.set(xlabel='Original weighted current RMSE (decade)',ylabel='Predicted median high-voltage slope',xlim=(.145,.22),ylim=(-.03,.72))
ax.legend(frameon=False,loc='lower right',fontsize=9);ax.set_title('b  Matching slope worsens the current fit',loc='left',fontweight='bold')
ax=axes[1,0];rate=np.linspace(-8,8,81);shared=[-.1,.3,.35,.02,0.];states=[stationary(.2,np.r_[shared,r],-.13) for r in rate]
currents=np.array([r['current_scale'] for r in states]);p11=np.array([r['probabilities'][3] for r in states])
ax.plot(rate,currents/currents[len(rate)//2],color=blue,label='Steady current / reference')
ax.set(xlabel='ln(side exchange rate / right-contact rate)',ylabel='Normalized steady current',ylim=(.96,1.04))
other=ax.twinx();other.plot(rate,p11,color=orange,ls='--',label='Joint occupation P11');other.set_ylabel('Joint occupation probability',color=orange)
ax.text(.04,.08,'Model counterfactual; not fitted device data',transform=ax.transAxes,fontsize=9)
lines=ax.get_lines()+other.get_lines();ax.legend(lines,[r.get_label() for r in lines],frameon=False,loc='upper right',fontsize=9)
ax.set_title('c  A rate is absent from the steady I–V',loc='left',fontweight='bold')
ax=axes[1,1];names=['constant','SET_return_low','full_SET','full_SET_and_history']
values=[info['candidates'][n]['five_fold']['groups']['all']['rmse_decade'] for n in names]
ax.bar(np.arange(4),values,color=[blue,blue,orange,orange],width=.65)
for i,value in enumerate(values):ax.text(i,value+.018,f'{value:.3f}',ha='center')
ax.set(xticks=np.arange(4),xticklabels=['Mean','SET return','Full SET','SET + history'],ylabel='RESET mean-log-current prediction RMSE',ylim=(0,1.0))
ax.set_title('d  Full SET features do not improve this map',loc='left',fontweight='bold')
for ax in axes.ravel():ax.spines[['top','right']].set_visible(False)
fig.suptitle('Four-state feedback: an exact simplification, with unresolved prediction constraints',fontsize=15,y=.985)
fig.text(.07,.022,'All 53 pairs retained. Slope penalties reuse the same data and do not replace primary held-out scores.\n'
         'Panel c: rate-independent steady current can coexist with rate-dependent correlations. Panel d: nested ridge maps, not microscopic state laws.',fontsize=9)
fig.tight_layout(rect=(.015,.08,1,.955),h_pad=2.5,w_pad=2.8)
folder=ROOT/'analysis/figures';folder.mkdir(exist_ok=True)
for suffix in ('png','pdf'):fig.savefig(folder/f'37_coupled_charge_evidence.{suffix}',dpi=180)
print(folder/'37_coupled_charge_evidence.png')
