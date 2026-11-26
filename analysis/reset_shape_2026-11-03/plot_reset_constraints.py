"""Plot two audited training witnesses; no new model selection or fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'mechanism_selection_2026-11-03/field_rupture_trial.npz'
with np.load(SOURCE) as z:
    ids=z['record_ids'];v=abs(z['command_voltage_v'])
    obs=z['observed_reset_abs_current_a'];g=z['prior_SET_conductance_s']
apex=np.argmax(v)
def ix(x,ret=False):
    branch=np.arange(len(v))>apex if ret else np.arange(len(v))<=apex
    found=np.flatnonzero(branch&np.isclose(v,x,atol=1e-7,rtol=0))
    assert len(found)==1
    return found[0]
def row(cid):return int(np.flatnonzero(ids==cid)[0])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.15,'savefig.dpi':300,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10.8,4.2))
blue,orange='#0072B2','#D55E00'
ax=axes[0]
for off,cid,color in [(-.16,11,blue),(.16,14,orange)]:
    current=obs[row(cid),[ix(.05),ix(.725)]]*1000
    ax.bar(np.arange(2)+off,current,width=.29,color=color,alpha=.85,
           yerr=.2*current,error_kw={'capsize':4,'ecolor':'#333333'},label=f'Cycle {cid}')
    for j,y in enumerate(current):
        ax.text(j+off,y*1.29,f'{y:.3f}',ha='center',fontsize=9)
ax.set(xticks=[0,1],xticklabels=['Early RESET\n0.05 V','Later RESET\n0.725 V'],
    ylabel='Current magnitude (mA)',yscale='log',ylim=(.06,3.5),
    title='a  Initial current ordering reverses')
ax.legend(frameon=False,loc='upper left')

ax=axes[1];cid=10;voltage=float(v[ix(.975)])
out=obs[row(cid),ix(voltage)];ret=obs[row(cid),ix(voltage,True)]
diff=out-ret;bound=voltage*g[row(cid)]
lower=.8*out-1.2*ret;upper=1.2*out-.8*ret
ax.bar([0,1],[diff*1e3,bound*1e3],width=.55,color=[orange,blue],alpha=.85,
    yerr=np.array([[diff-lower,.2*bound],[upper-diff,.2*bound]])*1e3,
    error_kw={'capsize':5,'ecolor':'#333333'})
ax.axhline(lower*1e3,color=orange,ls=':',lw=1.4)
ax.axhline(1.2*bound*1e3,color=blue,ls=':',lw=1.4)
gap=(lower-1.2*bound)*1e3
ax.text(.5,18.5,f'Intervals still separated by {gap:.3f} mA',ha='center',fontsize=10)
ax.set(xticks=[0,1],xticklabels=['Measured outward\nminus return','Maximum permitted\nfrom preceding SET'],
    ylabel='Current magnitude (mA)',ylim=(0,20),
    title='b  Pure-loss bound fails: cycle 10, 0.975 V')
fig.suptitle('RESET constraints that do not require selecting a transport spectrum',fontsize=14,y=1.01)
fig.text(.5,.015,'Error bars: declared ±20% sensitivity, not measured uncertainty. Training records only.\n'
    'Left assumes common protocol, unique scalar dynamics and monotone readout; right assumes static within-cycle background and no path growth after the SET read.',
    ha='center',fontsize=8)
fig.tight_layout(rect=(0,.12,1,.95),w_pad=2.3)
for ext in ['png','pdf']:fig.savefig(HERE/f'reset_constraint_witnesses.{ext}',bbox_inches='tight')
summary={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'record_ids':ids.tolist(),'witness_selection':'Strongest witnesses already selected and documented by RESET_SCALAR_ORDER_CHECK and RESET_SHAPE_DIAGNOSTIC; no additional selection here',
    'sensitivity_fraction':.2,'not_measured_error_bars':True,
    'cycle10_diff_A':float(diff),'cycle10_bound_A':float(bound),
    'cycle10_positive_interval_gap_A':float(lower-1.2*bound)}
(HERE/'reset_constraint_witnesses.metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
