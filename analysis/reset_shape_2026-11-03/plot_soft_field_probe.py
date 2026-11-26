"""Same-mask, same-cycle comparison of the frozen beta6 and field probes."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OLD=HERE/'soft_beta_conditional_forward.npz'
NEW=HERE/'soft_field_mobility_probe.npz'
with np.load(OLD) as z:
    v=z['command_voltage_v'];s=z['path_voltage_s'];obs=abs(z['observed_current_a'])
    mask=z['original_scoring_mask'];old=z['beta6_refined_current_a']
with np.load(NEW) as z:
    assert np.array_equal(v,z['command_voltage_v']) and np.array_equal(mask,z['original_scoring_mask'])
    assert np.array_equal(obs,abs(z['observed_current_a']))
    new=z['refined_current_a']
assert mask.sum()==500 and np.all(old[mask]>0) and np.all(new[mask]>0)
apex=np.argmax(v);out=(np.arange(len(v))<=apex)&mask;ret=~(np.arange(len(v))<=apex)&mask
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42,'savefig.dpi':300})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.6))
for ax,m,title in zip(axes,[out,ret],['a  Outward SET','b  Return SET']):
    for vals,color,style,label in [(obs,'#333333','-','Measured cycle 4'),
            (old,'#D55E00','--',r'$\beta=6$, constant mobility'),
            (new,'#0072B2','-',r'$\beta=6$, mobility $\propto |V_h|$')]:
        ax.plot(v[m],vals[m],style,lw=1.8,color=color,label=label)
    ax.set(xlabel='Applied command voltage (V)',ylabel='Current magnitude (A)',yscale='log',
           title=title,xlim=(0,5),ylim=(min(old[mask].min(),new[mask].min(),obs[mask].min())*.5,1e-4))
axes[1].invert_xaxis()
axes[0].axvline(.2,color='#777777',lw=.8,ls=':')
fig.suptitle('Complete single-cycle SET probes — conditioned at outward 0.2 V',fontsize=14,y=1.01)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,.96),fontsize=9)
fig.text(.5,.02,'Both use all 500 original eligible points; no fitting or low-bias subtraction.\n'
    'A measured 0.2 V anchor determines the initial state; this is not holdout prediction or a complete unified model.',
    ha='center',fontsize=8.5)
fig.tight_layout(rect=(0,.12,1,.90),w_pad=2)
for ext in ['png','pdf']:fig.savefig(HERE/f'soft_field_probe_comparison.{ext}',bbox_inches='tight')
metadata={'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD,NEW]},
    'mask_points':int(mask.sum()),'cycle':4,'topology':'1t1r','phase':'SET',
    'scope':'Conditional single-cycle diagnostic, not full-history or independent prediction',
    'raw_RMSE_decade':{'constant_mobility':float(np.sqrt(np.mean(np.log10(old[mask]/obs[mask])**2))),
                       'field_mobility':float(np.sqrt(np.mean(np.log10(new[mask]/obs[mask])**2)))}}
(HERE/'soft_field_probe_comparison.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata))
