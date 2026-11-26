import json, numpy as np
from collections import defaultdict
C='/Users/sujiangyu/Desktop/SIM2RRAM/outputs/vulcan-handoff/iv41_cache/'
idx=json.load(open(C+'index.json'))
for s,g in {9:1.3,10:1.4,11:1.5}.items():
    name=idx['names'][s-1]; meta={str(r[4]):r for r in idx['curves'][1:] if r[13]==name}
    d=np.load(C+f's{s}.npz'); seq=''
    for k in sorted(d.files,key=int):
        if meta[k][6]!='正': continue
        o,v,i=d[k].T
        if not np.isfinite(i).all(): continue
        a=np.argmax(v); r=np.arange(len(v))>a
        j=np.where(r)[0][np.argmin(np.abs(v[r]-.15))]
        seq+='H' if abs(i[j])/.15>30e-6 else 'l'
    n=len(seq); p=seq.count('H')/n
    runs=1+sum(seq[t]!=seq[t-1] for t in range(1,n))
    exp=1+2*n*p*(1-p)
    print(g,n,'H frac %.2f runs %d (iid expect %.1f)'%(p,runs,exp)); print(seq)
    print(' first70%% H=%.2f  last30%% H=%.2f'%(seq[:int(.7*n)].count('H')/int(.7*n), seq[int(.7*n):].count('H')/(n-int(.7*n))))
