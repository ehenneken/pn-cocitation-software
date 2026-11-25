import sys, json, os
sys.path.insert(0,'../audit'); sys.path.insert(0,'.')
os.chdir('../audit')
import tau2_audit as A
for k in [k for k in sys.modules if k=='pipeline' or k.startswith('pipeline.')]: del sys.modules[k]
sys.path.insert(0,'/tmp/claude-0/-home-claude/04ee3a90-8925-5b57-923c-1acb52b5d168/scratchpad/z5')
from pipeline import cohesion as C
print(C.__file__)
ys=A.years
data=dict(years=ys,seeds=A.seeds,annual_cit=A.ann_cit,annual_cocit=A.ann_co,
          cum_cit=A.cumulate(A.ann_cit,ys),cum_cocit=A.cumulate(A.ann_co,ys))
print('start',C.start_year(data))
r=C.analyse(data,n_boot=int(sys.argv[1]))
ref=json.load(open('final_tau2_results.json'))['2000']
print('tau',r['exp']['tau'],ref['exp']['tau']); print('beta_free',r['pl_free']['beta'],ref['pl_free']['beta']); print('beta95',r['pl_fixed']['beta'],ref['pl_origin1995']['beta'])
print(r.get('bootstrap')); print(ref['boot'])
