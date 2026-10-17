import numpy as np
import pandas as pd
from prospect_model import fit_evaluate
rng=np.random.default_rng(6);n=120;signal=rng.normal(size=n)
frame=pd.DataFrame({'age':np.arange(n)%40+20,'signal':signal,'sector':np.where(signal>0,'A','B'),'income':np.where(signal>0,'>50K','<=50K')})
model,result=fit_evaluate(frame)
assert result['test_rows']==24 and result['f0_5']>.9
assert 'prepare' in model.named_steps
try:fit_evaluate(frame.assign(income='unknown'))
except ValueError:pass
else:raise AssertionError('Unknown label accepted')
print('Train-fitted classification, held-out metric accounting, and invalid-label checks passed.')
