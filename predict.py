from pathlib import Path
import argparse,json,joblib,numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('csv',type=Path);a=p.parse_args();x=pd.read_csv(a.csv).drop(columns='income',errors='ignore').replace(r'^\s*\?\s*$',np.nan,regex=True)
for c in ['capital-gain','capital-loss']:
 if c in x:
  if (x[c]<0).any():raise ValueError('Nonnegative capital features required')
  x[c]=np.log1p(x[c])
print(json.dumps(joblib.load(a.model).predict(x).tolist()))
