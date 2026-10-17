"""Fit the released income-proxy classifier and export its fitted preprocessing."""
from pathlib import Path
import argparse,hashlib,json,joblib,pandas as pd
from prospect_model import fit_evaluate
p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
if a.output.exists():raise ValueError('Use a new output directory')
a.output.mkdir(parents=True);frame=pd.read_csv(a.csv);model,metrics=fit_evaluate(frame,seed=a.seed);joblib.dump(model,a.output/'model.joblib')
reloaded=joblib.load(a.output/'model.joblib')
# fit_evaluate applies log1p to capital features before its returned pipeline.
x=frame.drop(columns='income').copy()
import numpy as np
for col in ['capital-gain','capital-loss']:
 if col in x:x[col]=np.log1p(x[col])
assert (model.predict(x.head(5))==reloaded.predict(x.head(5))).all()
(a.output/'metrics.json').write_text(json.dumps({'seed':a.seed,'input_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),'metrics':metrics,'capital_preprocessing':'Apply log1p to capital-gain and capital-loss before pipeline inference','scope':'Income proxy classification, not measured donor response'},indent=2)+'\n')
