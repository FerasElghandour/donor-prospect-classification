"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from prospect_model import fit_evaluate
# Generated fixtures demonstrate current API behavior only.
rng=np.random.default_rng(6);n=120;signal=rng.normal(size=n)
frame=pd.DataFrame({'age':np.arange(n)%40+20,'signal':signal,'sector':np.where(signal>0,'A','B'),'income':np.where(signal>0,'>50K','<=50K')})
model,result=fit_evaluate(frame)
import json
print(json.dumps(result, indent=2))
