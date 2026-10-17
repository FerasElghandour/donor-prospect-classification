"""Train-fitted prospect classification from a user-provided feature table."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,precision_score,recall_score,fbeta_score

def fit_evaluate(frame,target='income',seed=0):
    if target not in frame or len(frame)<30:raise ValueError('Target and at least 30 records required')
    y=frame[target].astype(str).str.strip().str.replace(r'\.$','',regex=True).map({'<=50K':0,'>50K':1})
    if y.isna().any() or y.nunique()!=2:raise ValueError('Target must contain both <=50K and >50K labels')
    x=frame.drop(columns=target).replace(r'^\s*\?\s*$',np.nan,regex=True).copy()
    for col in ['capital-gain','capital-loss']:
        if col in x:
            if (x[col]<0).any():raise ValueError('Capital values must be nonnegative')
            x[col]=np.log1p(x[col])
    groups=pd.util.hash_pandas_object(x,index=False).to_numpy()
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(x,y,groups))
    if set(groups[train])&set(groups[test]):raise AssertionError('Duplicate-feature group leakage')
    if y.iloc[train].nunique()!=2:raise ValueError('Training partition must contain both classes')
    numeric=list(x.select_dtypes(include='number').columns);categorical=[c for c in x if c not in numeric]
    pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',MinMaxScaler())]),numeric),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
    model=Pipeline([('prepare',pre),('classify',AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,random_state=seed),n_estimators=100,learning_rate=.5,random_state=seed))])
    model.fit(x.iloc[train],y.iloc[train]);pred=model.predict(x.iloc[test]);truth=y.iloc[test]
    return model,{'test_rows':len(test),'accuracy':float(accuracy_score(truth,pred)),
       'precision':float(precision_score(truth,pred,zero_division=0)),
       'recall':float(recall_score(truth,pred,zero_division=0)),'f0_5':float(fbeta_score(truth,pred,beta=.5,zero_division=0))}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);args=p.parse_args()
    _,metrics=fit_evaluate(pd.read_csv(args.csv));print(json.dumps(metrics,indent=2))

