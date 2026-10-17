# Input contract

Provide a CSV with an income target containing <=50K and >50K labels and numeric/categorical predictors. Capital-gain/loss columns, if present, must be nonnegative. The pipeline learns imputation, scaling, and categorical vocabulary from training rows only. No census records, donor data, or model files are bundled. Income is an experimental proxy, not observed donation behavior.
