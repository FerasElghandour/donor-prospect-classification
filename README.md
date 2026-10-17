# Donor Prospect Classification

**Feras Elghandour** · Technical case study

Compare classifiers for an income-based prospect-ranking proxy and make the precision–recall trade-off explicit.

## Implementation

A portable train-fitted numeric/categorical preprocessing pipeline and AdaBoost evaluation with accuracy, precision, recall, and F0.5.

```mermaid
flowchart LR
    A[Validated inputs] --> B[Portable implementation]
    B --> C[Evaluation and contract checks]
    C --> D[Inspectable results]
```

## Completed study

The saved optimized model reports test accuracy 0.8690, precision 0.7811, recall 0.6426, and F0.5 0.7489. Restricting the model to its five leading features reduces F0.5 to 0.6883.

These historical figures describe the completed study. This refactored standalone implementation has separate validation and has not been used to recertify the historical benchmark. The public release contains the portable core and focused tests.

## Run locally

Python 3.11 or later.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
python prospect_model.py /path/to/census.csv
```

Prepare your own inputs as described in [DATA.md](DATA.md). Tests use generated fixtures and do not need credentials or provider calls. [VALIDATION.md](VALIDATION.md) records their scope.

## Practical relevance and limitations

Income is a proxy, not observed donation propensity or a measure of campaign lift. Demographic predictors require careful review before any targeting decision. Cross-validation operates on the training partition; the held-out comparison is a single experiment.

This edition makes the core behavior reusable and inspectable. Historical metrics and synthetic contract tests provide different kinds of evidence; neither establishes client deployment or commercial impact.
