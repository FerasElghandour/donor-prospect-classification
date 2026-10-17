# Reproduction and benchmark scope

A new full UCI Adult run trains the income-proxy classifier with duplicate-feature-group-safe holdout, normalized UCI labels, missing-category handling and fitted-pipeline export. The 9,770-row holdout has85.74% accuracy and F0.5=0.7259.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `train-csv` | Fit supplied CSV and export fitted artifacts; benchmark validity depends on input provenance | csv |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

Income is a proxy target and does not measure donation behavior. This new reference protocol differs from the retained historical optimization experiment.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Recorded command definitions

### checks

```text
{python} smoke_test.py
```

### train-csv

```text
{python} train.py {input:csv} --output {output}/training
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
