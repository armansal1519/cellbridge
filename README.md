# CellBridge

CellBridge predicts how a donor's surface proteins change between two conditions, using RNA and eight antibodies measured in both conditions. Each protein has one slope. Donor effects are removed, and the fit is closed form, so a prediction is an exact sum of feature contributions.

It is not a cell-level protein imputer, and it is not a second version of the abundance-shrinkage baseline in the same project. That baseline is a comparator. CellBridge matches it. CellBridge is much faster than totalVI and, in these two cohorts, more accurate than that totalVI fit. It does not beat the abundance-shrinkage baseline.

## Headline results

These numbers are the sealed tests. They are not development scores.

- **GSE334503, typhoid vaccination, 10 test donors.** Standardized mean absolute error was 0.284 for CD25 and CD69 and 0.322 across hidden proteins. The prespecified sequence rejected the training mean, both pseudobulk ridges and totalVI. It did not reject the abundance-shrinkage comparison, and the sequence stopped there. The primary claim was supported. The fit took 9.2 seconds. totalVI took 39 minutes.
- **E-MTAB-9357, COVID-19 follow-up, 29 test donors.** The same frozen method, refit and not otherwise changed, scored 0.530 and 0.471. The sequence rejected the all-protein comparisons with the mean and the ridges, then stopped at the CD25 and CD69 comparison with the mean. The primary claim was not supported. Later comparisons, including totalVI, were recorded and were not tested. On this cohort the eight anchors carry the gain. RNA alone sits on the training mean.

Both cohorts are small. Feature contributions describe the fitted prediction. They are not causal effects of measuring an antibody. The E-MTAB-9357 protocol was hashed before unblinding. A public OSF timestamp was not available.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Python 3.11 and 3.12 are the tested versions. Every argument of `fit`, `fit_anndata`, `predict`, `contributions` and `intervals` is listed in `docs/api.md`. The defaults are not the grid used in the paper. `examples/cellbridge_example.py` fits four synthetic donors with seed 0, and `examples/expected_output.json` is the output that test checks.

## Quickstart

The columns of `z` are genes, two technical covariates and the anchor antibodies, in that order. `n_anchors=0` below means the six columns are four genes and two technical covariates.

```python
import numpy as np
from cellbridge import fit

rng = np.random.default_rng(0)
donors = []
for i in range(4):
    zc, zp = rng.normal(size=(30, 6)), rng.normal(size=(30, 6))
    donors.append({
        "donor": f"d{i}",
        "z_control": zc, "y_control": zc[:, :1],
        "z_perturbed": zp, "y_perturbed": zp[:, :1] + 0.4,
    })
model = fit(donors, n_genes=4, n_anchors=0, ks=(1,),
            mixes=((0.0, 0.0),), lambdas=(1.0,))
print(model.predict(np.zeros(6)))
print(model.contributions(np.zeros(6), target=0))
```

`notebooks/cellbridge_tutorial.ipynb` is the same example. An object with `X`, `obs` and `obsm['protein']` can be passed to `fit_anndata`. AnnData itself is optional. `examples/anndata_tutorial.py` builds that object from synthetic cells, fits it, and calls `panel_curve`. Version 1.1.0 adds that helper. The sealed results remain the v1.0.1 release. The solver is unchanged.

## Reproduce the paper

The sealed predictions, decisions and donor-level losses are in `runs/cellbridge_v100/`. The commands that produced them are in `scripts/`, and `reproduce/README.md` lists one command per step. The count matrices are not in this repository. GSE334503 is on Zenodo (doi:10.5281/zenodo.20266085). E-MTAB-9357 is on ArrayExpress.

Check the frozen solver before trusting a checkout:

```bash
python scripts/verify_frozen.py
```

## Cite

Salehi, A. CellBridge: fast, closed-form prediction of donor protein responses from RNA and a small antibody panel. https://github.com/armansal1519/cellbridge

A Zenodo DOI is not assigned yet. See `CITATION.cff`.
