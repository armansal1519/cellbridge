# CellBridge API

The public calls are `fit`, `fit_anndata`, and the methods `predict`, `contributions` and `intervals`. The solver underneath is frozen. These wrappers do not change it.

`z` is one row per cell. The columns are the genes, then two technical columns, then the anchor proteins. `n_genes` counts the genes only. It does not count the two technical columns or the anchors. The width of `z` is `n_genes + 2 + n_anchors`.

The defaults of `fit` are not the grid used in the paper. A caller who wants that grid has to pass it.

## `fit`

```python
fit(donors, *, n_genes, n_anchors, ks=(1,), mixes=((0.0, 0.0), (1.0, 0.0)),
    lambdas=(1.0,), strategy="shared_top", top=0.05, seed=0)
```

`donors` is a list of mappings. Each mapping has:

- `donor`: a name. If it is omitted, the position in the list is used.
- `z_control`, `z_perturbed`: the cell-by-feature matrices for the two arms.
- `y_control`, `y_perturbed`: the proteins to predict. A one-dimensional array is treated as one protein.

At least three donors are required. Selection is leave-one-donor-out inside this list. Do not put a test donor in this list.

`n_genes` and `n_anchors` are required. There is no default.

`ks` is the metacell sizes. Default `(1,)`. The paper grid is `(1, 5, 20, 60)`.

`mixes` is a sequence of `(rho, tau)` pairs. `rho` mixes cell-level and donor-level information. `tau` is the second weight. Default `((0.0, 0.0), (1.0, 0.0))`. The paper grid uses tau 0 and rho in `(0, 0.25, 1, 4, 16, 64, 256)`. The package also defines a larger default inside the solver, with tau in `(0, 1, 16, 256)`. `fit` does not use that larger default unless the caller passes it.

`lambdas` is the ridge grid. Default `(1.0,)`. The paper grid is 25 values, log-spaced from `1e6` down to `1e-2`.

`strategy` chooses how the grid is reduced. Allowed values:

- `shared_top` (the default, and the paper setting): average the best `top` fraction of configurations, with one shared choice for every protein.
- `shared`: the single best shared configuration.
- `per_target`: the best configuration for each protein separately.
- `per_target_top`: the best `top` fraction for each protein separately.

`top` is that fraction. Default `0.05`. Values that round to an empty set still keep one configuration.

`seed` is an integer. Default `0`. It is offset by donor when the metacells are built. The paper fits use seed `20261007`.

The return value is a `FittedCellBridge` with `model`, `donors`, `n_genes` and `n_anchors`.

## `fit_anndata`

```python
fit_anndata(adata, *, donor="donor", arm="arm", protein="protein",
            control="control", perturbed="perturbed", n_genes, n_anchors, **kwargs)
```

`adata` is any object with `X`, `obs` and `obsm`. AnnData does not have to be installed. `X` is the model matrix. `obsm[protein]` is the protein matrix in the same cell order. `obs[donor]` and `obs[arm]` name the donor and the arm.

`donor`, `arm`, `protein`, `control` and `perturbed` are column and label names. The defaults are the strings above.

Each donor needs at least two cells in the control arm and two cells in the perturbed arm. Donors are grouped in sorted name order. Every other argument is passed to `fit`, including `ks`, `mixes`, `lambdas`, `strategy`, `top` and `seed`.

## `predict`

```python
FittedCellBridge.predict(dz)
```

`dz` is the donor-level feature difference, one row per donor, with the same columns as `z`. A one-dimensional array is treated as one donor. The return value has shape `(donors, proteins)`.

## `contributions`

```python
FittedCellBridge.contributions(dz, target=0)
```

`target` is the protein column. Default `0`. The return value has one column per feature of `dz`. The columns sum to the prediction after the training-mean shift is removed. They are an attribution of that prediction, not a causal effect.

## `intervals`

```python
FittedCellBridge.intervals(dz, calibration_scores, alpha=0.1)
```

`calibration_scores` are absolute errors on the same scale as `predict`. They must be finite and nonnegative, and the array must not be empty. `alpha` must be strictly between 0 and 1. Default `0.1`, which is a nominal 90 percent interval.

The interval is split conformal. The quantile uses NumPy's `higher` method at `ceil((n + 1) * (1 - alpha)) / n`, capped at 1. The same half-width is subtracted from and added to the point prediction. The two return arrays have the same shape as `predict`.

## What this page does not do

Passing the paper grid does not reproduce a sealed cohort. The cohort matrices are not in this repository. The commands that read those matrices are in `reproduce/README.md`.
