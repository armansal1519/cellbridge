# ResponseBridge 1.0: CellBridge

CellBridge replaces the v0.5 gated estimator. It is a closed-form linear model
trained on single cells rather than on 9–11 donor-level responses.

For target protein *t*, cell *i* of donor *d* in arm *a*:

    y_it = c_dt + gamma_t [a = perturbed] + z_i' beta_t + e_it

with RNA (log-normalised, 1000 genes), two RNA technical covariates and,
optionally, the eight anchor ADTs as features `z`. Donor effects and the
response intercept are unpenalised and profiled out. The loss is split into a
within-arm cell-state term and a between-arm donor-response term weighted by
`rho`. `rho = 0` learns slopes only from cell-to-cell covariation, `rho = 1`
is pooled cell-level least squares, and `rho -> infinity` is pseudobulk
response ridge. Metacells (RNA-only k-means, size `k`) reduce
errors-in-variables attenuation. The donor-response prediction is exact:
`m_y + (dz_q - m_z)' beta`. It is therefore an additive per-gene/per-anchor
decomposition. With `lambda -> infinity` it falls back to the training mean.

All grid configurations are solved from per-donor sufficient statistics in one
eigenbasis plus a rank-D Woodbury update. A fully nested
leave-one-donor-out fit over 2 x 4 x 7 x 25 = 1400 configurations takes about
5–20 s on a laptop. Selection averages the best 5% of configurations by
target-pooled inner error; an average of linear fits is still linear.

## Development (before any GSE334503 calibration/test access)

Nine tasks: eight Lawlor arms (previously exposed) and leave-one-donor-out inside
the 11 GSE334503 training donors. Mean standardized MAE (training-donor SD):

| Method | CD25/CD69 | All hidden |
|---|---:|---:|
| CellBridge | 0.727 | 0.777 |
| v0.4 abundance shrinkage (nested) | 0.825 | 0.896 |
| Joint pseudobulk ridge (tuned) | 0.986 | 0.985 |
| RNA-only pseudobulk ridge (tuned) | 0.991 | 0.988 |
| Training mean | 1.037 | 1.032 |

v0.4 remains better on Lawlor CD3/CD28 T (its development arm) and slightly
better on GSE334503 training-donor CD25/CD69. A between-donor abundance term
(`tau`) and response-variance gene selection were tested and did not improve the
nine-task mean, so they were not selected.

## Independent test

`configs/protocol_cellbridge_v100.json` fixes endpoints, comparators and a
fixed-sequence hypothesis order. `scripts/cellbridge_gse_v100.py` runs
`freeze -> fit -> (totalVI) -> seal -> unblind`. The v0.5 freeze and seal were
superseded before any outcome access; v0.5 sealed predictions remain secondary
comparators. A rehearsal of every stage used only training donors
(`runs/cellbridge_v100/gse334503_dryrun`).

On the 10 sealed GSE334503 test donors, CellBridge scored 0.284 (CD25/CD69)
and 0.322 (all 122 hidden proteins). It beat the training mean, tuned
pseudobulk ridges and totalVI on both endpoints (H1–H8 rejected in fixed
sequence). It was non-inferior to v0.4 but not shown superior (H9 p = 0.15).
Full results: `analyses/cellbridge_v100/report.md`; figure:
`manuscript/v100/figure_main.png`.

## Preservation

`analyses/cellbridge_v100/results_lock.json` hashes the sealed run, the frozen
sources, the protocol, the report, the figure and the development summaries.
`scripts/verify_results_lock_v100.py verify` rebuilds the headline numbers from
`evaluation/summary.csv` and `decision.json` and checks them against that lock.
The sealed run and the development runs are read-only. Later GSE334503 analysis
goes in `runs/cellbridge_v100/gse334503_posthoc/` with `post_hoc: true`
(`responsebridge.result_lock.write_posthoc`). A recovery archive is
`archive/cellbridge_v1.0.0.tar.gz` (see the sibling `.sha256`). Its seal time
is the laptop clock until that archive is uploaded.
