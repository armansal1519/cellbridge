# Post hoc final analysis, v1.1

This protocol is for cohorts that are already unblinded. It is not part of H1–H10. It does not change the frozen solver, `paper/numbers.tex`, the v1.0 protocols, or the sealed run directories.

The machine-readable copy is `configs/protocol_final_v110.json`. A public snapshot of both files is required before any block is scored. Until `docs/final_v110_timestamp.json` records that snapshot, the runner refuses to score.

Abundance shrinkage has an exact additive decomposition and donor-level intervals. Those properties are not what this protocol tests.

## Metric

Standardized MAE, with the training-donor `response_scale` and the truth the sealed evaluation used. Endpoints are E1 and E2. A paired interval is 10,000 test-donor resamples from seed 20261008, percentiles 2.5 and 97.5. Each interval starts a new generator from that seed.

The full-data refit must match the sealed test predictions of CellBridge and of v0.4 within absolute gap `1e-4`. That check is E-MTAB at 64 training donors and GSE334503 at 11.

## Learning curve

Expected direction: CellBridge has lower E2 loss when few donors are available.

E-MTAB sizes are 8, 16 and 32, five draws each, then the full 64. The draw seed is 20261008. GSE334503 sizes are 4, 6 and 8, five draws each, then the full 11. The draw seed is 20261009. Both methods use the same donor lists.

CellBridge genes stay the sealed selection. Its metacell seed is the sealed seed 20261007 plus 101 times the donor's index in the full training list. v0.4 rows are built on all training donors and then subset. On E-MTAB the cell cap and the gene pool follow the sealed `fit`. On GSE334503 the rows come from `paired_rows` and the fit from `baselines_queries`. The 2000 highly variable genes are chosen inside each v0.4 fit. The gene universe is not recomputed on the subset.

Within a size, draw-level losses are averaged inside each test donor. The interval resamples those donors. The difference is v0.4 minus CellBridge.

## Attribution

Expected direction: a split half of CellBridge agrees with its other half more closely than the two v0.4 halves agree.

Ten splits. E-MTAB is 32 against 32, seed 20261010. GSE334503 is 5 against 6, seed 20261011. A split is a permutation of the sealed training order. Test protein values are not read.

The measure is the Spearman correlation of per-gene contributions on the shared genes, for the same test donor and the same E2 target. A target with fewer than five shared genes, or with a zero-variance contribution, is skipped. The top-20 overlap is reported and is not part of the claim.

CellBridge terms are the gene columns of `contributions`. v0.4 terms, for the abundance formulation, are `(x[j] / x_scale[j]) * coef[t, j] * y_scale[t]`. They must sum to `rna_contribution` within `1e-8`. If they do not, the contribution comparison stops, one line is added to `paper/submission/AMENDMENTS.md`, and the claim uses the coefficient-vector Spearman instead.

Cognate recovery uses the full fit. A target is eligible when its cognate gene is in both selected feature sets. The percentile uses strict inequality. The null is 1,000 label permutations, seed 20261013.

The per-protein split by cognate gene uses the sealed predictions. It is descriptive.

## Other descriptive blocks

The E-MTAB ensemble is the prespecified 50/50 average, applied to the sealed vectors. The correlation of per-donor losses between CellBridge and v0.4 is Pearson, on both cohorts.

The scale-free block reports, per protein, the Pearson correlation across test donors and the fraction of donors whose prediction and outcome have the same strict sign. Medians are inside E1 and inside E2. Undefined correlations are not written as zero. Constant predictions are counted. sciPENN is also rescored after multiplying by the training-cell SD from its own `normalize_total` and `log1p` path, using `.venv_scipenn`.

E-MTAB H4 is the per-donor E1 gain over the training mean, for CellBridge and for v0.4, with Spearman correlations against each arm's cell count and against the donor's noise ceiling.

The noise ceiling resamples cells inside each test-donor arm, 200 times, seed 20261012. The floor is the mean of `sigma * sqrt(2/pi) / scale`. Each method's error is `(MAE - floor) / (training-mean MAE - floor)`. E-MTAB uses the vault matrices. GSE334503 uses prepared cells and also records the gap between the cell mean and the group-weighted truth.

The high-RNA-noise row of the existing simulation is cited as written. It is not refit.

## totalVI

`.venv_totalvi` only. Outputs stay under `*_posthoc/final/totalvi/`. GSE334503 runs first. E-MTAB waits until the E-MTAB learning curve and attribution fits are done.

Three fits use `scvi` seeds 20261008, 20261009 and 20261010, the sealed settings, and the sealed cell subsample. One fit uses the package default model and the package default `train` arguments, on CPU, with the same genes and the same cell cap. A non-finite default fit is unavailable and is not retried. The E-MTAB AnnData file is kept only if it is under 15 GiB.

The report is the E1 and E2 range across the sealed seed and these three seeds, and whether CellBridge has the lower loss on each seed. Sealed H7 and H8 remain the confirmatory comparisons.

## Venue rule

Evaluated after the learning curve and the attribution block exist. Nothing else in this protocol can change it.

The few-donor claim holds only when the E-MTAB E2 interval lies strictly above 0 at both 8 and 16 training donors, and the GSE334503 E2 point estimate at 6 donors is strictly positive.

The reproducibility claim holds only when the Spearman-difference interval lies strictly above 0 on both cohorts.

If either claim holds, the paper is framed as an Original Paper and that claim is the differentiator. If neither holds, the paper is framed as an Application Note and states that on these two cohorts the two estimators are equivalent. The outcome is written to `runs/cellbridge_v110/open/venue_decision.json` and to `paper/submission/FALLBACK.md`.
