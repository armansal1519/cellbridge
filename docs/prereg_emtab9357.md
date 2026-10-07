# Preregistration: CellBridge on E-MTAB-9357

This document was written before any protein value from E-MTAB-9357 was read.
The machine-readable protocol is `configs/protocol_cellbridge_v100_emtab9357.json`.
The author uploads both to OSF before unblinding. Until that URL exists, the
freeze records `osf_url: null`, and the paper says that the public timestamp
was not available before the test.

## Question

Does the frozen CellBridge v1.0 method, refit on this cohort's training
patients and not otherwise changed, predict held-out T-cell protein changes
from baseline (BL) to the follow-up draw (AC)?

## Cohort

E-MTAB-9357 (Su et al., Cell 2020). Patient roles were locked from metadata
on 2026-10-07 (seed 20261007; fingerprint
`ad2aa4c5d2714c2fdc9b8e8817da775f9f0ce7086d6497269a85a26e67c9a56a`): 73 train,
36 calibration, 36 test. A patient is eligible only with both a BL and an AC
sample and at least 30 RNA-labeled T cells in each arm. That RNA filter is
expected to leave at most 65, 31 and 29 patients. Unpaired healthy donors and
BL-only patients are not response donors.

## Inputs

The deposited matrices are log1p(CPM). CellBridge uses those values directly
as log-normalized RNA and log1p protein. totalVI and the v0.4 cell rows
receive the CPM (`expm1` of the deposit), because raw UMI counts were not
deposited. v0.4 uses at most 100 T cells per patient-arm, sampled with the
protocol seed. Donor-level RNA and those cell rows are limited to the 4,000
genes with the highest training-cell variance; the ridge then selects 2,000
of those by its own inner rule.

T cells are labeled from RNA markers only (CD3D, CD3E, CD3G, TRAC, and the
B, monocyte and NK sets in `subtype_rna_only`). A marker is detected when the
deposited value is positive. The winning lineage needs at least two detected
markers and a margin of 0.1. Protein and atlas annotations are not used.

Anchors, under deposited names: CD38-1, CD278_ICOS, CD279_PD-1, HLA-DR,
CD127_IL-7RA, CD27-1, CD28-1, CD45RO. Primary targets: CD25 and CD69-1.
Four isotype controls are excluded from targets. All eight anchors and both
primary targets are present, so the missing-protein rule is not triggered.

## Method

`src/responsebridge/cellbridge.py` and `cellbridge_data.py` must match the
SHA-256 values in the GSE334503 freeze. The grid is the v1.0 grid: 1000
cell-variance genes, metacell sizes 1, 5, 20 and 60, seven response weights
with no between-donor term, 25 penalties, shared top 5% by inner
leave-one-training-patient-out. No calibration or test protein is used to fit
or to select. A dry run, before the real freeze, splits the eligible training
patients only (8 train, 4 calibration, the rest as a pseudo-test) and never
opens the calibration or test vault.

## Comparators and tests

Comparators, in the fixed order H1–H10, are the training mean, the tuned
joint pseudobulk ridge, the tuned RNA pseudobulk ridge, totalVI (one
prespecified fit; query non-anchor proteins withheld), and v0.4 abundance
shrinkage refit on this cohort. A comparator that cannot be run before the
seal is skipped and does not consume alpha. The test is a two-sided sign-flip on donor-level loss differences at
alpha 0.05. With more than 16 test donors the sign flips are a 10,000-draw
sample (seed 20261007); the interval is a 10,000-draw paired bootstrap. The sequence stops at the first non-rejection.
Non-inferiority to v0.4 uses the margin 0.05. Split-conformal intervals use
the calibration patients at alpha 0.1.

## Meta-analysis

For each endpoint and each of those five comparators, the two cohorts
contribute the mean test-donor loss difference (comparator minus CellBridge).
The cohort variance is the sample variance of the donor differences divided
by the number of test donors. Report the inverse-variance fixed-effect mean
and the DerSimonian-Laird random-effects mean. This analysis is secondary
and does not change either cohort's sequence.

## What is not claimed

GSE334503 has already been unblinded and is not re-fit. Development ablations
are exploratory. Explanations are predictive, not causal. The BL-to-AC change
is not a causal effect of infection.
