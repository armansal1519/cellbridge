# Post hoc comparators, v1.1

This protocol is for sciPENN and scLinear on cohorts that are already unblinded. It is not part of H1–H10. It does not change the frozen solver, `paper/numbers.tex`, or the sealed run directories.

The machine-readable copy is `configs/protocol_comparators_v110.json`. A public snapshot of this file is required before either method is fit. Until `docs/comparators_v110_timestamp.json` exists, the runner refuses to fit.

## What is compared

sciPENN 0.9.6 and the pinned scLinear core predict cell-level protein from RNA. The donor-level response is the difference of arm means of that output. The loss is standardized MAE against the sealed estimand. Predictions are not rescaled to the truth.

CellBridge RNA-only is the vector already stored in the sealed `predictions/core.npz`. It is not refit.

## Inputs

GSE334503 uses raw counts. E-MTAB-9357 uses `expm1` of the deposited log1p(CPM), for RNA and for training protein, which is the count matrix totalVI received. Query protein is not read. Training donors fit the model. Test donors are scored. Calibration donors are unused.

sciPENN keeps its published preprocessing, including highly variable genes chosen on the union of training and query RNA, and the z-scoring of proteins. The package does not return that protein scale, so the donor response uses the returned point prediction as it is. scLinear is trained on log1p protein and is given counts with its published `do_log1p=True` switch, because the default `False` expects data that are already logged. Its truncated SVD uses the published 300 components and is fit on training cells only. Cells with zero RNA counts are omitted, which is that core's published minimum-count filter, and the kept cells are recorded.

## Install

`numba<=0.50.0` does not build here (`llvmlite` 0.33, Python 3.11, arm64). sciPENN 0.9.6 is installed with `--no-deps` into `.venv_scipenn`. AnnData 0.9.2 is used because the package calls `AnnData.concatenate`. The lock file lists the packages that were actually installed. The sciPENN source is not edited.

## Inference

For each method and each of E1 and E2, a paired sign-flip and a paired bootstrap compare that method with the sealed CellBridge losses. Every record has `post_hoc: true`. A method that hits the three-hour cap is unavailable. It is not given a favourable number.

## Dry run

`Lawlor_CD3_CD28_T`, at most 200 cells per donor-arm, two sciPENN epochs. The receipt says the path finished. Those numbers are not a result.

## Reporting

The supplementary table contains every scored row, in either direction. A main-text mention has to say post hoc. These tests do not reopen the fixed sequence.
