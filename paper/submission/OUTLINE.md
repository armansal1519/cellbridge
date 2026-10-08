# Outline

Write the scientific sections from these bullets. Do not paste this file into the manuscript. Every number in the manuscript must be a macro from `paper/numbers.tex`. The p-value macros already include the operator, so write `\(p\GseHOneP\)`, which becomes \(p=0.002\) or \(p<0.001\). Do not write `\(p=\GseHOneP\)`.

Word budget from `PAGE_FIT.md`. Captions are extra: about 45 words for a figure and 30 for a table.

Main-text figures: 1 (`fig:model`), 3 (`fig:gse`), 4 (`fig:emtab`), 5 (`fig:both`).
Supplementary figures: development (`fig:dev`), panel (`fig:panel`), per-protein (`fig:perprotein`), simulation (`fig:simulation`), totalVI traces (`fig:totalvi`), gene curve (`fig:genes`).

## Claims to avoid

- CellBridge is better than abundance shrinkage. On GSE334503, H9 was tested and was not rejected. On E-MTAB-9357, H9 and H10 were not tested.
- H7 or H8 was rejected on E-MTAB-9357. The sequence stopped at H4. Later rows were recorded and were not tested.
- H10 was tested on GSE334503. It was not. The sequence stopped at H9.
- A feature contribution is a causal effect of measuring an antibody.
- There is a public preregistration timestamp. The OSF URL was null when the E-MTAB-9357 protocol was hashed.
- The sealed tests include sciPENN or scLinear. They do not. Those methods can be named as related work only.
- RNA-only was a prespecified hypothesis. It was a descriptive ablation.
- An E-MTAB-9357 CellBridge wall-clock time. No such macro exists.
- The development scores in Supplementary Figure 1 are the sealed result.

## Abstract (150 words at most)

Five headings, in this order: Motivation, Results, Availability and implementation, Contact, Supplementary information.

- Motivation: donor-level protein change, RNA plus eight antibodies, not cell-level imputation.
- Results: name both cohorts. GSE334503 primary claim `\GseClaim`. E-MTAB-9357 primary claim `\EmClaim`. Use `\MaeGseCbEone`, `\MaeGseCbEtwo`, `\MaeEmCbEone`, `\MaeEmCbEtwo` if they fit. Do not add numbers that are not macros.
- Availability: GitHub `https://github.com/armansal1519/cellbridge`, tag `v1.0.1`. Release `swh:1:rel:03348d00ccd9f724d2ca51961f4bca30eaeb0559`. Snapshot `swh:1:snp:f854154bf0eacc86ed2cb29c5911f8c4fcfdacc4`. Contact is armansal1519@gmail.com until an institutional address exists.
- Supplementary information: one line pointing to the supplement.

Candidate citations are not required in the abstract.

## Introduction (700)

### What is measured (about 160)

- CITE-seq and REAP-seq read RNA and surface protein in the same cell. A panel is a small set of antibodies. Cell hashing is a separate use of antibodies.
- The quantity in this paper is the change between two conditions, averaged in RNA-labelled T cells of one donor. It is not an imputed protein count in one cell.
- Citations: `stoeckius2017`, `peterson2017`, `mimitou2019`, `stoeckius2018`.

### What existing models predict (about 180)

- totalVI, MultiVI, MOFA+, scVI and the Seurat bridge models integrate or impute modalities. cTP-net, sciPENN, scVAEIT and BABEL impute protein from RNA. scLinear is a linear protein predictor.
- Perturbation models (scGen, CPA, GEARS, CellOT) predict a shift, usually at cell level. A linear baseline can match more elaborate perturbation predictors.
- None of those statements is a claim that CellBridge wins a sealed comparison against sciPENN or scLinear.
- Citations: `gayoso2021`, `ashuach2023`, `argelaguet2020`, `lopez2018`, `stuart2019`, `hao2021`, `hao2024`, `zhou2020`, `lakkis2022`, `du2022`, `wu2021`, `hanhart2024`, `lotfollahi2019`, `lotfollahi2023`, `roohani2024`, `bunne2023`, `ahlmann2025`. Cite a subset. Do not cite a key that is missing from `references.bib`.

### The donor is the unit (about 160)

- Treating cells as independent replicates inflates evidence. Pseudobulk and mixed models are the usual correction. Metacells (MetaCell, SEACells, SuperCell) are a related aggregation, not the estimand here.
- Citations: `squair2021`, `crowell2020`, `zimmerman2021`, `junttila2022`, `baran2019`, `persad2023`, `bilous2022`.

### This paper (about 200)

- One slope per protein. Donor effects removed. The fit is closed form, so a prediction is a sum of feature contributions.
- Eight anchors are observed in both arms. The other proteins are hidden for calibration and test donors.
- Two sealed cohorts, Table `tab:cohorts`. GSE334503 is typhoid vaccination. E-MTAB-9357 is a COVID-19 follow-up. Lawlor is development only and was already exposed.
- State the primary claim as the fixed sequence H1 to H6, and say that abundance shrinkage is a comparator, not a second method. v0.4 is that comparator.
- Point to Figure `fig:model`.
- Citations: `su2020`, `king2026data`, `lawlor2021`, `kotliarov2020`, `mathew2020`.

## Methods (1,400, excluding the subsection below)

### Estimand (about 180)

- Per donor, the RNA-defined T-cell mean of cellwise log1p raw ADT counts. Protein is not used to call a cell a T cell.
- GSE334503 contrast: Typhim Vi day 7 minus day 0. Control label D0, perturbed label TyphimVi_day7. At least 30 cells per arm.
- E-MTAB-9357 contrast: follow-up (AC) minus baseline (BL). Deposited values are log1p(CPM). CellBridge uses those values. The ridge comparators and abundance shrinkage use CPM reconstructed by expm1.
- E1 is CD25 and CD69. E2 is all 122 non-anchor proteins. No outcome-based filtering.
- Metric: per donor, absolute error divided by the training-donor response standard deviation of that protein, averaged over the proteins in the endpoint, then over test donors.
- Citations: `cibrian2017` for CD69, `malek2010` for IL-2 and CD25. Use them only if you discuss those proteins biologically.

### Cohorts and roles (about 220)

- Point to Table `tab:cohorts`.
- GSE334503: `\NGseTrain` train, `\NGseCal` calibration, `\NGseTest` test. Roles were locked before protein outcomes were used for evaluation.
- E-MTAB-9357: `\NEmTrain` / `\NEmCal` / `\NEmTest`. Eligibility is both draws and at least 30 T cells in each arm. Roles were locked from metadata.
- Anchors, the same eight after name aliasing on E-MTAB-9357: CD38, ICOS, PD-1, HLA-DR, CD127, CD27, CD28, CD45RO.
- Calibration donors are used only for the intervals. Test donors are not used to fit or to choose settings.
- Dataset citations: `king2026data`, `emtab9357data`, `lawlor2021data` for the development matrices. The Lawlor paper is `lawlor2021`.

### Features (about 160)

- 1,000 genes, chosen on training donors from RNA only. Cognate genes of the panel proteins are forced in.
- Two technical columns: log1p of RNA UMIs, and log1p of the number of RNA genes detected.
- Column order: genes, then the two technical columns, then the anchors. `n_genes` counts genes only.
- Feature sets in the grid: RNA, and RNA plus anchors.

### The fit (about 250)

- The cell equation in Figure `fig:model` panel b: a donor intercept, a shared intercept, and one slope vector. Donor effects are removed before the slope is fit.
- Ridge penalty. `\rho` mixes cell-level and donor-level information. The published grid sets the second mix weight, tau, to 0. The software default also varies tau. Say which grid the sealed fits used: the protocol grid, tau fixed at 0.
- Metacell sizes `k` in {1, 5, 20, 60}.
- Lambdas: 25 values, log-spaced from \(10^{6}\) to \(10^{-2}\).
- The slope has a closed form. The Woodbury identity is the algebraic step. Measurement error is why a donor mean is not the same object as a cell.
- A prediction is an exact sum of feature contributions. That is an attribution of the fitted prediction, not a causal effect.
- Citations: `hoerl1970`, `hager1989`, `fuller1987`.

### Choosing the settings (about 180)

- Inner leave-one-training-donor-out over the full grid.
- Strategy `shared_top`, top 5 percent. The best 5 percent of configurations are averaged. An average of linear fits is linear, so the contributions still add up.
- The score for that choice is target-pooled standardized MAE on the inner holdout.
- No calibration outcome and no test outcome enters the choice.
- The detailed split is the next subsection. Do not repeat the whole argument here.
- Citations: `varma2006`, `kapoor2023`.

### Comparators (about 200)

- Training-donor mean response.
- RNA ridge: pseudobulk response ridge on 2,000 response-variance genes. Alpha in {0.1, 1, 10, 100, none}, chosen by inner leave-one-donor-out.
- Joint ridge: the same, plus the anchor responses.
- Abundance shrinkage: the ResponseBridge 0.4 estimator. Nested alpha and shrinkage, inner leave-one-donor-out. It is a comparator.
- totalVI: one prespecified scvi-tools fit. Query donors' non-anchor ADTs are missing. Deviations from the package default, fixed from the training loss before the vault was opened: counts were `round(expm1(x) / 100)`, and the learning rate was `\EmLr` rather than \(4\times 10^{-3}\).
- GSE334503 totalVI used `\GseTvCells` training cells. E-MTAB-9357 totalVI used `\EmTvCells` cells and `\EmTvGenes` genes.
- A hypothesis whose comparator was not run before the seal is removed and does not consume alpha. Say this, because it is why a missing comparator would not be tested.
- Citations: `gayoso2021`, `gayoso2022`.

### The sequence (about 210)

- Two-sided alpha 0.05. Fixed sequence. A hypothesis is tested only if every earlier null was rejected.
- Order, endpoint, comparator:
  - H1, E2, training mean
  - H2, E2, joint ridge
  - H3, E2, RNA ridge
  - H4, E1, training mean
  - H5, E1, joint ridge
  - H6, E1, RNA ridge
  - H7, E2, totalVI
  - H8, E1, totalVI
  - H9, E2, abundance shrinkage
  - H10, E1, abundance shrinkage
- The primary claim is H1 to H6, all rejected.
- GSE334503 uses an exact sign-flip over the `\NGseTest` test donors and a paired donor bootstrap with 10,000 draws.
- E-MTAB-9357 uses the same sign-flip function: exact enumeration up to 16 test donors, otherwise 10,000 random sign flips with seed 20261007. There are `\NEmTest` test donors, so the random version is the one that applies. The interval is a paired donor bootstrap with 10,000 draws.
- Non-inferiority against abundance shrinkage is separate from the sequence. The margin is 0.05 on the standardized MAE scale. Report the upper end of the interval and whether it held. Do not call this superiority.
- Split-conformal intervals use the calibration donors only. Nominal level 90 percent (`alpha` 0.1). The quantile is the higher method.
- The two-cohort combination is a random-effects summary. Report fixed and random estimates from the macros. They match for several comparators because there are two cohorts.
- Citations: `westfall2001`, `hemerik2020`, `efron1979`, `lei2018`, `angelopoulos2023`, `piaggio2012`, `higgins2002`, `dersimonian1986`.

## Training, cross-validation and the independent test (300)

This is the subsection the journal asks for. The performance numbers do not belong here. They belong in Results.

- State the three roles and the counts: GSE334503 `\NGseTrain` / `\NGseCal` / `\NGseTest`; E-MTAB-9357 `\NEmTrain` / `\NEmCal` / `\NEmTest`.
- Training donors are the only donors in the fit and in the inner loop.
- The inner loop is leave-one-donor-out. Each fold holds out one training donor. The fold is grouped by donor. Cells from one donor are not split across the inner fit and the inner holdout.
- Why this is not the leave-one-out the journal rejects: the unit of generalization is the donor; GSE334503 has only `\NGseTrain` training donors, so holding out one donor is the grouped cross-validation that uses each training donor once; the inner scores choose settings only; the reported error is the sealed test.
- No test donor appears in any inner fold, in gene selection, or in the fit.
- Calibration donors are not used to choose `k`, `\rho`, lambda, or the feature set. They are used for the conformal scores.
- Test donors stay sealed until the prediction file is written. The GSE334503 seal time on the laptop clock is 2026-10-07T12:46:56.735401+00:00. Say that this is a local clock time, not a public registry time.
- E-MTAB-9357 protocol was hashed before unblinding. There is no public OSF timestamp.
- Nested selection is the point of `varma2006`. Leakage across donors is the point of `kapoor2023` and `whalen2022`. Preregistration, and the missing public timestamp, is the point of `nosek2018`.
- Do not describe the inner loop as an estimate of test error.

## Implementation (350)

- The public package is CellBridge 1.0.1, MIT licence, `https://github.com/armansal1519/cellbridge`. The archived snapshot identifier is filled in with the Software Heritage identifier.
- `fit` takes donor arm matrices. `fit_anndata` is optional and does not require AnnData to be installed. `predict`, `contributions` and `intervals` are methods of the fitted object.
- The defaults of `fit` are not the paper grid. The sealed fits pass `k` in {1, 5, 20, 60}, the seven `\rho` values with tau 0, the 25 lambdas, `strategy="shared_top"` and `top=0.05`.
- At least three donors are required. Intervals require nonnegative finite calibration scores and `alpha` in (0, 1).
- Document the optional parameters by pointing to `docs/api.md` in the repository. The journal asks for every optional parameter.
- A synthetic test with a fixed seed and its expected output is in `examples/`.
- Runtime, one laptop: CellBridge on GSE334503 `\GseCbSeconds` seconds. totalVI on that cohort `\GseTvSeconds` seconds (`\GseTvMinutes` minutes). totalVI on E-MTAB-9357 `\EmTvSeconds` seconds (`\EmTvMinutes` minutes). Do not invent a CellBridge time for E-MTAB-9357.
- The solver files are frozen. `scripts/verify_frozen.py` checks them.
- Citations: `harris2020`, `virtanen2020`, `pedregosa2011`, `mckinney2010`, `hunter2007`, `gayoso2022`, and the software key `salehi2026` once the archive identifier is in the bib file.

## Results (1,200)

### GSE334503 (about 360)

- Point to Figure `fig:gse` and Table `tab:main`.
- CellBridge E1 `\MaeGseCbEone`, E2 `\MaeGseCbEtwo`.
- The sequence rejected H1 through H8. Give the gain, the interval and the p-value from the macros, or send the reader to the figure if the word budget breaks.
  - H1: gain `\GseHOneGain`, interval `\GseHOneLo` to `\GseHOneHi`, `\(p\GseHOneP\)`, wins `\GseHOneWins` of `\GseHOneN`.
  - H2: `\GseHTwoGain`, `\GseHTwoLo` to `\GseHTwoHi`, `\(p\GseHTwoP\)`, `\GseHTwoWins` of `\GseHTwoN`.
  - H3: `\GseHThreeGain`, `\GseHThreeLo` to `\GseHThreeHi`, `\(p\GseHThreeP\)`, `\GseHThreeWins` of `\GseHThreeN`.
  - H4: `\GseHFourGain`, `\GseHFourLo` to `\GseHFourHi`, `\(p\GseHFourP\)`, `\GseHFourWins` of `\GseHFourN`.
  - H5: `\GseHFiveGain`, `\GseHFiveLo` to `\GseHFiveHi`, `\(p\GseHFiveP\)`, `\GseHFiveWins` of `\GseHFiveN`.
  - H6: `\GseHSixGain`, `\GseHSixLo` to `\GseHSixHi`, `\(p\GseHSixP\)`, `\GseHSixWins` of `\GseHSixN`.
  - H7: `\GseHSevenGain`, `\GseHSevenLo` to `\GseHSevenHi`, `\(p\GseHSevenP\)`, `\GseHSevenWins` of `\GseHSevenN`.
  - H8: `\GseHEightGain`, `\GseHEightLo` to `\GseHEightHi`, `\(p\GseHEightP\)`, `\GseHEightWins` of `\GseHEightN`.
- H9 was tested and was not rejected: gain `\GseHNineGain`, interval `\GseHNineLo` to `\GseHNineHi`, `\(p\GseHNineP\)`, wins `\GseHNineWins` of `\GseHNineN`. The interval includes values below zero.
- H10 was not tested. Do not quote `\GseHTenP` as a test result. The macros exist because the contrast was recorded.
- Primary claim: `\GseClaim`.
- Non-inferiority versus abundance shrinkage: E1 upper `\GseNiEoneUpper`, `\GseNiEoneHold` (this is the margin of 0.05, so say it sits on the boundary). E2 upper `\GseNiEtwoUpper`, `\GseNiEtwoHold`.
- Conformal coverage `\GseCovCb`. The nominal level is 0.90. Do not type 0.90 if you can say "the nominal 90 percent level" without a bare decimal in Results. The checker flags bare decimals in Results. Prefer macros. The nominal level is not a macro; write "90 percent" in words.
- Table comparators, same order as the table: abundance shrinkage `\MaeGseAbEone`, `\MaeGseAbEtwo`; totalVI `\MaeGseTvEone`, `\MaeGseTvEtwo`; RNA-only `\MaeGseRoEone`, `\MaeGseRoEtwo`; joint ridge `\MaeGseJtEone`, `\MaeGseJtEtwo`; RNA ridge `\MaeGseRnEone`, `\MaeGseRnEtwo`; training mean `\MaeGseMnEone`, `\MaeGseMnEtwo`.
- Panel c is predicted against observed CD25 and CD69. Do not read a correlation off the figure.

### E-MTAB-9357 (about 360)

- Point to Figure `fig:emtab` and the same table.
- CellBridge E1 `\MaeEmCbEone`, E2 `\MaeEmCbEtwo`.
- H1, H2 and H3 were rejected. The joint ridge, the RNA ridge and the training mean have the same E1 and the same E2 in the table (`\MaeEmJtEone`, `\MaeEmRnEone`, `\MaeEmMnEone` are equal, and the E2 triple is equal). Say that the ridge fits matched the training mean on this cohort, so those three contrasts are the same contrast.
  - H1: `\EmHOneGain`, `\EmHOneLo` to `\EmHOneHi`, `\(p\EmHOneP\)`, `\EmHOneWins` of `\EmHOneN`.
  - H2 and H3 use the same recorded gain and the same p-value as H1.
- The sequence stopped at `\EmStopped`. H4 was tested and was not rejected: gain `\EmHFourGain`, interval `\EmHFourLo` to `\EmHFourHi`, `\(p\EmHFourP\)`, wins `\EmHFourWins` of `\EmHFourN`.
- H5 through H10 were not tested. Do not say H7 or H8 was rejected. Their macros are records, not tests. `\EmHSevenTested` and `\EmHEightTested` say they were not tested in sequence.
- Primary claim: `\EmClaim`.
- Non-inferiority: E1 upper `\EmNiEoneUpper`, `\EmNiEoneHold`. E2 upper `\EmNiEtwoUpper`, `\EmNiEtwoHold`.
- Abundance shrinkage `\MaeEmAbEone`, `\MaeEmAbEtwo`. totalVI `\MaeEmTvEone`, `\MaeEmTvEtwo`, which is worse than the training mean `\MaeEmMnEone`, `\MaeEmMnEtwo`.
- RNA-only is descriptive: E2 `\MaeEmRoEtwo`, E1 `\MaeEmRoEone`. On this cohort the RNA-only fit sits on the training mean. Panel d of Figure `fig:emtab` connects CellBridge to RNA-only. The anchors carry the gain. That sentence is descriptive, not a hypothesis test.
- Some totalVI gains are above the axis. The corner of panel a counts them. Do not hard-code that count.
- Conformal coverage `\EmCovCb`.

### Both cohorts (about 280)

- Point to Figure `fig:both`.
- Random-effects gain versus the training mean: `\MetaMnRe` with standard error `\MetaMnReSe`. The fixed-effect value is `\MetaMnFe` (standard error `\MetaMnSe`).
- Versus the joint ridge: `\MetaJtRe`, `\MetaJtReSe`.
- Versus the RNA ridge: `\MetaRnRe`, `\MetaRnReSe`.
- Versus totalVI: random effect `\MetaTvRe` with `\MetaTvReSe`. The fixed effect `\MetaTvFe` is not the same number. Do not treat them as interchangeable. GSE334503 has no totalVI coverage bar in panel b.
- Versus abundance shrinkage: `\MetaAbRe` with `\MetaAbReSe`. This is a small combined difference. It is not a superiority claim.
- Panel c: the three fit times already listed under Implementation. Use the macros once, not twice, if the word budget is tight.
- Coverage panel: CellBridge `\GseCovCb` and `\EmCovCb` against the nominal 90 percent level.

### What is only in the supplement (about 200)

- Development, nine tasks, Supplementary Figure `fig:dev`. Means: CellBridge `\DevCb`, abundance shrinkage `\DevAb`, joint ridge `\DevJt`, RNA ridge `\DevRn`, training mean `\DevMn`. These are not the sealed tests.
- Ablation means: full `\AblFull`, anchors in every fit `\AblAnchor`, `k` fixed at 1 `\AblKone`, `\rho` fixed at 0 `\AblRho`, RNA only `\AblRna`.
- Panel c of that figure is the minimum-to-maximum band across tasks. The means at 0, 1, 2, 4 and 8 anchors are `\PanelZero`, `\PanelOne`, `\PanelTwo`, `\PanelFour`, `\PanelEight`.
- Supplementary Figure `fig:panel` uses the same panel curve with a 25th-to-75th percentile band, not the min-max band. Panel b is the GSE334503 test donors. Panel c: the largest feature is an anchor in `\BootCdOneFiveFour` percent of resamples for CD154, `\BootCdTwentyFive` percent for CD25, and `\BootCdSixtyNine` percent for CD69.
- Do not type the CD25 or CD69 anchor-share lines as decimals. They are drawn from the source table and they are not macros.
- Simulation, Supplementary Figure `fig:simulation`: recovered slope relative to truth `\SimCells`, `\SimKmeans`, `\SimOracle` at the plotted noise level. Conformal coverage in that simulation `\SimCov`. Scaling `\ScaleMillionSeconds` seconds at the plotted large size. Say these are simulations.
- Per-protein curves and the totalVI ELBO traces are descriptive. Gene-count curve is a reduced grid, not the sealed grid.

## Discussion (600)

### What the sequence says (about 200)

- GSE334503 supports the primary claim. E-MTAB-9357 does not. The same frozen rule was refit. It was not otherwise changed.
- The stop on E-MTAB-9357 is at the CD25 and CD69 comparison with the training mean, not at totalVI and not at abundance shrinkage.
- Non-inferiority held on both E2 endpoints and on GSE334503 E1 at the margin. It did not hold on E-MTAB-9357 E1.

### What the anchors are doing (about 160)

- On E-MTAB-9357 the RNA-only fit does not leave the training mean, and the eight anchors account for the difference from that mean. Say this is predictive.
- The bootstrap percentages are about the fitted prediction on the development attribution, not a reason to add or drop an antibody in a new experiment.
- CD25 and CD69 are activation-associated proteins. Keep any biological sentence inside `cibrian2017`, `malek2010`, `kotliarov2020` and `mathew2020`, and do not invent a mechanism.

### Limits (about 240)

- Both test sets are small: `\NGseTest` and `\NEmTest` donors.
- The inner loop has `\NGseTrain` training donors on GSE334503. That is why the grouped leave-one-donor-out is used, and why the sealed donors are the result that matters.
- totalVI was one prespecified fit, with the two deviations named in Methods. A different totalVI setting was not tried after unblinding.
- sciPENN and scLinear were not in the sealed comparator list.
- Lawlor could not have been a sealed test. It had already been used for development.
- There is no public preregistration timestamp.
- Contributions are not causal.
- A Zenodo DOI for the software is not assigned yet. The Software Heritage identifier is the archive that exists. The author can add a Zenodo DOI later.

## Figure and table legends

Write these. The alternative text is already under each figure.

### Table `tab:cohorts` (about 30)

- Columns are the two cohorts. Rows are the contrast, the lineage, the three role counts (use the macros), the anchor rule, and the primary proteins.

### Table `tab:main` (about 30)

- Standardized MAE. Rows are CellBridge, abundance shrinkage, totalVI, RNA-only CellBridge, joint ridge, RNA ridge and the training mean. Columns are E1 and E2 in each cohort. The numbers are already the macros in the table. Do not retype them.

### Figure `fig:model`

- Panel a: three steps. Both arms observed for RNA and eight anchors. Donor-level change. Hidden proteins.
- Panel b: the equation. Donor effects removed. One slope. Closed-form ridge.
- Panel c: one CD69 prediction. Teal bars are anchors. Grey bars are genes. The donor is named on the panel. Do not add a donor name that is not on the panel.

### Figure `fig:gse`

- Panel a: donor-level gains of CellBridge over the five comparators, all hidden proteins, GSE334503 test donors. A gain above zero means CellBridge has the smaller error.
- Panel b: H1 to H10. Teal, rejected. Orange, tested and not rejected. Grey, not tested.
- Panel c: predicted against observed CD25 and CD69.

### Figure `fig:emtab`

- Same layout for `\NEmTest` test donors.
- Panel b stops at H4.
- Panel d: each donor's CellBridge error connected to its RNA-only error. A black line marks the mean.
- Panel a counts gains above the displayed range in the corner. Do not write the count as a word you read off once. Say that the corner count is the number of gains above the displayed range.

### Figure `fig:both`

- Panel a: donor-level gains in both cohorts and the random-effects estimate.
- Panel b: split-conformal coverage. The dashed line is the nominal 90 percent level. The GSE334503 totalVI bar is absent.
- Panel c: fit time on a log scale.

### Supplementary Figure `fig:dev`

- Panel a: five methods, nine development tasks. The horizontal mark is the mean.
- Panel b: the ablation and the two pseudobulk baselines named in the panel.
- Panel c: error against anchors from 0 to 8. The band is the minimum to the maximum across tasks, not an interquartile range.

### Supplementary Figure `fig:panel`

- Panel a: mean error against anchors. The band is the 25th to 75th percentile.
- Panel b: histogram of anchor share on the GSE334503 test donors. Lines mark CD25 and CD69.
- Panel c: percent of bootstrap draws in which the largest feature is an anchor. Use `\BootCdOneFiveFour`, `\BootCdTwentyFive`, `\BootCdSixtyNine`.

### Supplementary Figures `fig:perprotein`, `fig:simulation`, `fig:totalvi`, `fig:genes`

- Per protein: proteins ordered by error, CD25 and CD69 marked, one panel per cohort.
- Simulation: recovered slope for single cells, k-means and a shared state; seconds against cell count.
- totalVI: training and validation ELBO, one panel per cohort.
- Genes: mean error against gene count on the reduced grid. This figure is one column wide.

## Author tasks that are not prose

- Replace the affiliation placeholders: department, institution, street, postal code, city, country.
- Add an ORCID iD. An institutional email is preferred if you have one.
- Confirm the funding sentence and the conflict-of-interest sentence.
- Confirm the CRediT roles for the cover letter.
- After the Software Heritage identifier is filled in, read the availability sentences once more.
