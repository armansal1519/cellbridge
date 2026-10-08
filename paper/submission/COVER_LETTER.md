# Cover letter

The letter is written by the author. These are the points it has to cover.

- The paper is an Original Paper for Bioinformatics. It reports a closed-form method for a donor-level protein response, with a prespecified test on two public cohorts.
- Leave-one-donor-out is used only to choose settings inside the training donors. GSE334503 has 11 training donors. Each inner fold holds out one training donor, and cells from that donor are not split. Calibration donors are used only for the intervals. The reported error is the sealed test. No test donor is in any fold. This is the reason the paper uses leave-one-donor-out. The journal rejects leave-one-out unless that reason is given.
- totalVI was run once, before the outcome vault was opened. Two settings differ from the package default: the counts were `round(expm1(x) / 100)`, and the learning rate was \(4\times 10^{-4}\) rather than \(4\times 10^{-3}\).
- A language model was used for code, figure scripts, the reference search, software documentation, formal statements, and an earlier draft that is not submitted. The manuscript text is written by the author. The log is in the supplement.
- Say whether a preprint of this manuscript exists. None is recorded in the repository.
- Confirm funding. The draft sentence is "This work received no specific funding."
- Confirm that no conflict of interest is declared, or replace that sentence.
- Give the CRediT roles.
- Give the ORCID iD and, if you have one, an institutional email.
- The software archive is the Software Heritage snapshot named in the abstract. A Zenodo DOI is not assigned yet. You can add it later by turning on the Zenodo integration and publishing a release.
- The sealed tests do not include sciPENN or scLinear. Do not claim that they do.
