# Reproduce CellBridge v1.0

Each command is run from the repository root. The count matrices and the outcome vault are not in this repository. The scripts refuse to unblind twice and refuse a prediction file that does not match its seal.

## Frozen solver

```bash
python scripts/verify_frozen.py
```

This checks `src/responsebridge/cellbridge.py` and `cellbridge_data.py` against `runs/cellbridge_v100/gse334503/freeze.json`.

## GSE334503

Obtain the prepared cohort used by the protocol, then:

```bash
python scripts/cellbridge_gse_v100.py freeze
python scripts/cellbridge_gse_v100.py fit
python scripts/cellbridge_totalvi_v100.py
python scripts/cellbridge_gse_v100.py seal
python scripts/cellbridge_gse_v100.py unblind
```

Do not rerun `unblind` on the archived result. totalVI needs scvi-tools, which is not part of the CellBridge install.

## E-MTAB-9357

The protocol and the preregistration are `configs/protocol_cellbridge_v100_emtab9357.json` and `docs/prereg_emtab9357.md`. The OSF URL was null when they were hashed.

```bash
python scripts/cellbridge_sealed_v100.py freeze
python scripts/cellbridge_sealed_v100.py fit
python scripts/cellbridge_totalvi_emtab_v100.py
python scripts/cellbridge_sealed_v100.py seal
python scripts/cellbridge_sealed_v100.py unblind
```

## Paper numbers and figures

These read the sealed outputs already stored in `results/`. They do not refit the model.

```bash
python scripts/cellbridge_meta_v100.py
python scripts/cellbridge_posthoc_v100.py
python scripts/cellbridge_numbers_v100.py
python scripts/cellbridge_paper_figures_v100.py
```

`paper/main.tex` and `paper/supplement.tex` compile with [tectonic](https://tectonic-typesetting.github.io/). The numbers in the text come from `paper/numbers.tex`.
