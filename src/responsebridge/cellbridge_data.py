"""Cell tables for CellBridge with a hidden-target guard.

Hidden donors expose RNA and anchor ADTs (observed query inputs in both arms)
but never target ADT columns. Gene selection uses RNA only.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse

from .cellbridge import donor_stats

COGNATE = {
    "CD11a": "ITGAL", "CD11b": "ITGAM", "CD11c": "ITGAX", "CD123": "IL3RA", "CD127": "IL7R",
    "CD14": "CD14", "CD16": "FCGR3A", "CD161": "KLRB1", "CD183": "CXCR3", "CD185": "CXCR5",
    "CD19": "CD19", "CD194": "CCR4", "CD195": "CCR5", "CD196": "CCR6", "CD197": "CCR7",
    "CD20": "MS4A1", "CD21": "CR2", "CD22": "CD22", "CD235": "GYPA", "CD24": "CD24",
    "CD25": "IL2RA", "CD27": "CD27", "ICOS": "ICOS", "PD-1": "PDCD1", "CD28": "CD28",
    "CD3": "CD3E", "CD34": "CD34", "CD38": "CD38", "CD4": "CD4", "CD45RA": "PTPRC",
    "CD45RO": "PTPRC", "CD45": "PTPRC", "CD56": "NCAM1", "CD57": "B3GAT1",
    "CD66ace": "CEACAM1", "CD69": "CD69", "CD79b": "CD79B", "CD8a": "CD8A", "CD8": "CD8A",
    "HLA-DR": "HLA-DRA", "IgD": "IGHD", "IgM": "IGHM", "CD62L": "SELL", "CD86": "CD86",
    "CD40": "CD40", "CD154": "CD40LG", "CD52": "CD52", "CD33": "CD33", "CD7": "CD7",
    "CD105": "ENG", "CD44": "CD44", "CD49f": "ITGA6", "CD31": "PECAM1", "CD146": "MCAM",
    "CD5": "CD5", "CD32": "FCGR2A", "CD47": "CD47", "CD48": "CD48", "CD2": "CD2",
    "CD1c": "CD1C", "CD64": "FCGR1A", "CD1d": "CD1D", "CD35": "CR1", "CD39": "ENTPD1",
    "CX3CR1": "CX3CR1", "CD18": "ITGB2", "CD29": "ITGB1", "CD49b": "ITGA2", "CD71": "TFRC",
    "CD26": "DPP4", "CD36": "CD36", "CD49a": "ITGA1", "CD49d": "ITGA4", "CD99": "CD99",
    "CLEC12A": "CLEC12A", "CD94": "KLRD1", "CD23": "FCER2", "GPR56": "ADGRG1",
    "HLA-E": "HLA-E", "CD82": "CD82", "CD163": "CD163", "CD83": "CD83", "CD13": "ANPEP",
    "CD41": "ITGA2B", "CD42b": "GP1BA", "CD54": "ICAM1", "CD224": "GGT1",
    "anti-mouse/human CD44": "CD44", "anti-human/mouse CD49f": "ITGA6",
    "anti-human/mouse integrin β7": "ITGB7", "HLA-A;B;C": "HLA-A", "TCR α/β": "TRAC",
    "Ig light chain κ": "IGKC", "Ig light chain λ": "IGLC2", "CD57 Recombinant": "B3GAT1",
}


def cognate_gene(protein, genes):
    genes = set(genes)
    name = str(protein)
    base = re.sub(r"\.\d+$", "", name.split(" (")[0]).strip()
    for candidate in (COGNATE.get(name), COGNATE.get(base)):
        if candidate and candidate in genes:
            return candidate
    inner = re.findall(r"\(([^)]*)\)", name)
    for token in (inner[0].replace(";", " ").split() if inner else []):
        if token in genes:
            return token
    if base in genes:
        return base
    return None


def _lognorm(counts):
    x = sparse.csr_matrix(counts, dtype=np.float32)
    total = np.asarray(x.sum(axis=1)).ravel()
    if np.any(total <= 0):
        raise ValueError("Zero RNA library")
    x = sparse.diags((1e4 / total).astype(np.float32)) @ x
    x = sparse.csr_matrix(x)
    np.log1p(x.data, out=x.data)
    return x, total


class CellTable:
    def __init__(self, prepared, lineage, control, perturbed, *, donors=None, hidden_donors=()):
        prepared = Path(prepared)
        obs = pd.read_csv(prepared / "obs.csv", usecols=["donor", "lineage", "condition", "is_control"])
        ctrl = obs.is_control.astype(str).str.lower().isin(["true", "1"]) & (obs.condition.astype(str) == str(control))
        pert = obs.condition.astype(str) == str(perturbed)
        keep = (obs.lineage.astype(str) == str(lineage)) & (ctrl | pert)
        if donors is not None:
            keep &= obs.donor.astype(str).isin([str(d) for d in donors])
        self.rows = np.flatnonzero(keep.to_numpy())
        self.obs = obs.iloc[self.rows].reset_index(drop=True)
        self.obs["arm"] = np.where(pert.to_numpy()[self.rows], "perturbed", "control")
        self.genes = np.asarray(json.loads((prepared / "genes.json").read_text()), dtype=str)
        self.proteins = np.asarray(json.loads((prepared / "proteins.json").read_text()), dtype=str)
        rna = sparse.load_npz(prepared / "rna_counts.npz").tocsr()[self.rows]
        self.n_genes_detected = np.asarray((rna > 0).sum(axis=1)).ravel()
        self.rna, self.library = _lognorm(rna)
        del rna
        self._adt = np.load(prepared / "protein_counts.npy", mmap_mode="r")
        self.hidden = {str(d) for d in hidden_donors}
        self.donors = sorted(set(self.obs.donor.astype(str)))

    def protein_index(self, names):
        lookup = {p: i for i, p in enumerate(self.proteins)}
        missing = [n for n in names if n not in lookup]
        if missing:
            raise ValueError(f"Missing proteins: {missing}")
        return np.array([lookup[n] for n in names], dtype=int)

    def cells(self, donor, arm):
        return np.flatnonzero(((self.obs.donor.astype(str) == str(donor)) & (self.obs.arm == arm)).to_numpy())

    def _adt_log(self, local_rows, cols):
        rows = self.rows[local_rows]
        if not len(cols):
            return np.empty((len(rows), 0))
        return np.log1p(np.asarray(self._adt[np.ix_(rows, cols)], dtype=np.float64))

    def inputs(self, donor, arm, genes, anchors):
        """RNA (selected genes), RNA technical covariates and anchor ADTs."""
        idx = self.cells(donor, arm)
        x = self.rna[idx][:, genes].toarray().astype(np.float64)
        tech = np.column_stack([np.log1p(self.library[idx]), np.log1p(self.n_genes_detected[idx])])
        return np.hstack([x, tech, self._adt_log(idx, anchors)])

    def outcomes(self, donor, arm, targets):
        if str(donor) in self.hidden:
            raise PermissionError(f"Target ADTs of hidden donor {donor} are sealed")
        return self._adt_log(self.cells(donor, arm), targets)

    def adt_counts(self, donor, arm, cols, *, anchors):
        """Raw ADT counts; hidden donors only for the declared anchor columns."""
        cols = np.asarray(cols, dtype=int)
        if str(donor) in self.hidden and not set(cols.tolist()) <= set(np.asarray(anchors).tolist()):
            raise PermissionError(f"Only anchor ADTs of hidden donor {donor} are readable")
        rows = self.rows[self.cells(donor, arm)]
        return np.asarray(self._adt[np.ix_(rows, cols)], dtype=np.float32)

    def pseudobulk_rna_response(self, donor):
        """Mean log-normalised RNA (all genes), perturbed minus control."""
        out = []
        for arm in ("perturbed", "control"):
            idx = self.cells(donor, arm)
            out.append(np.asarray(self.rna[idx].mean(axis=0), dtype=np.float64).ravel())
        return out[0] - out[1]

    def select_genes(self, n, *, donors, forced=(), min_detection=0.01, rule="cell"):
        """RNA-only gene selection on the given donors' cells.

        ``cell``: top variance of log-normalised expression across cells.
        ``union``: half by cell variance, half by variance of donor-arm means.
        """
        mask = self.obs.donor.astype(str).isin([str(d) for d in donors]).to_numpy()
        rows = np.flatnonzero(mask)
        x = self.rna[rows]
        mean = np.asarray(x.mean(axis=0)).ravel()
        sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
        detected = np.asarray((x > 0).mean(axis=0)).ravel()
        var = np.where(detected >= min_detection, sq - mean ** 2, -np.inf)
        if rule == "cell":
            top = list(np.argsort(-var, kind="stable")[:n])
        elif rule == "union":
            keys = (self.obs.donor.astype(str) + "|" + self.obs.arm).to_numpy()[rows]
            _, inv = np.unique(keys, return_inverse=True)
            s = sparse.csr_matrix((np.ones(len(rows)), (inv, np.arange(len(rows)))))
            s = sparse.diags(1 / np.asarray(s.sum(1)).ravel()) @ s
            means = np.asarray((s @ x).todense())
            between = np.where(detected >= min_detection, means.var(0), -np.inf)
            top = list(np.argsort(-var, kind="stable")[:n // 2])
            for g in np.argsort(-between, kind="stable"):
                if len(top) >= n:
                    break
                if g not in top:
                    top.append(g)
        else:
            raise ValueError(f"Unknown gene rule {rule}")
        lookup = {g: i for i, g in enumerate(self.genes)}
        for g in forced:
            if g in lookup and lookup[g] not in top and np.isfinite(var[lookup[g]]):
                top.append(lookup[g])
        return np.array(sorted(top), dtype=int)


def feature_layout(n_genes, n_anchors):
    rna = np.arange(n_genes + 2)
    return {"rna": rna, "rna_anchor": np.arange(n_genes + 2 + n_anchors)}


def build_stats(table, donors, genes, anchors, targets, *, ks=(1,), seed=0, with_outcomes=True):
    """DonorStats for donors with outcomes; plain dz for query donors."""
    stats, dz = [], {}
    for i, donor in enumerate(donors):
        zc = table.inputs(donor, "control", genes, anchors)
        zp = table.inputs(donor, "perturbed", genes, anchors)
        dz[donor] = zp.mean(0) - zc.mean(0)
        if with_outcomes:
            yc = table.outcomes(donor, "control", targets)
            yp = table.outcomes(donor, "perturbed", targets)
            stats.append(donor_stats(donor, zc, yc, zp, yp, ks=ks,
                                     rna_columns=np.arange(len(genes)), seed=seed + 101 * i))
    return stats, dz
