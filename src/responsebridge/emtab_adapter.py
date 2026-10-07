"""E-MTAB-9357 adapter. Panel names and the deposited transform live here.

The CellBridge solver is not modified. Deposited matrices are already
log1p(CPM); CellBridge consumes those values, and count-like CPM is expm1
of them for comparators that expect counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .artifacts import sha256
from .cellbridge_data import cognate_gene
from .subtype_rna_only import MARKERS_V050

FROZEN_SOURCES = (
    "src/responsebridge/cellbridge.py",
    "src/responsebridge/cellbridge_data.py",
)


def assert_gse_frozen_sources(root: Path) -> dict:
    """The v1.0 solver must still match the GSE334503 freeze."""
    root = Path(root)
    frozen = json.loads((root / "runs/cellbridge_v100/gse334503/freeze.json").read_text())
    found = {}
    for rel in FROZEN_SOURCES:
        digest = sha256(root / rel)
        if digest != frozen["sources_sha256"][rel]:
            raise ValueError(f"{rel} no longer matches the GSE334503 freeze")
        found[rel] = digest
    return found


def deposited_lineage_labels(markers: dict[str, np.ndarray]) -> np.ndarray:
    """Label cells from deposited log1p(CPM) marker values. Protein is not an input.

    ``markers`` maps a gene symbol to a length-n vector. A lineage needs two
    markers present in the map. A cell is labeled only when at least two of
    those values are positive and the mean beats the runner-up by 0.1.
    """
    n = len(next(iter(markers.values())))
    scores = []
    detected = []
    names = []
    for lineage, genes in MARKERS_V050.items():
        present = [markers[g] for g in genes if g in markers]
        if len(present) < 2:
            raise ValueError(f"Too few RNA markers to label {lineage}")
        stacked = np.column_stack(present)
        scores.append(stacked.mean(axis=1))
        detected.append((stacked > 0).sum(axis=1))
        names.append(lineage)
    score = np.column_stack(scores)
    detect = np.column_stack(detected)
    order = np.argsort(-score, axis=1)
    best, second = order[:, 0], order[:, 1]
    margin = score[np.arange(n), best] - score[np.arange(n), second]
    labels = np.full(n, "unassigned", dtype=object)
    for i, lineage in enumerate(names):
        take = (best == i) & (detect[:, i] >= 2) & (margin > 0.1)
        labels[take] = lineage
    return labels


def select_genes_from_moments(total, total_sq, n_cells, detected, n_genes, forced):
    """Top cell-variance genes, the same rule as CellTable.select_genes('cell')."""
    n_cells = float(n_cells)
    mean = total / n_cells
    var = total_sq / n_cells - mean ** 2
    keep = np.asarray(detected, dtype=float) / n_cells >= 0.01
    var = np.where(keep, var, -np.inf)
    top = list(np.argsort(-var, kind="stable")[:n_genes])
    chosen = set(top)
    for gene in forced:
        if gene not in chosen and np.isfinite(var[gene]):
            top.append(int(gene))
            chosen.add(int(gene))
    return np.array(sorted(top), dtype=int)


def cognate_for_deposited(name, genes):
    """Map a TotalSeq-style column onto a gene symbol, if one is present."""
    genes = list(genes)
    gene_set = set(genes)
    hit = cognate_gene(name, genes)
    if hit:
        return hit
    for token in str(name).replace("-", "_").split("_"):
        if token in gene_set:
            return token
        hit = cognate_gene(token, genes)
        if hit:
            return hit
    return None


def sparse_from_rows(indices, data, n_cols):
    """CSR matrix from per-row nonzero indices and values."""
    lengths = [len(ix) for ix in indices]
    indptr = np.zeros(len(indices) + 1, dtype=np.int32)
    np.cumsum(lengths, out=indptr[1:])
    if indptr[-1] == 0:
        return sparse.csr_matrix((len(indices), n_cols), dtype=np.float32)
    return sparse.csr_matrix(
        (np.concatenate(data).astype(np.float32), np.concatenate(indices).astype(np.int32), indptr),
        shape=(len(indices), n_cols),
    )


def save_arm(path, rna, protein, detected, library_cpm, *, full_protein):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rna = rna.tocsr().astype(np.float32)
    payload = {
        "rna_data": rna.data, "rna_indices": rna.indices, "rna_indptr": rna.indptr,
        "rna_shape": np.array(rna.shape, dtype=np.int64),
        "detected": np.asarray(detected, dtype=np.float32),
        "library_cpm": np.asarray(library_cpm, dtype=np.float32),
        "full_protein": np.array([1 if full_protein else 0]),
    }
    if full_protein:
        payload["protein"] = np.asarray(protein, dtype=np.float32)
    else:
        payload["protein_anchors"] = np.asarray(protein, dtype=np.float32)
    np.savez_compressed(path, **payload)


def load_arm(path):
    with np.load(path, allow_pickle=False) as blob:
        shape = tuple(int(x) for x in blob["rna_shape"])
        rna = sparse.csr_matrix((blob["rna_data"], blob["rna_indices"], blob["rna_indptr"]), shape=shape)
        protein = blob["protein"] if "protein" in blob.files else None
        anchors = blob["protein_anchors"] if "protein_anchors" in blob.files else None
        return {
            "rna": rna,
            "protein": None if protein is None else protein.copy(),
            "protein_anchors": None if anchors is None else anchors.copy(),
            "detected": blob["detected"].copy(),
            "library_cpm": blob["library_cpm"].copy(),
            "full_protein": bool(int(blob["full_protein"][0])),
        }


def query_targets(arm, columns):
    """Refuse target proteins stored beside query inputs. The vault is separate."""
    if arm["full_protein"]:
        raise PermissionError("Query arms must not carry target proteins")
    if columns is None:
        return arm["protein_anchors"]
    raise PermissionError("Target proteins of a query patient are in the outcome vault")
