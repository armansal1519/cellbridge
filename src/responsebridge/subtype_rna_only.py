"""RNA-only lineage labels for v0.5. Protein values never enter the rule."""
from __future__ import annotations

import numpy as np

from .prepare import rna_lineage_labels

MARKERS_V050 = {
    "T": ["CD3D", "CD3E", "CD3G", "TRAC"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "Monocyte": ["LST1", "LYZ", "FCN1", "CTSS", "S100A8"],
    "NK": ["GNLY", "NKG7", "KLRD1", "PRF1"],
}


def rna_lineage_labels_v050(rna, gene_names, markers=None):
    """Assign each cell the RNA marker lineage with the highest score.

    A lineage is eligible only if at least two of its markers are detected.
    The winning score must exceed the runner-up by 0.1 on the mean log1p
    library-normalized scale. Unassigned cells are dropped by callers.
    """
    markers = markers or MARKERS_V050
    total = np.asarray(rna.sum(axis=1)).ravel()
    lookup = {x: i for i, x in enumerate(gene_names)}
    scores, detected = {}, {}
    for lineage, genes in markers.items():
        present = [lookup[g] for g in genes if g in lookup]
        if len(present) < 2:
            raise ValueError(f"Too few RNA markers to label {lineage}")
        counts = rna[:, present].toarray() if hasattr(rna[:, present], "toarray") else np.asarray(rna[:, present])
        scores[lineage] = np.log1p(counts * (1e4 / np.maximum(total, 1))[:, None]).mean(axis=1)
        detected[lineage] = (counts > 0).sum(axis=1)
    names = list(markers)
    stacked = np.column_stack([scores[n] for n in names])
    order = np.argsort(-stacked, axis=1)
    best = order[:, 0]
    second = order[:, 1]
    labels = np.full(rna.shape[0], "unassigned", dtype=object)
    for i, lineage in enumerate(names):
        win = best == i
        margin = stacked[np.arange(len(best)), best] - stacked[np.arange(len(best)), second]
        use = win & (detected[lineage] >= 2) & (margin > 0.1)
        labels[use] = lineage
    return labels, {"markers": markers, "minimum_detected": 2, "minimum_mean_log_score_difference": 0.1,
                    "protein_used": False, "version": "0.5.0"}


def v040_compatible_labels(rna, gene_names):
    """Unmodified v0.4 T/Monocyte rule, for continuity checks only."""
    return rna_lineage_labels(rna, gene_names)
