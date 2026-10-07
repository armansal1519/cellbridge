import numpy as np
import pytest

from responsebridge.emtab_adapter import (assert_gse_frozen_sources, deposited_lineage_labels,
                                          query_targets, select_genes_from_moments)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_solver_still_matches_the_gse_freeze():
    found = assert_gse_frozen_sources(ROOT)
    assert set(found) == {"src/responsebridge/cellbridge.py", "src/responsebridge/cellbridge_data.py"}


def test_deposited_lineage_uses_rna_markers_only():
    n = 6
    markers = {}
    for gene in ("CD3D", "CD3E", "CD3G", "TRAC"):
        markers[gene] = np.zeros(n)
    for gene in ("MS4A1", "CD79A", "CD79B", "CD19", "BANK1"):
        markers[gene] = np.zeros(n)
    for gene in ("LST1", "LYZ", "FCN1", "CTSS", "S100A8"):
        markers[gene] = np.zeros(n)
    for gene in ("GNLY", "NKG7", "KLRD1", "PRF1"):
        markers[gene] = np.zeros(n)
    markers["CD3D"][:2] = 2
    markers["CD3E"][:2] = 2
    markers["LYZ"][2:4] = 2
    markers["FCN1"][2:4] = 2
    markers["CTSS"][2:4] = 2
    labels = deposited_lineage_labels(markers)
    assert list(labels[:2]) == ["T", "T"]
    assert list(labels[2:4]) == ["Monocyte", "Monocyte"]
    assert list(labels[4:]) == ["unassigned", "unassigned"]


def test_gene_selection_matches_the_cell_variance_rule():
    total = np.array([10.0, 0.0, 5.0])
    total_sq = np.array([30.0, 0.0, 5.0])
    detected = np.array([8, 0, 8])
    chosen = select_genes_from_moments(total, total_sq, 10, detected, 1, forced=[2])
    assert chosen.tolist() == [0, 2]


def test_query_arm_refuses_target_proteins():
    arm = {"full_protein": False, "protein_anchors": np.zeros((2, 3))}
    assert query_targets(arm, None).shape == (2, 3)
    with pytest.raises(PermissionError):
        query_targets(arm, [0])
