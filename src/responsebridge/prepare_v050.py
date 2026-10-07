"""v0.5 Lawlor prepare that can retain additional HTO arms.

Does not modify prepare_lawlor. Writes a new output directory. IgM/IgG B-cell
development uses conditions Baseline + IgM_IgG and RNA-only B/T/Monocyte/NK labels.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import ResourceGuard, complete_stage, sha256, write_json
from .datasets import PreparedDataset, audit_lawlor, save_prepared
from .prepare import (_collapse_genes, _config, canonical_protein, lawlor_gene_symbols,
                      read_rds_counts)
from .subtype_rna_only import rna_lineage_labels_v050


def prepare_lawlor_conditions(raw_dir, output_dir, conditions, config=None):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    cfg = _config(config)
    conditions = list(conditions)
    if "Baseline" not in conditions:
        raise ValueError("Baseline control must be retained")
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Prepared output exists; use a new version")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_lawlor(raw_dir)
    if not all(item["verified"] for item in audit["files"].values()):
        raise ValueError("All Lawlor inputs must pass the published checksum gate")
    guard = ResourceGuard(output_dir, max_rss_gib=cfg["max_rss_gib"],
                          minimum_free_gib=cfg["minimum_free_gib"], max_hours=cfg["max_hours"])
    guard.check()
    metadata = pd.read_csv(raw_dir / "CZI.PBMC.cell.annotations.csv", index_col=0)
    if not metadata.index.is_unique:
        raise ValueError("Duplicate deposited metadata cell IDs")
    raw_rna, genes, rna_cells = read_rds_counts(raw_dir / "CZI.PBMC.RNA.matrix.Rds")
    if not sparse.issparse(raw_rna):
        raise ValueError("Expected sparse RNA count matrix")
    annotation_manifest = cfg.get("lawlor_annotation_manifest", raw_dir / "annotation/manifest.json")
    genes, gene_mapping, gene_annotation_audit = lawlor_gene_symbols(genes, annotation_manifest)
    gene_mapping.to_csv(output_dir / "gene_id_mapping.csv", index=False)
    gene_annotation_audit["gene_id_mapping_sha256"] = sha256(output_dir / "gene_id_mapping.csv")
    meta = metadata.reindex(rna_cells)
    if meta.Demuxlet_Classification.isna().any():
        raise ValueError("RNA cells lack matching metadata")
    selected = (meta.Demuxlet_Classification.eq("SNG") & meta.HTO_Classification.isin(conditions)
                & meta.Donor_of_Origin.notna())
    rna = raw_rna.T.tocsr()[selected.to_numpy()]
    cells = np.asarray(rna_cells)[selected.to_numpy()]
    meta = meta.loc[selected].copy()
    del raw_rna
    rna, genes = _collapse_genes(rna, genes)
    total = np.asarray(rna.sum(axis=1)).ravel()
    detected = np.diff(rna.indptr)
    mito_columns = np.array([str(g).upper().startswith("MT-") for g in genes])
    mitochondrial = np.asarray(rna[:, mito_columns].sum(axis=1)).ravel() / np.maximum(total, 1)
    lineage, lineage_rule = rna_lineage_labels_v050(rna, genes)
    available = set(genes)
    lineage_rule["available_markers"] = {k: [g for g in v if g in available]
                                         for k, v in lineage_rule["markers"].items()}
    lineage_rule["unavailable_markers"] = {k: [g for g in v if g not in available]
                                           for k, v in lineage_rule["markers"].items()}
    keep = ((detected >= cfg["rna_min_genes"]) & (total >= cfg["rna_min_counts"])
            & (mitochondrial <= cfg["rna_max_mito_fraction"]) & (lineage != "unassigned"))
    rna, meta, cells, lineage = rna[keep], meta.iloc[np.flatnonzero(keep)], cells[keep], lineage[keep]
    adt, antibody_names, adt_cells = read_rds_counts(raw_dir / "CZI.PBMC.ADT.matrix.Rds")
    locations = pd.Index(adt_cells).get_indexer(cells)
    if np.any(locations < 0):
        raise ValueError("RNA-selected cells lack measured ADT")
    antibody_names = [re.sub(r"-[ACGT]{15}$", "", n) for n in antibody_names]
    is_iso = np.array([n.startswith("control_") for n in antibody_names])
    is_qc = np.array([n in {"bad_struct", "no_match", "total_reads"} for n in antibody_names])
    biological = ~(is_iso | is_qc)
    proteins = np.asarray(adt)[biological][:, locations].T
    isotypes = np.asarray(adt)[is_iso][:, locations].T
    mapping = {c: c for c in conditions}
    mapping["Baseline"] = "non-targeting"
    obs = pd.DataFrame({"cell_id": cells, "donor": meta.Donor_of_Origin.astype(str).to_numpy(),
        "lineage": lineage, "context": lineage, "condition": meta.HTO_Classification.to_numpy(),
        "perturbation": meta.HTO_Classification.map(mapping).to_numpy(),
        "is_control": meta.HTO_Classification.eq("Baseline").to_numpy(), "guide": "not_applicable",
        "library": meta.Run_Identifier.astype(str).to_numpy(), "donor_status": "SNG",
        "rna_counts": total[keep], "rna_genes": detected[keep], "rna_mito_fraction": mitochondrial[keep]})
    amendment = {"dataset": "Lawlor_PBMC_CITEseq", "conditions": conditions, "version": "0.5.0",
                 "config": {k: v for k, v in cfg.items() if k != "lawlor_annotation_manifest"},
                 "rna_lineage_rule": lineage_rule, "gene_annotation": gene_annotation_audit,
                 "eligibility": "Genotype singlet, listed HTO conditions, RNA QC and RNA marker labels only",
                 "source_sha256": sha256(__file__), "v040_prepare_unmodified": True,
                 "original_ADT_informed_annotation_used": False}
    write_json(output_dir / "preparation_amendment.json", amendment)
    manifest = {"dataset": "Lawlor_PBMC_CITEseq", "raw_files": audit["files"],
                "gene_annotation": gene_annotation_audit, "conditions": conditions, "version": "0.5.0",
                "preparation_amendment_sha256": sha256(output_dir / "preparation_amendment.json"),
                "transfer_gate": "unverified_antibody_clone_and_scale_compatibility",
                "outcome_effects_evaluated": False,
                "source": "https://doi.org/10.3389/fimmu.2021.636720"}
    obs.groupby(["donor", "lineage", "condition"]).size().rename("n_cells").reset_index().to_csv(
        output_dir / "group_coverage.csv", index=False)
    protein_names = [canonical_protein(n) for n, keep_ab in zip(antibody_names, biological) if keep_ab]
    result = save_prepared(PreparedDataset(rna, proteins, obs, genes, protein_names, isotypes,
                                           [n for n, flag in zip(antibody_names, is_iso) if flag], manifest),
                           output_dir)
    complete_stage(output_dir, {"n_cells": int(len(obs)), "conditions": conditions})
    guard.check()
    return result
