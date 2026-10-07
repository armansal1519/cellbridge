"""Auditable raw downloads and an independent, count-preserving data contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import gzip
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse

LAWLOR_VERSION = "2021-02-22T16%3A08%3A23.715288Z"
LAWLOR_FILES = {
    "CZI.PBMC.RNA.matrix.Rds": ("347257fa-4174-5741-9a9d-99ef21fa3707", 480584213, "c6c1b29e8737085d89f2af35c23fb9ce0e189369e65061f18cfe8532bd1e80c5"),
    "CZI.PBMC.ADT.matrix.Rds": ("d3b085d0-4d42-5e0f-bf11-da491031d4b8", 21609958, "bab0c4c2b43bb2c8c942491a370623b0685ef0dd10ce58aa45734371a8fb59e5"),
    "CZI.PBMC.cell.annotations.csv": ("773e8edd-48af-599a-8726-3f368823dc22", 26242593, "76a6548089867687461603ed0031ba02f9203885cd8a623b7497bdceccaf7229"),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _download_ranges(url, tmp, size, *, attempts=3, block_bytes=4 * 1024**2):
    """Bound each signed HTTP response; resume bytes already durably written."""
    tmp = Path(tmp)
    while (tmp.stat().st_size if tmp.exists() else 0) < size:
        for attempt in range(attempts):
            offset = tmp.stat().st_size if tmp.exists() else 0
            end = min(offset + block_bytes, size) - 1
            request = urllib.request.Request(url, headers={"User-Agent": "ResponseBridge research data audit", "Range": f"bytes={offset}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {offset}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Unexpected range response; expected {expected}")
                    remaining = end - offset + 1
                    with tmp.open("ab") as out:
                        while remaining:
                            chunk = response.read(min(256 * 1024, remaining))
                            if not chunk:
                                raise OSError("Truncated HTTP range")
                            out.write(chunk)
                            remaining -= len(chunk)
                break
            except OSError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(1)


def download_lawlor(raw_dir, *, minimum_free_gib=30, attempts=3):
    """GET published processed files, verify HCA SHA256, rename atomically.

    HEAD is deliberately not used: the HCA redirect signs the GET method.
    Existing complete files are reverified; interrupted transfers use Range.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, (uid, size, checksum) in LAWLOR_FILES.items():
        url = f"https://service.azul.data.humancellatlas.org/repository/files/{uid}?catalog=dcp60&version={LAWLOR_VERSION}"
        destination = raw_dir / name
        if not destination.exists():
            tmp = destination.with_name(name + ".partial")
            offset = tmp.stat().st_size if tmp.exists() else 0
            if offset > size:
                raise ValueError(f"Oversized partial file: {name}")
            if shutil.disk_usage(raw_dir).free - (size - offset) < minimum_free_gib * 2**30:
                raise RuntimeError("Download would violate the free-space reserve")
            _download_ranges(url, tmp, size, attempts=attempts)
            if tmp.stat().st_size != size or sha256(tmp) != checksum:
                raise ValueError(f"HCA checksum/size mismatch: {name}")
            os.replace(tmp, destination)
        if destination.stat().st_size != size or sha256(destination) != checksum:
            raise ValueError(f"Existing raw file fails integrity check: {name}")
        entries.append({"file": name, "bytes": size, "sha256": checksum, "url": url})
    registry = {"dataset": "Lawlor_PBMC_CITEseq", "source": "https://explore.data.humancellatlas.org/projects/efea6426-510a-4b60-9a19-277e52bfa815", "files": entries, "raw_only": True}
    write_json(raw_dir / "download_manifest.json", registry)
    return registry


@dataclass
class PreparedDataset:
    rna: sparse.csr_matrix
    proteins: np.ndarray
    obs: pd.DataFrame
    gene_names: list[str]
    protein_names: list[str]
    isotypes: np.ndarray
    isotype_names: list[str]
    manifest: dict

    def validate(self):
        n = len(self.obs)
        if self.rna.shape != (n, len(self.gene_names)) or self.proteins.shape != (n, len(self.protein_names)):
            raise ValueError("Count matrix and named dimensions do not agree")
        if self.isotypes.shape != (n, len(self.isotype_names)):
            raise ValueError("Isotype dimensions disagree")
        required = {"cell_id", "donor", "lineage", "context", "perturbation", "is_control"}
        if required - set(self.obs):
            raise ValueError(f"Missing metadata fields: {required - set(self.obs)}")
        if not self.obs.cell_id.is_unique:
            raise ValueError("Duplicate cell identifiers")
        for names in [self.gene_names, self.protein_names, self.isotype_names]:
            if len(set(names)) != len(names):
                raise ValueError("Duplicate canonical feature names")
        for counts in [self.rna.data, self.proteins, self.isotypes]:
            flat = np.asarray(counts).reshape(-1)
            for start in range(0, len(flat), 1_000_000):
                block = flat[start:start + 1_000_000]
                if np.any(block < 0) or not np.isfinite(block).all() or np.any(block != np.floor(block)):
                    raise ValueError("Only finite nonnegative raw integer counts are allowed")
        return self


def save_prepared(dataset, folder):
    dataset.validate()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(folder / "rna_counts.npz", dataset.rna, compressed=True)
    np.save(folder / "protein_counts.npy", dataset.proteins, allow_pickle=False)
    np.save(folder / "isotype_counts.npy", dataset.isotypes, allow_pickle=False)
    dataset.obs.to_csv(folder / "obs.csv", index=False)
    for name, values in [("genes", dataset.gene_names), ("proteins", dataset.protein_names), ("isotypes", dataset.isotype_names)]:
        write_json(folder / f"{name}.json", values)
    manifest = dict(dataset.manifest)
    manifest.update({"schema_version": 1, "n_cells": len(dataset.obs), "n_genes": len(dataset.gene_names),
        "n_proteins": len(dataset.protein_names), "n_isotypes": len(dataset.isotype_names),
        "orientation": "cells_by_features", "counts": "raw_integer", "rna_nnz": int(dataset.rna.nnz)})
    names = ["rna_counts.npz", "protein_counts.npy", "isotype_counts.npy", "obs.csv", "genes.json", "proteins.json", "isotypes.json"]
    manifest["files"] = {name: {"sha256": sha256(folder / name), "bytes": (folder / name).stat().st_size} for name in names}
    write_json(folder / "manifest.json", manifest)
    return manifest


def load_prepared(folder, *, verify=True):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if verify:
        for name, entry in manifest["files"].items():
            if sha256(folder / name) != entry["sha256"]:
                raise ValueError(f"Prepared file checksum mismatch: {name}")
    obj = PreparedDataset(sparse.load_npz(folder / "rna_counts.npz").tocsr(),
        np.load(folder / "protein_counts.npy", mmap_mode="r"),
        pd.read_csv(folder / "obs.csv", keep_default_na=False, dtype={"donor": str, "cell_id": str}),
        json.loads((folder / "genes.json").read_text()), json.loads((folder / "proteins.json").read_text()),
        np.load(folder / "isotype_counts.npy", mmap_mode="r"), json.loads((folder / "isotypes.json").read_text()), manifest)
    return obj.validate()


def audit_gse278572(raw_dir):
    """Metadata-only audit: no RNA/ADT values or prior outcomes are inspected."""
    raw_dir = Path(raw_dir)
    features = pd.read_csv(raw_dir / "GSE278572_features.tsv.gz", sep="\t", header=None, names=["id", "symbol", "type"], dtype=str)
    bio = features.type.eq("Antibody Capture") & ~features.id.str.startswith("HTO_") & ~features.symbol.str.startswith("Isotype_")
    guide = pd.read_csv(raw_dir / "GSE278572_protospacer_calls_per_cell.csv.gz", usecols=["cell_barcode", "num_features", "feature_call"])
    donor = pd.read_csv(raw_dir / "publisher_metadata/all_cells_donor_demux_out.csv", usecols=["status", "Donor"], dtype=str)
    with gzip.open(raw_dir / "GSE278572_matrix.mtx.gz", "rt") as stream:
        header = stream.readline().strip()
        line = stream.readline()
        while line.startswith("%"):
            line = stream.readline()
        shape = list(map(int, line.split()))
    return {"dataset": "GSE278572", "audit_scope": "feature definitions, dimensions and assignment metadata only",
        "matrix_header": header, "raw_features_cells_entries": shape,
        "feature_types": {str(k): int(v) for k, v in features.type.value_counts().items()},
        "biological_adt_features": int(bio.sum()), "antibody_names": features.loc[bio, "symbol"].tolist(),
        "hashing_features": features.loc[features.id.str.startswith("HTO_"), "symbol"].tolist(),
        "isotype_features": features.loc[features.symbol.str.startswith("Isotype_"), "symbol"].tolist(),
        "donor_singlets": {str(k): int(v) for k, v in donor.loc[donor.status.eq("singlet"), "Donor"].value_counts().items()},
        "guide_metadata_rows": len(guide), "guide_cell_ids_unique": bool(guide.cell_barcode.is_unique),
        "matrix_bytes": (raw_dir / "GSE278572_matrix.mtx.gz").stat().st_size,
        "S14_eligibility": "Forbidden: outcome-selected membership. Use only as technical sample-label agreement audit.",
        "prior_exposure": "RNA and GRN outcomes previously used in BioReFi; not an untouched study."}


def audit_lawlor(raw_dir):
    """Verify deposited files and metadata without reading protein outcomes."""
    raw_dir = Path(raw_dir)
    files = {}
    for name, (_, expected_size, expected_hash) in LAWLOR_FILES.items():
        path = raw_dir / name
        valid = path.is_file() and path.stat().st_size == expected_size and sha256(path) == expected_hash
        files[name] = {"exists": path.is_file(), "verified": valid, "expected_bytes": expected_size, "expected_sha256": expected_hash}
    result = {"dataset": "Lawlor_PBMC_CITEseq", "audit_scope": "checksums and cell metadata only; ADT outcomes unread", "files": files}
    path = raw_dir / "CZI.PBMC.cell.annotations.csv"
    if files[path.name]["verified"]:
        meta = pd.read_csv(path, index_col=0)
        result["metadata_columns"] = meta.columns.tolist()
        eligible = meta.Demuxlet_Classification.eq("SNG") & meta.HTO_Classification.isin(["Baseline", "LPS", "CD3_CD28"])
        result["genotype_and_condition_eligible"] = int(eligible.sum())
        result["donors"] = sorted(meta.loc[eligible, "Donor_of_Origin"].dropna().astype(str).unique().tolist())
        result["lineage_policy"] = "Fixed RNA-only broad T/monocyte markers; original protein-informed annotation is secondary only."
    return result
