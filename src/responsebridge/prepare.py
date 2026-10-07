"""Streaming importers. Biological ADTs never determine cell eligibility.

GSE278572 experimental lineages/conditions are reconstructed from sample tags,
not CD25/CD69 or the outcome-selected membership of the publisher S14 table.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse

from .datasets import PreparedDataset, save_prepared, sha256, write_json
from .artifacts import ResourceGuard

DEFAULT_CONFIG = {"rna_min_genes": 200, "rna_min_counts": 500, "rna_max_mito_fraction": .2,
    "hto_min_top": 100, "hto_min_ratio": 3, "hto_agreement_min": .95,
    "chunk_lines": 300000, "max_rss_gib": 32, "minimum_free_gib": 30,
    "max_hours": 168, "validate_s14": True}


def canonical_protein(name):
    """Names only; antigen aliases do not establish antibody clone equivalence."""
    name = re.sub(r"-[ACGT]{15}$", "", str(name))
    name = re.sub(r"^(HuMsRt|HuMs|Hu)\.", "", name)
    aliases = {"CD127_IL7Ra": "CD127", "CD183_CXCR3": "CD183", "CD185_CXCR5": "CD185",
        "CD194_CCR4": "CD194", "CD195_CCR5": "CD195", "CD196_CCR6": "CD196", "CD197_CCR7": "CD197",
        "CD278_ICOS": "ICOS", "CD278": "ICOS", "CD279_PD1": "PD-1", "CD279": "PD-1",
        "HLA.DR": "HLA-DR", "CD79b_IgB": "CD79b"}
    # Clone suffixes are retained separately in the source feature table.
    name = re.sub(r"_(HIT2|UCHT1|RPA\.T4|M5E2|2H7|43A3|11A8|HI30)$", "", name)
    return aliases.get(name, name)


def decode_guide(call, count):
    n = float(count)
    if not np.isfinite(n) or n < 0 or n != int(n):
        raise ValueError("Invalid guide count")
    names = [] if pd.isna(call) or call == "" else str(call).split("|")
    if len(names) != int(n) or len(set(names)) != len(names):
        raise ValueError("Guide names and counts disagree")
    if not names:
        return None
    targets = set()
    for name in names:
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9.-]*)_([1-9]\d*)_CRISPRi", name)
        if match is None:
            raise ValueError(f"Unrecognized guide: {name}")
        targets.add(match[1])
    if len(targets) != 1:
        return None
    target = targets.pop()
    control = target == "Non-Targeting"
    return ("non-targeting" if control else target, control, "|".join(sorted(names)), len(names))


def read_gse_assignments(raw_dir, barcodes):
    raw_dir = Path(raw_dir)
    donor = pd.read_csv(raw_dir / "publisher_metadata/all_cells_donor_demux_out.csv", dtype=str,
                        usecols=["barcode", "well", "status", "Donor"])
    donor = donor.loc[donor.status.eq("singlet") & donor.Donor.isin(["A", "B"])].copy()
    donor["cell_id"] = donor.barcode.str.replace(r"-\d+$", "", regex=True) + "-" + donor.well
    donor = donor[["cell_id", "Donor", "well"]].drop_duplicates().rename(columns={"Donor": "donor", "well": "library"})
    if donor.cell_id.duplicated().any():
        raise ValueError("Conflicting donor calls")
    guide = pd.read_csv(raw_dir / "GSE278572_protospacer_calls_per_cell.csv.gz")
    if guide.cell_barcode.duplicated().any():
        raise ValueError("Duplicate guide assignment barcodes")
    decoded = [decode_guide(c, n) for c, n in zip(guide.feature_call, guide.num_features)]
    retained = np.array([x is not None for x in decoded])
    assignment = pd.DataFrame([x for x in decoded if x is not None], columns=["perturbation", "is_control", "guide", "n_guides"])
    assignment["cell_id"] = guide.loc[retained, "cell_barcode"].to_numpy()
    obs = donor.merge(assignment, on="cell_id", validate="one_to_one")
    # Align with the raw column order, never with outcome-selected S14 rows.
    obs = obs.set_index("cell_id").reindex(barcodes).dropna(subset=["donor", "perturbation"]).reset_index()
    obs.rename(columns={obs.columns[0]: "cell_id"}, inplace=True)
    obs["is_control"] = obs.is_control.astype(bool)
    obs["donor_status"] = "singlet"
    return obs


def matrix_chunks(path, chunk_lines=300000):
    """Yield header then 0-based integer coordinate blocks, validating counts."""
    with gzip.open(path, "rt", encoding="ascii") as stream:
        if stream.readline().strip().lower() != "%%matrixmarket matrix coordinate integer general":
            raise ValueError("Only integer general MatrixMarket is accepted")
        line = stream.readline()
        while line.startswith("%"):
            line = stream.readline()
        nrow, ncol, nnz = map(int, line.split())
        yield nrow, ncol, nnz
        observed = 0
        while True:
            lines = []
            for _ in range(chunk_lines):
                line = stream.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            triplets = np.fromstring("".join(lines), sep=" ", dtype=np.int64)
            if triplets.size != 3 * len(lines):
                raise ValueError("Malformed count coordinate")
            triplets = triplets.reshape(-1, 3)
            row, col, value = triplets.T
            if np.any(row < 1) or np.any(row > nrow) or np.any(col < 1) or np.any(col > ncol) or np.any(value < 0):
                raise ValueError("Invalid count/index in raw matrix")
            observed += len(value)
            yield row - 1, col - 1, value
        if observed != nnz:
            raise ValueError(f"Truncated matrix: observed {observed}, expected {nnz}")


def classify_hashtags(counts, names, config):
    expected = {"Resting_Treg", "Stimulated_Treg", "Resting_Teff", "Stimulated_Teff"}
    if set(names) != expected:
        raise ValueError("Unexpected experimental sample tags")
    order = np.argsort(counts, axis=1, kind="stable")
    idx = np.arange(len(counts))
    first, second = counts[idx, order[:, -1]], counts[idx, order[:, -2]]
    valid = (first >= config["hto_min_top"]) & (first > second) & (first / (second + 1) >= config["hto_min_ratio"])
    labels = np.array([f"{n.split('_')[1]}_{'stim' if n.startswith('Stimulated') else 'rest'}" for n in names])
    result = np.full(len(counts), "unassigned", dtype=object)
    result[valid] = labels[order[valid, -1]]
    return result, valid


def read_s14_labels(path):
    """Stream only barcode and sample label from XLSX; never outcome columns."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    strings, header, labels = [], {}, {}
    with zipfile.ZipFile(path) as archive:
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as stream:
                for _, elem in ET.iterparse(stream, events=("end",)):
                    if elem.tag == ns + "si":
                        strings.append("".join(elem.itertext()))
                        elem.clear()
        with archive.open("xl/worksheets/sheet1.xml") as stream:
            for _, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != ns + "row":
                    continue
                row = {}
                for cell in elem.findall(ns + "c"):
                    key = re.sub(r"\d", "", cell.attrib["r"])
                    if elem.attrib["r"] != "1" and key not in header:
                        continue
                    v = cell.find(ns + "v")
                    value = "" if v is None else v.text
                    if cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(cell.find(ns + "is").itertext())
                    if elem.attrib["r"] == "1":
                        if value in {"cell", "HTO_maxID"}:
                            header[key] = value
                    else:
                        row[header[key]] = value
                if "cell" in row and "HTO_maxID" in row:
                    labels[row["cell"]] = row["HTO_maxID"]
                elem.clear()
    if not labels:
        raise ValueError("No published sample labels found for technical audit")
    return labels


def _config(config):
    result = dict(DEFAULT_CONFIG)
    result.update(config or {})
    return result


def _collapse_genes(rna, symbols):
    symbols = np.asarray(symbols, dtype=str)
    names, inverse = np.unique(symbols, return_inverse=True)
    if len(names) == len(symbols):
        return rna, symbols.tolist()
    mapping = sparse.csr_matrix((np.ones(len(symbols), dtype=np.uint32), (np.arange(len(symbols)), inverse)), shape=(len(symbols), len(names)))
    return (rna @ mapping).tocsr(), names.tolist()


def prepare_gse278572(raw_dir, output_dir, config=None):
    """Stream the billion-entry file once using <=32GiB RSS by default.

    CellRanger stores coordinates by cell. The importer checks that ordering
    and refuses unsorted input rather than silently constructing corrupt CSR.
    A scratch pair of binary arrays holds only eligible RNA coordinates.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    cfg = _config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Prepared output already exists; reuse or choose a new version")
    amendment = {"config": cfg, "dataset": "GSE278572", "lineage_source": "experimental sample HTO, not biological ADT",
        "S14": "label agreement only; never cell selection", "prior_exposure": "RNA/GRN outcomes previously examined in BioReFi",
        "source_sha256": sha256(__file__)}
    amendment_path = output_dir / "preparation_amendment.json"
    if amendment_path.exists() and json.loads(amendment_path.read_text()) != amendment:
        raise FileExistsError("Preparation inputs/code changed; choose a new output version")
    write_json(amendment_path, amendment)
    guard = ResourceGuard(output_dir, max_rss_gib=cfg["max_rss_gib"], minimum_free_gib=cfg["minimum_free_gib"], max_hours=cfg["max_hours"])
    guard.check()
    features = pd.read_csv(raw_dir / "GSE278572_features.tsv.gz", sep="\t", header=None, names=["id", "symbol", "type"], dtype=str)
    barcodes = pd.read_csv(raw_dir / "GSE278572_barcodes.tsv.gz", sep="\t", header=None, dtype=str)[0].to_numpy()
    if len(set(barcodes)) != len(barcodes):
        raise ValueError("Duplicate raw cell barcodes")
    obs = read_gse_assignments(raw_dir, barcodes)
    n = len(obs)
    cell_map = pd.Index(obs.cell_id).get_indexer(barcodes).astype(np.int32)
    rna_rows = np.flatnonzero(features.type.eq("Gene Expression"))
    hto_rows = np.flatnonzero(features.id.str.startswith("HTO_"))
    iso_rows = np.flatnonzero(features.type.eq("Antibody Capture") & features.symbol.str.startswith("Isotype_"))
    protein_rows = np.flatnonzero(features.type.eq("Antibody Capture") & ~features.symbol.str.startswith("Isotype_") & ~features.id.str.startswith("HTO_"))
    maps = []
    for rows in [rna_rows, protein_rows, iso_rows, hto_rows]:
        mapping = np.full(len(features), -1, dtype=np.int32)
        mapping[rows] = np.arange(len(rows))
        maps.append(mapping)
    rna_map, protein_map, iso_map, hto_map = maps
    mito = features.symbol.str.upper().str.startswith("MT-").to_numpy()
    proteins = np.zeros((n, len(protein_rows)), dtype=np.uint32)
    isotypes = np.zeros((n, len(iso_rows)), dtype=np.uint32)
    hto = np.zeros((n, len(hto_rows)), dtype=np.uint32)
    totals = np.zeros(n, dtype=np.int64)
    detected = np.zeros(n, dtype=np.int32)
    mt = np.zeros(n, dtype=np.int64)
    nonzeros = np.zeros(n, dtype=np.int64)
    scratch = output_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    stream = matrix_chunks(raw_dir / "GSE278572_matrix.mtx.gz", cfg["chunk_lines"])
    dimensions = next(stream)
    if dimensions[:2] != (len(features), len(barcodes)):
        raise ValueError("Raw dimensions do not align with feature/barcode annotation")
    observed, last_cell = 0, -1
    started = time.monotonic()
    print(json.dumps({"stage": "stream_counts", "eligible_metadata_cells": n, "raw_dimensions": dimensions}), flush=True)
    with (scratch / "indices.i4").open("wb") as indices_out, (scratch / "counts.u4").open("wb") as counts_out:
        for rows, columns, values in stream:
            if columns[0] < last_cell or np.any(columns[1:] < columns[:-1]):
                raise ValueError("Input is not column-sorted; do not construct CSR")
            last_cell = int(columns[-1])
            if np.any(values > np.iinfo(np.uint32).max):
                raise OverflowError("Raw count exceeds uint32")
            cells = cell_map[columns]
            eligible = (cells >= 0) & (values > 0)
            rr, cc, vv = rows[eligible], cells[eligible], values[eligible]
            mask = rna_map[rr] >= 0
            r, c, v = rna_map[rr[mask]], cc[mask], vv[mask]
            r.astype(np.int32).tofile(indices_out)
            v.astype(np.uint32).tofile(counts_out)
            np.add.at(totals, c, v)
            np.add.at(detected, c, 1)
            np.add.at(nonzeros, c, 1)
            selected_mt = mito[rr[mask]]
            np.add.at(mt, c[selected_mt], v[selected_mt])
            for mapping, array in [(protein_map, proteins), (iso_map, isotypes), (hto_map, hto)]:
                use = mapping[rr] >= 0
                np.add.at(array, (cc[use], mapping[rr[use]]), vv[use].astype(np.uint32))
            observed += len(rows)
            if observed % (cfg["chunk_lines"] * 20) == 0:
                guard.check()
                write_json(output_dir / "progress.json", {"observed_entries": observed, "total_entries": dimensions[2], "elapsed_seconds": time.monotonic() - started})
                print(json.dumps({"stage": "stream_counts", "fraction": round(observed / dimensions[2], 4), "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
    context, assigned = classify_hashtags(hto, features.iloc[hto_rows].symbol.tolist(), cfg)
    obs["context"] = context
    obs["lineage"] = obs.context.str.split("_").str[0]
    obs["condition"] = obs.context.str.split("_").str[-1]
    obs["rna_counts"], obs["rna_genes"] = totals, detected
    obs["rna_mito_fraction"] = mt / np.maximum(totals, 1)
    good = assigned & (detected >= cfg["rna_min_genes"]) & (totals >= cfg["rna_min_counts"]) & (obs.rna_mito_fraction.to_numpy() <= cfg["rna_max_mito_fraction"])
    label_audit = {"enabled": cfg["validate_s14"]}
    if cfg["validate_s14"]:
        published = read_s14_labels(raw_dir / "S14_metadata_Treg_Teff_perturbseq.xlsx")
        comparison = []
        for barcode, label, usable in zip(obs.cell_id, context, assigned):
            if usable and barcode in published:
                text = str(published[barcode]).lower()
                expected = ("Treg" if "treg" in text else "Teff" if "teff" in text else "unknown") + ("_stim" if "stim" in text and "unstim" not in text else "_rest")
                comparison.append(label == expected)
        agreement = float(np.mean(comparison)) if comparison else 0.
        label_audit.update({"n_overlap": len(comparison), "agreement": agreement})
        write_json(output_dir / "sample_label_audit.json", label_audit)
        if agreement < cfg["hto_agreement_min"]:
            raise ValueError("HTO labels fail the prospective agreement gate")
    indptr = np.r_[0, np.cumsum(nonzeros)].astype(np.int64)
    indices = np.memmap(scratch / "indices.i4", dtype=np.int32, mode="r")
    counts = np.memmap(scratch / "counts.u4", dtype=np.uint32, mode="r")
    rna = sparse.csr_matrix((counts, indices, indptr), shape=(n, len(rna_rows)), copy=False)
    rna = rna[good].tocsr()
    rna.sum_duplicates()
    rna, gene_names = _collapse_genes(rna, features.iloc[rna_rows].symbol)
    protein_names = [canonical_protein(x) for x in features.iloc[protein_rows].symbol]
    features.to_csv(output_dir / "source_features.tsv", sep="\t", index=False)
    cohort = obs.loc[good].reset_index(drop=True)
    cohort.groupby(["donor", "lineage", "context", "perturbation", "is_control"]).size().rename("n_cells").reset_index().to_csv(output_dir / "group_coverage.csv", index=False)
    manifest = {"dataset": "GSE278572", "source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278572",
        "publication": "https://doi.org/10.1038/s41586-024-08314-y", "source_raw_dir": str(raw_dir.resolve()),
        "preparation_amendment_sha256": sha256(output_dir / "preparation_amendment.json"),
        "input_dimensions": dimensions, "metadata_eligible": n, "rna_qc_and_sample_assignment_retained": int(good.sum()),
        "hto_audit": label_audit, "antigen_aliases_do_not_certify_clone_matching": True,
        "prior_exposure": "GSE278572 RNA/GRN examined in BioReFi; this is not a previously untouched cohort"}
    guard.check()
    result = save_prepared(PreparedDataset(rna, proteins[good], cohort, gene_names, protein_names,
        isotypes[good], features.iloc[iso_rows].symbol.tolist(), manifest), output_dir)
    del rna, counts, indices
    for name in ["counts.u4", "indices.i4"]:
        (scratch / name).unlink()
    scratch.rmdir()
    guard.check()
    print(json.dumps({"stage": "complete", "n_cells": result["n_cells"], "n_proteins": result["n_proteins"]}), flush=True)
    return result


def read_rds_counts(path):
    """Read named sparse Matrix or base data.frame through rds2py's parser.

    Parsing the base data.frame directly avoids requiring R/BiocFrame and
    preserves its original cell columns. No pickle or executable R is used.
    """
    import rds2py
    raw = rds2py.parse_rds(str(path))
    attrs = raw.get("attributes", {})
    # S4 class metadata is a top-level key in rds2py; base data.frame class
    # information can instead live in attributes (class_name may be vector).
    classes = list(attrs.get("class", {}).get("data", []))
    if raw.get("class_name"):
        classes.append(raw["class_name"])
    if "dgCMatrix" in classes:
        shape = tuple(int(v) for v in attrs["Dim"]["data"])
        matrix = sparse.csc_matrix((attrs["x"]["data"], attrs["i"]["data"], attrs["p"]["data"]), shape=shape)
        dimnames = attrs["Dimnames"]["data"]
        names = [list(dimnames[0]["data"]), list(dimnames[1]["data"])]
    elif "data.frame" in classes:
        names = [list(attrs["row.names"]["data"]), list(attrs["names"]["data"])]
        matrix = np.column_stack([column["data"] for column in raw["data"]])
    else:
        raise ValueError(f"Unsupported deposited RDS class: {classes}")
    names = [[str(x) for x in axis] for axis in names]
    if matrix.shape != (len(names[0]), len(names[1])) or len(set(names[1])) != len(names[1]):
        raise ValueError("RDS names and matrix dimensions do not align")
    return matrix, names[0], names[1]


def rna_lineage_labels(rna, gene_names):
    """Fixed broad lineage rule. No ADT values or fitted reference labels.

    Require two detected markers and a positive difference between the mean
    log-normalized T and myeloid marker expressions. This is an explicit,
    conservative rule, not a claimed universal cell type classifier.
    """
    markers = {"T": ["CD3D", "CD3E", "CD3G", "TRAC"],
               "Monocyte": ["LST1", "LYZ", "FCN1", "CTSS", "S100A8"]}
    total = np.asarray(rna.sum(axis=1)).ravel()
    lookup = {x: i for i, x in enumerate(gene_names)}
    scores, detected = {}, {}
    for lineage, genes in markers.items():
        present = [lookup[g] for g in genes if g in lookup]
        if len(present) < 2:
            raise ValueError(f"Too few RNA markers to label {lineage}")
        counts = rna[:, present].toarray()
        scores[lineage] = np.log1p(counts * (1e4 / np.maximum(total, 1))[:, None]).mean(axis=1)
        detected[lineage] = (counts > 0).sum(axis=1)
    labels = np.full(rna.shape[0], "unassigned", dtype=object)
    for lineage, other in [("T", "Monocyte"), ("Monocyte", "T")]:
        use = (detected[lineage] >= 2) & (scores[lineage] > scores[other] + .1)
        labels[use] = lineage
    return labels, {"markers": markers, "minimum_detected": 2, "minimum_mean_log_score_difference": .1}


def lawlor_gene_symbols(gene_ids, annotation_manifest):
    """Resolve deposited Ensembl IDs using an explicitly pinned GTF artifact.

    ``raw/annotation/manifest.json`` must contain ``format: gencode_gtf``,
    ``artifact`` (a relative GTF or GTF.gz path), ``sha256`` and ``source_url``.
    For the audited import, use GENCODE v46 basic GRCh38 annotation from
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/
    gencode.v46.basic.annotation.gtf.gz, SHA256
    d620d548ad23dad6c2d67486b7679d12f00f02f66dd5787183d8e6678dceb9b4.
    Only exact stable IDs are joined; no retired-ID or gene-symbol aliases
    are inferred. Unmapped IDs and their counts remain in the RNA matrix.
    """
    annotation_manifest = Path(annotation_manifest)
    if not annotation_manifest.is_file():
        raise FileNotFoundError(
            f"Missing pinned Lawlor gene annotation manifest: {annotation_manifest}. "
            "Place the GENCODE GTF and a manifest with format=gencode_gtf, "
            "artifact, sha256 and source_url under raw/annotation; "
            "see lawlor_gene_symbols for the audited source URL and checksum.")
    receipt = json.loads(annotation_manifest.read_text())
    required = {"format", "artifact", "sha256", "source_url"}
    if not required.issubset(receipt) or receipt["format"] != "gencode_gtf":
        raise ValueError("Invalid Lawlor gene annotation manifest")
    relative = Path(receipt["artifact"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Gene annotation artifact must be relative to its manifest")
    artifact = annotation_manifest.parent / relative
    if not artifact.is_file() or sha256(artifact) != receipt["sha256"]:
        raise ValueError("Lawlor gene annotation artifact checksum mismatch or missing file")
    if any(re.fullmatch(r"ENSG\d{11}", str(g)) is None for g in gene_ids):
        raise ValueError("Expected deposited, unversioned Lawlor Ensembl gene IDs")
    annotation, mitochondrial_ids = {}, set()
    opener = gzip.open if artifact.suffix == ".gz" else open
    with opener(artifact, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError("Malformed gene annotation GTF record")
            if fields[2] != "gene":
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]+)"', fields[8]))
            if not attrs.get("gene_id") or not attrs.get("gene_name"):
                raise ValueError("GTF gene record lacks gene_id or gene_name")
            # Keep a possible PAR_Y suffix distinct; remove only the version.
            stable_id = re.sub(r"\.\d+(?=(_PAR_Y)?$)", "", attrs["gene_id"])
            value = (attrs["gene_name"], fields[0])
            if stable_id in annotation and annotation[stable_id] != value:
                raise ValueError(f"Ambiguous GTF gene annotation: {stable_id}")
            annotation[stable_id] = value
            if fields[0] in {"chrM", "MT"}:
                if not attrs["gene_name"].startswith("MT-"):
                    raise ValueError("Mitochondrial GTF gene lacks an MT- symbol")
                mitochondrial_ids.add(stable_id)
    if not mitochondrial_ids:
        raise ValueError("GTF annotation lacks mitochondrial genes; RNA QC cannot proceed")
    observed_mito = set(gene_ids) & mitochondrial_ids
    if not observed_mito:
        raise ValueError("No deposited mitochondrial IDs match the pinned annotation")
    rows = [{"source_gene_id": g, "output_gene_name": annotation.get(g, (g, ""))[0],
             "chromosome": annotation.get(g, (g, ""))[1], "mapped": g in annotation,
             "mitochondrial": g in mitochondrial_ids} for g in gene_ids]
    table = pd.DataFrame(rows)
    symbol_mito = table.output_gene_name.str.startswith("MT-")
    if not np.array_equal(symbol_mito.to_numpy(), table.mitochondrial.to_numpy()):
        raise ValueError("Mitochondrial gene symbols and chromosome annotation disagree")
    audit = {"annotation": receipt, "annotation_manifest_sha256": sha256(annotation_manifest),
        "n_input_genes": len(gene_ids), "n_mapped_ids": int(table.mapped.sum()),
        "n_unmapped_ids_retained": int((~table.mapped).sum()), "n_ambiguous_ids": 0,
        "mitochondrial_reference_gene_count": len(mitochondrial_ids),
        "mitochondrial_input_ids": sorted(observed_mito),
        "mapping_rule": "Exact stable Ensembl ID; strip only GTF version; retain unmapped IDs and counts"}
    return table.output_gene_name.tolist(), table, audit


def prepare_lawlor(raw_dir, output_dir, config=None):
    """Canonical counts and RNA-only eligibility; no effect evaluation.

    The caller must freeze its experiment protocol before invoking this stage.
    Raw abundance scales/clones are not certified for inter-study transfer.
    """
    from .datasets import audit_lawlor
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    cfg = _config(config)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Prepared output exists; use a new version")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_lawlor(raw_dir)
    if not all(item["verified"] for item in audit["files"].values()):
        raise ValueError("All Lawlor inputs must pass the published checksum gate")
    guard = ResourceGuard(output_dir, max_rss_gib=cfg["max_rss_gib"], minimum_free_gib=cfg["minimum_free_gib"], max_hours=cfg["max_hours"])
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
    selected = meta.Demuxlet_Classification.eq("SNG") & meta.HTO_Classification.isin(["Baseline", "LPS", "CD3_CD28"]) & meta.Donor_of_Origin.notna()
    rna = raw_rna.T.tocsr()[selected.to_numpy()]
    cells = np.asarray(rna_cells)[selected.to_numpy()]
    meta = meta.loc[selected].copy()
    del raw_rna
    rna, genes = _collapse_genes(rna, genes)
    total = np.asarray(rna.sum(axis=1)).ravel()
    detected = np.diff(rna.indptr)
    mito_columns = np.array([str(g).upper().startswith("MT-") for g in genes])
    mitochondrial = np.asarray(rna[:, mito_columns].sum(axis=1)).ravel() / np.maximum(total, 1)
    lineage, lineage_rule = rna_lineage_labels(rna, genes)
    available = set(genes)
    lineage_rule["available_markers"] = {k: [g for g in v if g in available]
        for k, v in lineage_rule["markers"].items()}
    lineage_rule["unavailable_markers"] = {k: [g for g in v if g not in available]
        for k, v in lineage_rule["markers"].items()}
    keep = (detected >= cfg["rna_min_genes"]) & (total >= cfg["rna_min_counts"]) & (mitochondrial <= cfg["rna_max_mito_fraction"]) & (lineage != "unassigned")
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
    obs = pd.DataFrame({"cell_id": cells, "donor": meta.Donor_of_Origin.astype(str).to_numpy(),
        "lineage": lineage, "context": lineage, "condition": meta.HTO_Classification.to_numpy(),
        "perturbation": meta.HTO_Classification.map({"Baseline": "non-targeting", "LPS": "LPS", "CD3_CD28": "CD3_CD28"}).to_numpy(),
        "is_control": meta.HTO_Classification.eq("Baseline").to_numpy(), "guide": "not_applicable",
        "library": meta.Run_Identifier.astype(str).to_numpy(), "donor_status": "SNG",
        "rna_counts": total[keep], "rna_genes": detected[keep], "rna_mito_fraction": mitochondrial[keep]})
    amendment = {"dataset": "Lawlor_PBMC_CITEseq", "config": cfg, "rna_lineage_rule": lineage_rule,
        "gene_annotation": gene_annotation_audit,
        "eligibility": "Genotype singlet, known condition/donor, RNA QC and fixed RNA marker labels only",
        "source_sha256": sha256(__file__), "original_ADT_informed_annotation_used": False}
    write_json(output_dir / "preparation_amendment.json", amendment)
    manifest = {"dataset": "Lawlor_PBMC_CITEseq", "raw_files": audit["files"],
        "gene_annotation": gene_annotation_audit,
        "preparation_amendment_sha256": sha256(output_dir / "preparation_amendment.json"),
        "transfer_gate": "unverified_antibody_clone_and_scale_compatibility", "outcome_effects_evaluated": False,
        "source": "https://doi.org/10.3389/fimmu.2021.636720"}
    obs.groupby(["donor", "lineage", "condition"]).size().rename("n_cells").reset_index().to_csv(output_dir / "group_coverage.csv", index=False)
    guard.check()
    return save_prepared(PreparedDataset(rna, proteins, obs, genes,
        [canonical_protein(n) for n, use in zip(antibody_names, biological) if use], isotypes,
        [n for n, use in zip(antibody_names, is_iso) if use], manifest), output_dir)
