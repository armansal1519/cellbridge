"""RNA-only selection, aggregation and partitions with disjoint control cells.

For an affine RNA predictor, averaging predictions and predicting the average
are identical. We fit group means with explicit balanced weights to give
donors/contexts comparable influence rather than weighting by cell abundance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold


@dataclass
class GroupData:
    x: np.ndarray
    y: np.ndarray
    y_var: np.ndarray
    obs: pd.DataFrame
    genes: np.ndarray
    proteins: np.ndarray
    technical: np.ndarray | None = None
    technical_names: np.ndarray | None = None

    def subset(self, indices):
        indices = np.asarray(indices)
        return GroupData(self.x[indices], self.y[indices], self.y_var[indices], self.obs.iloc[indices].reset_index(drop=True),
            self.genes, self.proteins, None if self.technical is None else self.technical[indices], self.technical_names)

    def save(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "groups.npz", x=self.x, y=self.y, y_var=self.y_var,
            genes=np.asarray(self.genes, dtype=str), proteins=np.asarray(self.proteins, dtype=str),
            technical=np.empty((len(self.x), 0)) if self.technical is None else self.technical,
            technical_names=np.asarray([] if self.technical_names is None else self.technical_names, dtype=str))
        self.obs.to_csv(folder / "groups.csv", index=False)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        with np.load(folder / "groups.npz", allow_pickle=False) as v:
            return cls(v["x"], v["y"], v["y_var"], pd.read_csv(folder / "groups.csv", keep_default_na=False),
                v["genes"], v["proteins"], v["technical"], v["technical_names"])


def balanced_weights(obs):
    """Each donor and each donor-context stratum receive equal total weight."""
    obs = obs.reset_index(drop=True)
    keys = obs["donor"].astype(str) + "|" + obs["context"].astype(str)
    counts = keys.value_counts()
    contexts_per_donor = obs.groupby("donor")["context"].nunique()
    w = np.array([1 / (counts[k] * contexts_per_donor[d]) for k, d in zip(keys, obs.donor)])
    return w / w.mean()


def _stable_block(barcode, seed, blocks):
    return int(hashlib.sha256(f"{seed}:{barcode}".encode()).hexdigest()[:16], 16) % blocks


def aggregate(rna, protein_counts, obs, genes, proteins, *, seed=20261004,
              control_blocks=12, minimum_cells=30, technical_counts=None, technical_names=None,
              rna_chunk_cells=5000):
    """Independent transforms: log1p ADT; per-cell RNA library normalization.

    Input obs requires donor, lineage, context, perturbation, is_control and
    cell_id. No target abundance is consulted for selection or annotation.
    Control blocks are independent sets of cells, never re-used across a fold.
    Groups below minimum_cells are discarded by cell counts alone.
    """
    obs = obs.copy().reset_index(drop=True)
    required = {"donor", "lineage", "context", "perturbation", "is_control", "cell_id"}
    if required - set(obs): raise ValueError(f"Missing metadata: {required - set(obs)}")
    if not obs.cell_id.is_unique: raise ValueError("Duplicate cell identifiers")
    if len(obs) != rna.shape[0] or len(obs) != len(protein_counts): raise ValueError("Cell alignment mismatch")
    ctrl = obs.is_control.astype(str).str.lower().isin(["true", "1"])
    obs["is_control"] = ctrl
    if control_blocks < 1 or minimum_cells < 2 or rna_chunk_cells < 1:
        raise ValueError("Positive block/chunk sizes and at least two cells required")
    obs["block"] = obs.perturbation.astype(str)
    # Balanced, disjoint control blocks avoid randomly discarding small strata.
    # The number of blocks depends only on cell counts, never target abundance.
    for _, indices in obs.loc[ctrl].groupby(["donor", "lineage", "context"], sort=True).groups.items():
        indices = list(indices)
        k = min(control_blocks, max(1, len(indices) // minimum_cells))
        indices.sort(key=lambda i: hashlib.sha256(f"{seed}:{obs.at[i, 'cell_id']}".encode()).hexdigest())
        for block, members in enumerate(np.array_split(indices, k)):
            obs.loc[members, "block"] = f"control_{block:02d}"
    keys = ["donor", "lineage", "context", "perturbation", "is_control", "block"]
    grouped = obs.groupby(keys, sort=True, observed=True).indices
    retained = [(key, idx) for key, idx in grouped.items() if len(idx) >= minimum_cells]
    if not retained: raise ValueError("No groups pass minimum cell coverage")
    n = len(obs)
    row = np.concatenate([np.repeat(i, len(idx)) for i, (_, idx) in enumerate(retained)])
    col = np.concatenate([idx for _, idx in retained])
    w = np.concatenate([np.repeat(1 / len(idx), len(idx)) for _, idx in retained])
    averaging = sparse.csr_matrix((w, (row, col)), shape=(len(retained), n))
    # A full float64 CSR copy of a billion-entry matrix can exceed laptop RAM.
    # Only a bounded cell block is transformed before adding its group means.
    mean_x = np.zeros((len(retained), rna.shape[1]), dtype=np.float64)
    for start in range(0, n, rna_chunk_cells):
        stop = min(start + rna_chunk_cells, n)
        x = sparse.csr_matrix(rna[start:stop], dtype=np.float64)
        total = np.asarray(x.sum(axis=1)).ravel()
        if np.any(total <= 0): raise ValueError("Zero RNA library after QC")
        x = sparse.diags(1e4 / total) @ x
        np.log1p(x.data, out=x.data)
        mean_x += (averaging[:, start:stop] @ x).toarray()
    y = np.log1p(np.asarray(protein_counts, dtype=np.float64))
    if np.any(np.asarray(protein_counts) < 0) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid ADT counts")
    mean_y = np.asarray(averaging @ y)
    sizes = np.array([len(idx) for _, idx in retained])
    # Unbiased variance of each mean; cell-independence is a sampling model,
    # not a replacement for donor-level biological replication.
    var_y = np.maximum(np.asarray(averaging @ (y * y)) - mean_y**2, 0) / (sizes[:, None] - 1)
    frame = pd.DataFrame([key for key, _ in retained], columns=keys)
    frame["n_cells"] = sizes
    frame["group_id"] = [hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:20] for key, _ in retained]
    frame["cell_ids_sha256"] = [hashlib.sha256("\n".join(sorted(obs.iloc[idx].cell_id)).encode()).hexdigest() for _, idx in retained]
    if "guide" in obs:
        frame["guide_ids"] = [";".join(sorted(set(obs.iloc[idx].guide.astype(str)))) for _, idx in retained]
    tech = None if technical_counts is None else np.asarray(averaging @ np.log1p(technical_counts))
    return GroupData(mean_x, mean_y, var_y, frame, np.asarray(genes), np.asarray(proteins), tech, technical_names)


def fold_labels(obs, axis):
    if axis == "perturbation_gene": return obs.block.astype(str).to_numpy()
    if axis in {"donor", "context"}: return obs[axis].astype(str).to_numpy()
    raise ValueError(f"Unsupported partition axis: {axis}")


def make_splits(obs, axis, n_splits=5, *, leave_one_group_out=True):
    """Default donor/context behavior is LOO; nested CV can request grouped K-fold."""
    groups = fold_labels(obs, axis)
    k = min(len(np.unique(groups)), n_splits)
    if k < 2: raise ValueError("At least two independent groups needed")
    if leave_one_group_out and axis in {"donor", "context"}: k = len(np.unique(groups))
    splits = list(GroupKFold(k).split(np.arange(len(obs)), groups=groups))
    return [(np.asarray(train), np.asarray(test)) for train, test in splits]


def contrast_matrix(obs, minimum_cells=30):
    """Match within donor/lineage/context, using only controls in this partition."""
    ctrl = obs.is_control.astype(str).str.lower().isin(["true", "1"]).to_numpy()
    rows, meta = [], []
    for i in np.flatnonzero(~ctrl):
        row = obs.iloc[i]
        matched = ctrl & (obs.donor == row.donor).to_numpy() & (obs.lineage == row.lineage).to_numpy() & (obs.context == row.context).to_numpy()
        js = np.flatnonzero(matched)
        if not len(js) or obs.iloc[js].n_cells.sum() < minimum_cells or row.n_cells < minimum_cells: continue
        c = np.zeros(len(obs)); c[i] = 1
        counts = obs.iloc[js].n_cells.to_numpy(dtype=float)
        c[js] = -counts / counts.sum()
        rows.append(c)
        record = row.to_dict(); record["control_cells"] = int(counts.sum())
        record["reference_ids"] = ";".join(obs.iloc[js].group_id)
        meta.append(record)
    if not rows: raise ValueError("No eligible matched-control contrasts in partition")
    return np.stack(rows), pd.DataFrame(meta)


def paired_donor_gate(obs, minimum, required_perturbations, minimum_cells=30):
    """Metadata-only eligibility gate on the full cohort, never inside LODO.

    Counts matched donor pairs separately for every requested condition in each
    retained lineage/context. This is a design threshold, not a power guarantee.
    """
    if not isinstance(minimum, (int, np.integer)) or minimum < 2:
        raise ValueError("minimum paired donors must be an integer of at least two")
    if not required_perturbations or len(set(required_perturbations)) != len(required_perturbations):
        raise ValueError("required_perturbations must be a nonempty unique list")
    _, matched = contrast_matrix(obs, minimum_cells=minimum_cells)
    records = []
    for lineage, context in obs[["lineage", "context"]].drop_duplicates().itertuples(index=False, name=None):
        for condition in required_perturbations:
            eligible = matched.loc[(matched.lineage == lineage) & (matched.context == context) &
                                   (matched.perturbation == condition)]
            donors = sorted(eligible.donor.astype(str).unique().tolist())
            if len(donors) < minimum:
                raise ValueError(f"Paired donor gate failed for {lineage}/{context}/{condition}: "
                                 f"{len(donors)} eligible donors; require {minimum}")
            records.append({"lineage": str(lineage), "context": str(context),
                            "perturbation": str(condition), "paired_donors": len(donors),
                            "donor_ids": donors})
    return records


def abundance_group_weights(obs, mode="legacy_rows"):
    """Weight group-mean RNA training; contrast-response weights are unchanged.

    condition_balanced gives equal weight to donors, contexts within a donor,
    and perturbations within a donor/context. Rows within a condition share its
    weight, so duplicating all control blocks does not change condition mass.
    Recompute inside every RNA training and validation partition. Slicing the
    original weights can change a condition's total when blocks are held out.
    """
    if mode == "legacy_rows":
        return balanced_weights(obs)
    if mode != "condition_balanced":
        raise ValueError(f"Unknown abundance group weighting: {mode}")
    if not len(obs):
        raise ValueError("Cannot weight empty metadata")
    keys = ["donor", "context", "perturbation"]
    missing = set(keys) - set(obs)
    if missing:
        raise ValueError(f"Missing weight metadata: {sorted(missing)}")
    obs = obs.reset_index(drop=True)
    rows_per_condition = obs.groupby(keys, dropna=False)["perturbation"].transform("size").to_numpy(float)
    conditions_per_context = obs.groupby(keys[:2], dropna=False)["perturbation"].transform("nunique").to_numpy(float)
    contexts_per_donor = obs.groupby("donor", dropna=False)["context"].transform("nunique").to_numpy(float)
    weights = 1 / (rows_per_condition * conditions_per_context * contexts_per_donor)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Invalid abundance weight metadata")
    return weights / weights.mean()
