"""Nested, donor/context-aware response benchmarks and blind query artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .artifacts import complete_stage, fingerprint, sha256, validate_receipt, write_json, require_receipt, validate_scvaeit
from .benchmark_data import GroupData, balanced_weights, abundance_group_weights, contrast_matrix, fold_labels, make_splits
from .rna import RNARegressor, choose_alpha, cross_fit


def _save_arrays(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".partial").open("wb") as f: np.savez_compressed(f, **arrays)
    path.with_suffix(".partial").replace(path)


def _read(path):
    with np.load(path, allow_pickle=False) as data: return {k: data[k] for k in data.files}


def _contrast(data, baseline):
    c, obs = contrast_matrix(data.obs)
    return {"x": c @ data.x, "y": c @ data.y, "baseline": c @ baseline,
            "variance": (c*c) @ data.y_var,
            "technical": np.empty((len(c), 0)) if data.technical is None else c @ data.technical}, obs


def _oof_contrasts(data, oof, fold_ids):
    arrays, metas = [], []
    for fold in np.unique(fold_ids):
        ids = np.flatnonzero(fold_ids == fold)
        try: part, obs = _contrast(data.subset(ids), oof[ids])
        except ValueError: continue
        arrays.append(part); obs["oof_fold"] = int(fold); metas.append(obs)
    if not arrays: raise ValueError("No OOF matched contrasts; insufficient control coverage")
    return {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}, pd.concat(metas, ignore_index=True)


def _fit_bundle(data, protocol):
    weight_mode = protocol.get("abundance_group_weighting", "legacy_rows")
    residual_axis = protocol.get("residual_crossfit_axis", "perturbation_gene")
    weights = abundance_group_weights(data.obs, weight_mode)
    weight_fn = None if weight_mode == "legacy_rows" else lambda ids: abundance_group_weights(data.obs.iloc[ids], weight_mode)
    groups = fold_labels(data.obs, residual_axis)
    hvg, alphas = protocol["preprocessing"]["rna_hvg"], protocol["ridge_alphas"]
    alpha, scores = choose_alpha(data.x, data.y, groups, weights, alphas, hvg, weight_fn=weight_fn)
    model = RNARegressor(alpha, hvg).fit(data.x, data.y, weights)
    oof, ids, oof_alphas = cross_fit(data.x, data.y, groups, weights, alphas, hvg, weight_fn=weight_fn)
    contrasts, obs = _oof_contrasts(data, oof, ids)
    if residual_axis == "donor":
        _, expected = contrast_matrix(data.obs)
        if len(obs) != len(expected):
            raise ValueError("Donor residual cross-fitting lost eligible matched contrasts")
    return model, contrasts, obs, data.y-oof, {"alpha": alpha, "alpha_scores": scores,
        "oof_alphas": oof_alphas, "residual_crossfit_axis": residual_axis,
        "abundance_group_weighting": weight_mode,
        "rna_alpha_scoring": "equal_fold_mean_partition_balanced" if weight_fn is not None else "legacy_pooled_fixed_weights"}


def freeze_splits(data, run, axis, protocol, limit=None):
    import hashlib
    run = Path(run)
    if limit is not None and limit < 1: raise ValueError("Fold limit must be positive")
    eligibility = None
    if protocol.get("minimum_paired_donors_primary") is not None:
        from .benchmark_data import paired_donor_gate
        if axis != "donor" or protocol.get("residual_crossfit_axis") != "donor" or protocol.get("inner_axis") != "donor":
            raise ValueError("Paired-donor protocol requires donor outer, residual and inner splits")
        eligibility = paired_donor_gate(data.obs, protocol["minimum_paired_donors_primary"],
            protocol["primary_perturbations"], protocol.get("minimum_cells_per_arm", 30))
    splits = make_splits(data.obs, axis)
    digests = {name: hashlib.sha256(np.ascontiguousarray(getattr(data, name)).tobytes()).hexdigest()
               for name in ["x", "y", "y_var"]}
    payload = {"axis": axis, "group_ids": data.obs.group_id.to_list(), "metadata_digest": fingerprint(data.obs.to_dict("list")),
        "array_sha256": digests, "genes": data.genes.astype(str).tolist(), "proteins": data.proteins.astype(str).tolist(),
        "folds": [{"train": train.tolist(), "test": test.tolist()} for train, test in splits],
        "interpretation": "Gene folds include disjoint hash-partitioned controls; donor/context splits are disjoint.",
        "pilot_fold_limit": limit}
    if eligibility is not None:
        payload["paired_donor_eligibility"] = eligibility
        payload["inner_axis"] = protocol["inner_axis"]
        payload["inner_folds"] = protocol["inner_folds"]
    write_json(run / "splits.json", payload, immutable=True)
    return splits if limit is None else splits[:limit]


def fit_rna_stage(data, run, protocol, *, axis="perturbation_gene", limit=None, guard=None):
    run = Path(run)
    splits = freeze_splits(data, run, axis, protocol, limit)
    for fold, (train, test) in enumerate(splits):
        dest = run / f"fold_{fold:02d}" / "rna"
        if validate_receipt(dest): continue
        if guard: guard.check()
        dest.mkdir(parents=True, exist_ok=True)
        dev, query = data.subset(train), data.subset(test)
        model, contrast, obs, abundance, diagnostic = _fit_bundle(dev, protocol)
        model.save(dest / "model.npz")
        comparator_weights = ({"weights": abundance_group_weights(dev.obs, "condition_balanced")}
            if protocol.get("abundance_group_weighting") == "condition_balanced" else {})
        _save_arrays(dest / "development_groups.npz", x=dev.x, y=dev.y, **comparator_weights)
        _save_arrays(dest / "train.npz", **contrast, scales=model.y_scale,
                     abundance_residuals=abundance, abundance_weights=abundance_group_weights(
                         dev.obs, protocol.get("abundance_group_weighting", "legacy_rows")),
                     technical_names=np.asarray([] if data.technical_names is None else data.technical_names, dtype=str),
                     proteins=data.proteins.astype(str))
        obs.to_csv(dest / "train.csv", index=False)
        write_json(dest / "fit.json", diagnostic)
        # Outcome vault is consumed by evaluation and the privileged masking
        # step only; the predictor below accepts blind query files exclusively.
        outcome, outcome_obs = _contrast(query, model.predict(query.x))
        vault = dest.parent / "outcomes"; vault.mkdir(exist_ok=True)
        _save_arrays(vault / "test.npz", **outcome)
        c, _ = contrast_matrix(query.obs)
        _save_arrays(vault / "groups.npz", x=query.x, y=query.y, contrast=c)
        outcome_obs.to_csv(vault / "test.csv", index=False)
        complete_stage(vault)
        for inner, (it, iv) in enumerate(make_splits(dev.obs, protocol.get("inner_axis", "perturbation_gene"),
                protocol["inner_folds"], leave_one_group_out=False)):
            if guard: guard.check()
            inner_dir = dest / f"inner_{inner:02d}"
            if validate_receipt(inner_dir): continue
            inner_dir.mkdir(exist_ok=True)
            im, ic, io, _, diag = _fit_bundle(dev.subset(it), protocol)
            im.save(inner_dir / "model.npz")
            vc, vo = _contrast(dev.subset(iv), im.predict(dev.x[iv]))
            _save_arrays(inner_dir / "train.npz", **ic, scales=im.y_scale)
            _save_arrays(inner_dir / "validation.npz", **vc)
            io.to_csv(inner_dir / "train.csv", index=False); vo.to_csv(inner_dir / "validation.csv", index=False)
            write_json(inner_dir / "fit.json", diag)
            complete_stage(inner_dir)
        complete_stage(dest, {"outer_train_groups": len(train), "outer_test_groups": len(test)})


def save_basis(basis, path):
    _save_arrays(path, loadings=basis.loadings, scales=basis.scales,
        singular_values=basis.singular_values, protein_names=np.asarray(basis.protein_names, dtype=str))


def load_basis(path):
    from .model import ResponseBasis
    return ResponseBasis(**_read(path))


def _targets(proteins, names):
    lookup = {str(p): i for i, p in enumerate(proteins)}
    missing = set(names)-set(lookup)
    if missing: raise ValueError(f"Required proteins absent after antigen audit: {sorted(missing)}")
    return np.array([lookup[n] for n in names], dtype=int)


def _primary_response_mask(obs, primary_perturbations=None):
    """Prespecified response subset, independent of ADT outcomes."""
    if primary_perturbations is None:
        mask = np.ones(len(obs), dtype=bool)
    else:
        if not isinstance(primary_perturbations, (list, tuple)) or not primary_perturbations:
            raise ValueError("primary_perturbations must be a nonempty list")
        if len(set(primary_perturbations)) != len(primary_perturbations):
            raise ValueError("primary_perturbations must contain unique conditions")
        mask = obs.perturbation.isin(primary_perturbations).to_numpy()
    if not np.any(mask):
        raise ValueError("No prespecified primary response in validation/evaluation partition")
    return mask


def _prediction(basis, val, anchors):
    from .model import ObservableInput, ResponseBridge
    return ResponseBridge(basis).predict(ObservableInput(val["baseline"],
        (val["y"]-val["baseline"])[:, anchors], anchors)).prediction


def _development_noise(train, obs):
    """Worst donor/context mean sampling SD, using only development ADTs."""
    strata = obs.groupby(["donor", "context"], sort=True).indices
    return np.sqrt(np.max([train["variance"][ids].mean(axis=0) for ids in strata.values()], axis=0))


def fit_response_stage(run, protocol, guard=None):
    from .model import ResponseBasis, NumericalRankError
    from .panels import select_panel
    from .uncertainty import bootstrap_basis
    run = Path(run)
    failure_policy = protocol.get("bootstrap_rank_failure_policy", "raise")
    if failure_policy not in {"raise", "refuse_hidden"}:
        raise ValueError("bootstrap_rank_failure_policy must be raise or refuse_hidden")
    for fold in sorted(run.glob("fold_*")):
        rna, out = fold / "rna", fold / "response"
        require_receipt(rna)
        if validate_receipt(out): continue
        out.mkdir(exist_ok=True)
        train = _read(rna / "train.npz"); proteins = train["proteins"]
        targets = _targets(proteins, protocol["primary_targets"])
        excluded = _targets(proteins, protocol["forbidden_anchors"])
        obs = pd.read_csv(rna / "train.csv")
        scores, errors = [], {}
        for size in protocol["panel_sizes"]:
            for rank in protocol["ranks"]:
                rank_scores, rank_errors = [], []
                for inner in sorted(rna.glob("inner_*")):
                    if guard: guard.check()
                    tr, va = _read(inner / "train.npz"), _read(inner / "validation.npz")
                    to, vo = pd.read_csv(inner / "train.csv"), pd.read_csv(inner / "validation.csv")
                    primary = _primary_response_mask(vo, protocol.get("primary_perturbations"))
                    if rank > min(tr["y"].shape): continue
                    try:
                        basis = ResponseBasis.fit(tr["y"]-tr["baseline"], rank,
                            sample_weight=balanced_weights(to), protein_names=proteins, scales=tr["scales"])
                    except NumericalRankError:
                        continue
                    panel = select_panel(basis, size, targets, excluded_indices=excluded,
                        anchor_noise=_development_noise(tr, to))
                    pred = _prediction(basis, va, panel.indices)
                    if not np.isfinite(pred[primary][:, targets]).all():
                        continue
                    standardized = (pred - va["y"])/tr["scales"]
                    rank_scores.append(float(np.average(np.mean(np.abs(standardized[primary][:, targets]), axis=1),
                        weights=balanced_weights(vo.loc[primary]))))
                    rank_errors.append(standardized)
                if rank_scores and len(rank_scores) == len(list(rna.glob("inner_*"))):
                    scores.append({"size": size, "rank": rank, "error": float(np.mean(rank_scores)), "fold_scores": rank_scores})
                    errors[size, rank] = np.concatenate(rank_errors)
        selected = {}
        for size in protocol["panel_sizes"]:
            candidates = [s for s in scores if s["size"] == size]
            if not candidates: raise ValueError("No valid ranks in development CV")
            chosen = min(candidates, key=lambda s: (s["error"], s["rank"]))
            rank = chosen["rank"]; selected[str(size)] = rank
            basis = ResponseBasis.fit(train["y"]-train["baseline"], rank,
                sample_weight=balanced_weights(obs), protein_names=proteins, scales=train["scales"])
            save_basis(basis, out / f"basis_{size}.npz")
            bootstrap_groups = obs.donor.astype(str).to_numpy()
            bootstrap_unit = "donor"
            if len(np.unique(bootstrap_groups)) < 2:
                bootstrap_groups = obs.context.astype(str).to_numpy()
                bootstrap_unit = "context_within_single_development_donor"
            bootstrap = bootstrap_basis(train["y"]-train["baseline"], bootstrap_groups, rank,
                n_bootstrap=protocol["bootstrap_replicates"], sample_weight=balanced_weights(obs),
                protein_names=proteins, scales=train["scales"], seed=protocol["seed"]+size,
                failure_policy="record" if failure_policy == "refuse_hidden" else "raise")
            draws = bootstrap.draws if failure_policy == "refuse_hidden" else bootstrap
            valid = bootstrap.valid if failure_policy == "refuse_hidden" else np.ones(len(draws), dtype=bool)
            failures = bootstrap.failures if failure_policy == "refuse_hidden" else []
            _save_arrays(out / f"bootstrap_{size}.npz", valid=valid,
                loadings=np.stack([np.full_like(basis.loadings, np.nan) if b is None else b.loadings for b in draws]))
            write_json(out / f"bootstrap_{size}.json", {"unit": bootstrap_unit,
                "independent_units": len(np.unique(bootstrap_groups)), "replicates": len(draws),
                "rank_failure_policy": failure_policy, "valid_replicates": int(valid.sum()),
                "failed_replicates": int((~valid).sum()), "failures": failures,
                "failure_interpretation": "Failed draws retain their original slots. Any unavailable draw refuses hidden targets; no rank reduction, replacement or successful-only bootstrap quantile.",
                "interpretation": "Conditional basis sensitivity, holding upstream RNA fits/scales fixed. Sparse donor counts cannot support population CIs."})
            # Post-selection development residual envelopes are descriptive,
            # not distribution-free calibrated coverage or independent tests.
            err = errors[size, rank] * train["scales"]
            q = np.nanquantile(np.abs(err), 1-protocol["uncertainty_alpha"], axis=0)
            _save_arrays(out / f"calibration_{size}.npz", context_error_quantiles=q, development_errors=err)
            # Abundance ablation holds RNA model, rank, panel and inputs fixed.
            ab = ResponseBasis.fit(train["abundance_residuals"], rank, sample_weight=train["abundance_weights"],
                protein_names=proteins, scales=train["scales"])
            save_basis(ab, out / f"abundance_basis_{size}.npz")
        write_json(out / "rank_selection.json", {"selected": selected, "scores": scores,
            "primary_perturbations": protocol.get("primary_perturbations"),
            "calibration_limit": "Rank-selected development envelopes. Honest outer folds assess coverage; not universal unseen-context CIs."})
        complete_stage(out)


def select_panel_stage(run, protocol):
    from .panels import select_panel
    from .model import ResponseBasis
    for fold in sorted(Path(run).glob("fold_*")):
        dest = fold / "panels"
        require_receipt(fold / "rna"); require_receipt(fold / "response")
        if validate_receipt(dest): continue
        dest.mkdir(exist_ok=True)
        panels = []
        for size in protocol["panel_sizes"]:
            basis = load_basis(fold / "response" / f"basis_{size}.npz")
            targets = _targets(basis.protein_names, protocol["primary_targets"])
            excluded = _targets(basis.protein_names, protocol["forbidden_anchors"])
            tr = _read(fold / "rna/train.npz")
            meta = pd.read_csv(fold / "rna/train.csv")
            chosen = select_panel(basis, size, targets, excluded_indices=excluded,
                anchor_noise=_development_noise(tr, meta))
            choices = [("optimized", chosen.indices), ("fixed", _targets(basis.protein_names, protocol[f"fixed_panel_{size}"]))]
            rng = np.random.default_rng(protocol["seed"] + size)
            allowed = np.setdiff1d(np.arange(len(basis.protein_names)), excluded)
            choices += [(f"random_{i:02d}", rng.choice(allowed, size=size, replace=False)) for i in range(protocol["random_panel_replicates"])]
            for kind, indices in choices:
                panels.append({"id": f"{kind}_{size}", "kind": kind, "size": size,
                    "indices": np.asarray(indices, dtype=int).tolist(), "proteins": [str(basis.protein_names[i]) for i in indices]})
                # Use each actual panel, never reuse the optimized panel's
                # errors for the fixed/random panel's uncertainty envelope.
                errors = []
                for inner in sorted((fold / "rna").glob("inner_*")):
                    tr, va = _read(inner / "train.npz"), _read(inner / "validation.npz")
                    meta = pd.read_csv(inner / "train.csv")
                    ib = ResponseBasis.fit(tr["y"]-tr["baseline"], basis.rank,
                        sample_weight=balanced_weights(meta), protein_names=basis.protein_names, scales=tr["scales"])
                    errors.append(np.abs(_prediction(ib, va, indices)-va["y"]))
                err = np.concatenate(errors)
                finite = np.isfinite(err).all(axis=0)
                q = np.zeros(err.shape[1])
                q[finite] = np.quantile(err[:, finite], 1-protocol["uncertainty_alpha"], axis=0)
                _save_arrays(dest / f"{kind}_{size}_envelope.npz", quantiles=q, available=finite)
        write_json(dest / "panels.json", panels, immutable=True); complete_stage(dest)


def build_blind_query(baseline, measured_anchor_response, anchor_variance, anchor_indices, x_features):
    """This boundary intentionally has no argument for hidden target values."""
    a = np.asarray(anchor_indices, int)
    if measured_anchor_response.shape != (len(baseline), len(a)): raise ValueError("Anchor shape mismatch")
    return {"baseline": np.asarray(baseline), "anchor_residuals": measured_anchor_response-baseline[:, a],
        "anchor_response": measured_anchor_response, "anchor_variance": anchor_variance,
        "anchor_indices": a, "x_features": x_features}


def mask_queries_stage(run):
    """Privileged public-data masking step; writes strictly observable inputs."""
    for fold in sorted(Path(run).glob("fold_*")):
        dest = fold / "queries"
        for prerequisite in ["rna", "panels", "outcomes"]: require_receipt(fold / prerequisite)
        if validate_receipt(dest): continue
        dest.mkdir(exist_ok=True)
        outcome = _read(fold / "outcomes" / "test.npz")
        groups = _read(fold / "outcomes" / "groups.npz")
        rna = RNARegressor.load(fold / "rna" / "model.npz")
        for panel in json.loads((fold / "panels" / "panels.json").read_text()):
            a = np.array(panel["indices"])
            blind = build_blind_query(outcome["baseline"], outcome["y"][:, a], outcome["variance"][:, a], a,
                outcome["x"][:, rna.features] / rna.x_scale)
            _save_arrays(dest / f"{panel['id']}.npz", **blind)
            _save_arrays(dest / f"{panel['id']}_scvaeit.npz", x=groups["x"],
                anchor_values=groups["y"][:, a], anchor_indices=a)
        _save_arrays(dest / "contrasts.npz", contrast=groups["contrast"])
        complete_stage(dest, {"hidden_target_inputs": False, "normalization": "Independent log1p; train-only RNA features"})


def _ridge_prediction(train_x, train_y, query_x, weights, alpha):
    mean = np.average(train_x, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((train_x-mean)**2, axis=0, weights=weights)), 1e-6)
    model = Ridge(alpha=alpha, solver="cholesky").fit((train_x-mean)/scale, train_y, sample_weight=weights/weights.sum())
    return model.predict((query_x-mean)/scale)


def _select_baseline_alpha(fold, anchors, kind, alphas, targets, primary_perturbations=None):
    # Reuse exactly the nested response-validation partitions. Splitting an
    # already constructed contrast table would share control observations and
    # upstream RNA fits across the penalty-selection boundary.
    validations = []
    for inner in sorted((fold / "rna").glob("inner_*")):
        tr, va = _read(inner / "train.npz"), _read(inner / "validation.npz")
        to, vo = pd.read_csv(inner / "train.csv"), pd.read_csv(inner / "validation.csv")
        primary = _primary_response_mask(vo, primary_perturbations)
        if kind == "direct":
            rna = RNARegressor.load(inner / "model.npz")
            tx = np.column_stack([tr["x"][:, rna.features]/rna.x_scale, tr["y"][:, anchors]/tr["scales"][anchors]])
            vx = np.column_stack([va["x"][:, rna.features]/rna.x_scale, va["y"][:, anchors]/tr["scales"][anchors]])
            ty, vy = tr["y"], va["y"]
        else:
            ty, vy = tr["y"]-tr["baseline"], va["y"]-va["baseline"]
            tx, vx = ty[:, anchors], vy[:, anchors]
        validations.append((tx, ty, vx[primary], vy[primary], tr["scales"],
            balanced_weights(to), balanced_weights(vo.loc[primary])))
    if not validations: raise ValueError("Baseline selection needs nested validation artifacts")
    scores = []
    for alpha in alphas:
        fold_scores = []
        for tx, ty, vx, vy, scales, tw, vw in validations:
            pred = _ridge_prediction(tx, ty, vx, tw, alpha)
            errors = np.abs((pred-vy)/scales)[:, targets].mean(axis=1)
            fold_scores.append(np.average(errors, weights=vw))
        scores.append(np.mean(fold_scores))
    return alphas[int(np.argmin(scores))]


def predict_stage(run, protocol, guard=None):
    from .model import ObservableInput, ResponseBridge, ResponseBasis
    for fold in sorted(Path(run).glob("fold_*")):
        dest = fold / "predictions"
        for prerequisite in ["rna", "response", "panels", "queries"]: require_receipt(fold / prerequisite)
        if validate_receipt(dest): continue
        dest.mkdir(exist_ok=True)
        train = _read(fold / "rna" / "train.npz")
        obs = pd.read_csv(fold / "rna" / "train.csv"); weights = balanced_weights(obs)
        rna = RNARegressor.load(fold / "rna" / "model.npz")
        targets = _targets(train["proteins"], protocol["primary_targets"])
        baseline_alpha, basis_availability = {}, {}
        for panel in json.loads((fold / "panels" / "panels.json").read_text()):
            if guard: guard.check()
            query = _read(fold / "queries" / f"{panel['id']}.npz")
            a = query["anchor_indices"]
            inputs = ObservableInput(query["baseline"], query["anchor_residuals"], a)
            basis = load_basis(fold / "response" / f"basis_{panel['size']}.npz")
            envelope = _read(fold / "panels" / f"{panel['id']}_envelope.npz")
            q = envelope["quantiles"]
            bootstrap = _read(fold / "response" / f"bootstrap_{panel['size']}.npz")
            loadings = bootstrap["loadings"]
            valid = bootstrap.get("valid", np.ones(len(loadings), dtype=bool))
            if valid.shape != (len(loadings),) or valid.dtype != np.dtype(bool):
                raise ValueError("Bootstrap validity must be a boolean for every retained draw slot")
            if not valid.all() and protocol.get("bootstrap_rank_failure_policy", "raise") != "refuse_hidden":
                raise ValueError("Unavailable bootstrap draws require the explicit refuse_hidden protocol")
            if not np.isnan(loadings[~valid]).all():
                raise ValueError("Unavailable bootstrap slots must retain NaN loadings")
            draws = [ResponseBasis(x, basis.scales, protein_names=basis.protein_names) if ok else None
                for x, ok in zip(loadings, valid)]
            covariance = np.array([np.diag(v) for v in query["anchor_variance"]])
            result = ResponseBridge(basis).predict(inputs, anchor_covariance=covariance,
                basis_draws=draws, context_error_quantiles=q, alpha=protocol["uncertainty_alpha"])
            tx = train["x"][:, rna.features]/rna.x_scale
            train_direct = np.column_stack([tx, train["y"][:, a]/train["scales"][a]])
            query_direct = np.column_stack([query["x_features"], query["anchor_response"]/train["scales"][a]])
            alpha_direct = _select_baseline_alpha(fold, a, "direct", protocol["ridge_alphas"], targets,
                protocol.get("primary_perturbations"))
            direct = _ridge_prediction(train_direct, train["y"], query_direct, weights, alpha_direct)
            residual = train["y"]-train["baseline"]
            alpha_residual = _select_baseline_alpha(fold, a, "residual", protocol["ridge_alphas"], targets,
                protocol.get("primary_perturbations"))
            residual_pred = query["baseline"]+_ridge_prediction(residual[:, a], residual, query["anchor_residuals"], weights, alpha_residual)
            abundance = ResponseBridge(load_basis(fold / "response" / f"abundance_basis_{panel['size']}.npz")).predict(inputs).prediction
            _save_arrays(dest / f"{panel['id']}.npz", responsebridge=result.prediction, rna_ridge=query["baseline"],
                without_identifiability_refusal=result.raw_prediction, accepted=result.accepted,
                availability_status=result.availability_status, basis_draw_valid=valid,
                basis_draws_complete=np.array(valid.all()),
                rna_anchor_ridge=direct, residual_ridge=residual_pred, abundance_basis=abundance,
                observable=result.observable, null_ambiguity=result.null_ambiguity,
                noise_amplification=result.noise_amplification, measurement_variance=result.measurement_variance,
                basis_error_radius=result.basis_error_radius, context_error_radius=result.context_error_radius,
                lower=np.where(envelope["available"], result.lower, np.nan),
                upper=np.where(envelope["available"], result.upper, np.nan), anchor_indices=a)
            baseline_alpha[panel["id"]] = {"rna_anchor_ridge": alpha_direct, "residual_ridge": alpha_residual}
            basis_availability[panel["id"]] = {"complete": bool(valid.all()),
                "valid_draws": int(valid.sum()), "unavailable_draw_indices": np.flatnonzero(~valid).tolist()}
        write_json(dest / "assumptions.json", {"identifiability": "Conditional on the learned response basis, not biology in an arbitrary new context",
            "measurement_covariance": "Diagonal variance-of-means approximation; no donor-population interpretation",
            "basis_uncertainty": "Conditional donor-block (single-donor context-block fallback) basis refits. RNA fit uncertainty excluded; too few donors for population CIs.",
            "context_uncertainty": "Rank-selected inner-development error envelope; external coverage must be measured",
            "transfer": "Antigen/clone/scale compatibility must be verified before cross-study transfer",
            "scVAEIT": "Separate optional runner; not silently substituted", "baseline_alphas": baseline_alpha,
            "bootstrap_rank_failure_policy": protocol.get("bootstrap_rank_failure_policy", "raise"),
            "basis_draw_availability": basis_availability,
            "unavailable_basis_policy": "Any unavailable requested-rank bootstrap draw refuses all hidden target points/envelopes. Base observability and raw diagnostic remain separate; no successful-only basis quantile."})
        complete_stage(dest)


def evaluate_stage(run, protocol):
    """Score all targets, and compare refusal on the identical target-row set.

    A method cannot meet the accuracy gate by silently dropping hard responses.
    Coverage is always reported; direction error is undefined for refusals.
    """
    dest = Path(run) / "evaluation"
    if validate_receipt(dest):
        return pd.read_csv(dest / "primary_summary.csv")
    records = []
    has_published_baseline = True
    for fold in sorted(Path(run).glob("fold_*")):
        for prerequisite in ["rna", "predictions", "outcomes"]: require_receipt(fold / prerequisite)
        published_complete = validate_scvaeit(fold / "scvaeit")
        truth = _read(fold / "outcomes" / "test.npz")["y"]
        meta = pd.read_csv(fold / "outcomes" / "test.csv")
        training = _read(fold / "rna" / "train.npz")
        names, scales = training["proteins"], training["scales"]
        for path in sorted((fold / "predictions").glob("*.npz")):
            pred = _read(path)
            hidden = np.setdiff1d(np.arange(len(names)), pred["anchor_indices"])
            methods = ["responsebridge", "rna_ridge", "rna_anchor_ridge", "residual_ridge",
                       "abundance_basis", "without_identifiability_refusal"]
            published = fold / "scvaeit" / path.name
            if published.exists() and published_complete:
                pred["scVAEIT_Gaussian_group_adapter"] = _read(published)["prediction"]
                methods.append("scVAEIT_Gaussian_group_adapter")
            else:
                has_published_baseline = False
            for method in methods:
                error = pred[method]-truth
                for j in hidden:
                    for i in range(len(truth)):
                        finite = bool(np.isfinite(pred[method][i, j]))
                        interval = method == "responsebridge" and bool(np.isfinite(pred["lower"][i, j]) and np.isfinite(pred["upper"][i, j]))
                        records.append({"fold": fold.name, "panel": path.stem, "method": method,
                            "protein": str(names[j]), "donor": str(meta.iloc[i].donor), "context": str(meta.iloc[i].context),
                            "perturbation": str(meta.iloc[i].perturbation), "group_id": str(meta.iloc[i].group_id),
                            "truth": float(truth[i, j]), "prediction": float(pred[method][i, j]),
                            "absolute_error": float(abs(error[i, j])), "squared_error": float(error[i, j]**2),
                            "standardized_absolute_error": float(abs(error[i,j])/scales[j]),
                            "standardized_squared_error": float((error[i,j]/scales[j])**2),
                            "signed_bias": float(error[i, j]), "prediction_available": finite,
                            "common_accepted": bool(np.isfinite(pred["responsebridge"][i, j])),
                            "direction_error": float(np.sign(pred[method][i,j]) != np.sign(truth[i,j])) if finite else np.nan,
                            "observable": bool(pred["observable"][j]) if method == "responsebridge" else None,
                            "availability_status": (str(pred["availability_status"][j]) if "availability_status" in pred else
                                ("available" if finite else "prediction_unavailable")) if method == "responsebridge" else
                                ("available" if finite else "prediction_unavailable"),
                            "basis_draws_complete": bool(pred.get("basis_draws_complete", True)) if method == "responsebridge" else None,
                            "noise_amplification": float(pred["noise_amplification"][j]) if method == "responsebridge" else np.nan,
                            "interval_available": interval if method == "responsebridge" else np.nan,
                            "precision_usable": bool(finite and interval and
                                (pred["upper"][i,j]-pred["lower"][i,j])/2 <= protocol["equivalence_margin_development_sd"]*scales[j]) if method == "responsebridge" else np.nan,
                            "interval_covered": float(pred["lower"][i,j] <= truth[i,j] <= pred["upper"][i,j]) if interval else np.nan,
                            "interval_width": float(pred["upper"][i,j]-pred["lower"][i,j]) if interval else np.nan})
    if not records: raise ValueError("No predictions to evaluate")
    frame = pd.DataFrame(records)
    dest.mkdir(exist_ok=True)
    frame.to_csv(dest / "effects.csv.gz", index=False)
    primary = frame.loc[frame.protein.isin(protocol["primary_targets"])]
    primary = primary.loc[_primary_response_mask(primary, protocol.get("primary_perturbations"))]
    metrics = ["standardized_absolute_error", "absolute_error", "squared_error", "standardized_squared_error",
               "signed_bias", "prediction_available", "direction_error", "interval_available", "precision_usable", "interval_covered", "interval_width"]
    def summarize(data):
        # First average repeated perturbations within each context, then contexts
        # within donor, then donors. Cells and control blocks are not replicates.
        contexts = data.groupby(["panel", "method", "donor", "context"], dropna=False)[metrics].mean()
        donors = contexts.groupby(["panel", "method", "donor"]).mean()
        summary = donors.groupby(["panel", "method"]).mean().reset_index()
        summary["rmse"] = np.sqrt(summary.squared_error)
        summary["standardized_rmse"] = np.sqrt(summary.standardized_squared_error)
        return summary
    summary = summarize(primary)
    summary.to_csv(dest / "primary_summary.csv", index=False)
    common = summarize(primary.loc[primary.common_accepted])
    common.to_csv(dest / "common_accepted_summary.csv", index=False)
    summarize(frame).to_csv(dest / "all_hidden_summary.csv", index=False)
    # Preserve both donor-transfer directions instead of conflating two donors
    # with a sample large enough for population confidence intervals.
    primary.groupby(["fold", "donor", "panel", "method"])[metrics].mean().to_csv(dest / "donor_directions.csv")
    comparisons = []
    for panel, group in summary.loc[summary.panel.str.startswith("optimized")].groupby("panel"):
        values = group.set_index("method")
        rb, ridge = values.loc["responsebridge"], values.loc["rna_anchor_ridge"]
        gain = 1-rb.standardized_absolute_error/max(ridge.standardized_absolute_error, 1e-12)
        direction = rb.direction_error-ridge.direction_error
        complete = bool(rb.prediction_available == 1)
        comparisons.append({"panel": panel, "relative_mae_gain": float(gain) if complete and np.isfinite(gain) else None,
            "direction_error_change": float(direction) if complete and np.isfinite(direction) else None,
            "accuracy_comparison_scope": "full_primary" if complete else "unavailable_incomplete_primary_coverage",
            "point_available_fraction": float(rb.prediction_available),
            "precision_usable_fraction": float(rb.precision_usable),
            "operational_accuracy_gate_passed": bool(complete and gain >= protocol["advancement"]["minimum_relative_mae_gain_over_rna_anchor_ridge"]
                and direction <= protocol["advancement"]["maximum_increase_direction_error"])})
    evaluated_donors = int(primary.donor.nunique())
    unresolved = ["Independent-study replication", "Technical-adjusted independent biological validation",
                  "Published comparator convergence and development-only tuning adequacy"]
    if evaluated_donors <= 2:
        unresolved.append("Population uncertainty with more than two donors")
    if not has_published_baseline: unresolved.append("Executed published panel-completion baseline")
    write_json(dest / "advancement.json", {"comparisons": comparisons, "publication_ready": False,
        "evaluated_donors": evaluated_donors,
        "published_group_adapter_executed": has_published_baseline, "unresolved": unresolved,
        "gate_refusal_policy": "Requires full primary point-prediction coverage; common accepted-row performance is reported separately.",
        "pilot": json.loads((Path(run)/"splits.json").read_text()).get("pilot_fold_limit") is not None})
    from .diagnostics import technical_audit
    technical_audit(run, protocol)
    complete_stage(dest)
    return summary
