"""Previous abundance-RNA/low-rank architecture on the fixed v0.3 donor panels.

This preserves abundance RNA alpha selection, nested residual cross-fitting,
uncentered low-rank fitting, and failed-bootstrap refusals. Panel optimization
is replaced by the same existing panels for an equal-information comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .artifacts import complete_stage, require_receipt, validate_receipt, write_json
from .benchmark_data import GroupData, make_splits, balanced_weights
from .experiment import _fit_bundle, _contrast, save_basis, load_basis
from .model import ResponseBasis, ResponseBridge, ObservableInput, NumericalRankError
from .rna import RNARegressor
from .uncertainty import bootstrap_basis


def _read(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k] for k in value.files}


def _response(rna, x):
    return rna.predict(x) - rna.predict(np.zeros((1, x.shape[1])))


def fit_legacy(fold, protocol, guard=None):
    fold = Path(fold); dest = fold / 'legacy_fit'
    require_receipt(fold / 'development')
    if validate_receipt(dest):
        return
    raw = _read(fold / 'development/abundance.npz')
    obs = pd.read_csv(fold / 'development/abundance.csv', keep_default_na=False)
    data = GroupData(raw['x'], raw['y'], raw['y_var'], obs, raw['genes'], raw['proteins'])
    old = {'preprocessing': {'rna_hvg': protocol['rna_hvg']},
           'ridge_alphas': [a for a in protocol['ridge_alphas'] if a is not None],
           'abundance_group_weighting': 'condition_balanced', 'residual_crossfit_axis': 'donor'}
    if guard: guard()
    rna, train, train_obs, _, diagnostic = _fit_bundle(data, old)
    panels = json.loads((fold / 'panels.json').read_text())
    targets = np.array([list(data.proteins).index(p) for p in protocol['primary_targets']])
    inner = []
    for it, iv in make_splits(data.obs, 'donor', protocol.get('legacy_inner_folds', 3), leave_one_group_out=False):
        if guard: guard()
        im, tr, to, _, _ = _fit_bundle(data.subset(it), old)
        va, _ = _contrast(data.subset(iv), im.predict(data.x[iv]))
        inner.append((im, tr, to, va))
    dest.mkdir(parents=True, exist_ok=True)
    rna.save(dest / 'rna.npz')
    scores = {}
    for panel, av in panels.items():
        a = np.array(av, dtype=int); candidates = []
        for rank in protocol['ablation_ranks']:
            losses = []
            for im, tr, to, va in inner:
                try:
                    b = ResponseBasis.fit(tr['y']-tr['baseline'], rank,
                        sample_weight=balanced_weights(to), scales=im.y_scale, protein_names=data.proteins)
                    pred = ResponseBridge(b).predict(ObservableInput(va['baseline'], (va['y']-va['baseline'])[:, a], a)).prediction
                    loss = np.mean(np.abs((pred-va['y'])/im.y_scale)[:, targets])
                    losses.append(float(loss) if np.isfinite(loss) else np.inf)
                except NumericalRankError:
                    losses.append(np.inf)
            candidates.append({'rank': rank, 'mae': float(np.mean(losses))})
        valid = [s for s in candidates if np.isfinite(s['mae'])]
        score = {'scores': [{**s, 'mae': s['mae'] if np.isfinite(s['mae']) else None} for s in candidates]}
        if valid:
            rank = min(valid, key=lambda s: (s['mae'], s['rank']))['rank']
            b = ResponseBasis.fit(train['y']-train['baseline'], rank,
                sample_weight=balanced_weights(train_obs), scales=rna.y_scale, protein_names=data.proteins)
            save_basis(b, dest / f'{panel}_basis.npz')
            boot = bootstrap_basis(train['y']-train['baseline'], train_obs.donor.astype(str), rank,
                n_bootstrap=protocol.get('legacy_bootstrap_replicates', 200), sample_weight=balanced_weights(train_obs),
                protein_names=data.proteins, scales=rna.y_scale, seed=protocol['seed']+len(a), failure_policy='record')
            np.savez_compressed(dest / f'{panel}_bootstrap.npz', valid=boot.valid,
                loadings=np.stack([np.full_like(b.loadings, np.nan) if v is None else v.loadings for v in boot.draws]))
            score.update(selected_rank=rank, failed_bootstraps=int((~boot.valid).sum()), bootstrap_failures=boot.failures)
        else:
            score['status'] = 'no_identified_development_rank'
        scores[panel] = score
    np.savez_compressed(dest / 'residual_scale.npz',
                        scale=np.maximum(np.sqrt(np.mean((train['y']-train['baseline'])**2, axis=0)), 1e-6))
    write_json(dest / 'fit.json', {'rna': diagnostic, 'panels': scores,
        'changes_from_historical_run': 'Same CD3/CD28-only donor cohort and fixed panels; rank selection scores those fixed panels. Historical panel optimization and outcomes are not reused.',
        'uncertainty': 'Residual scale is descriptive. Legacy bootstrap failures and structural refusal are retained; independent conformal calibration can still refuse.'})
    complete_stage(dest)


def predict_legacy(fold, panel, query):
    """No hidden targets or outcome vault are read."""
    dest = Path(fold) / 'legacy_fit'; require_receipt(dest)
    rna = RNARegressor.load(dest / 'rna.npz')
    baseline = _response(rna, query['x']); a = query['anchor_indices']
    path = dest / f'{panel}_basis.npz'
    if not path.exists():
        return np.full_like(baseline, np.nan), _read(dest / 'residual_scale.npz')['scale']
    basis = load_basis(path); boot = _read(dest / f'{panel}_bootstrap.npz')
    draws = [ResponseBasis(v, basis.scales, protein_names=basis.protein_names) if good else None
             for v, good in zip(boot['loadings'], boot['valid'])]
    result = ResponseBridge(basis).predict(ObservableInput(baseline, query['anchor_values']-baseline[:, a], a), basis_draws=draws)
    return result.prediction, _read(dest / 'residual_scale.npz')['scale']
