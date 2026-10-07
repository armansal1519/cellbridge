"""Fixed-hyperparameter donor-deletion sensitivity; no independent CI claim."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .artifacts import complete_stage, require_receipt, validate_receipt, write_json
from .shrinkage import ResponseShrinkage


def _read(path):
    with np.load(path, allow_pickle=False) as v:
        return {k: v[k] for k in v.files}


def training_sensitivity(run, protocol, guard=None):
    rows = []; root = Path(run)
    roster = json.loads((root / 'target_roster.json').read_text())
    for fold in sorted(root.glob('fold_*')):
        for name in ('fit', 'development', 'queries/test'):
            require_receipt(fold / name)
        dest = fold / 'training_sensitivity'
        if not validate_receipt(dest):
            base = ResponseShrinkage.load(fold / 'fit/response_shrinkage.npz')
            tr = _read(fold / 'development/responses.npz')
            panels = json.loads((fold / 'panels.json').read_text())
            queries = {p: _read(fold / 'queries/test' / f'{p}.npz') for p in panels}
            deleted = sorted(set(tr['donors'])); predictions = {p: [] for p in panels}
            coefficients = {p: [] for p in panels}
            for donor in deleted:
                if guard: guard()
                keep = tr['donors'] != donor
                obj = ResponseShrinkage().fit(tr['x'][keep], tr['y'][keep], tr['donors'][keep],
                    alpha=base.alpha, shrinkage=base.shrinkage, n_hvg=base.n_hvg, guard=guard)
                for panel, q in queries.items():
                    pp = obj.predict(q['x'], q['anchor_values'], q['anchor_indices'])
                    predictions[panel].append(pp['prediction']); coefficients[panel].append(pp['anchor_operator'])
            dest.mkdir(parents=True, exist_ok=True)
            for panel, q in queries.items():
                np.savez_compressed(dest / f'{panel}.npz', predictions=np.stack(predictions[panel]),
                    coefficients=np.stack(coefficients[panel]), deleted_donors=np.array(deleted), query_donors=q['donors'])
            write_json(dest / 'interpretation.json', {'conditional_on': 'Selected alpha, shrinkage and panel',
                'refitted': 'RNA features/scaling/coefficients and every residual crossfit/covariance after removing each training donor',
                'not_refitted': 'Hyperparameter selection and cohort definition',
                'meaning': 'Empirical training-set sensitivity, not a frequentist parameter confidence interval or calibrated prediction interval',
                'test_outcomes_accessed': False})
            complete_stage(dest)
        for panel in json.loads((fold / 'panels.json').read_text()):
            values = _read(dest / f'{panel}.npz')
            pred = values['predictions']
            for j, donor in enumerate(values['query_donors']):
                for p in roster['secondary']:
                    rows.append({'fold': fold.name, 'panel': panel, 'donor': str(donor), 'protein': roster['proteins'][p],
                        'training_deletion_sd': float(pred[:, j, p].std(ddof=1)),
                        'training_deletion_min': float(pred[:, j, p].min()), 'training_deletion_max': float(pred[:, j, p].max()),
                        'deleted_training_donors': len(pred)})
    output = root / 'training_sensitivity'
    if not validate_receipt(output):
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output / 'sensitivity.csv', index=False)
        complete_stage(output, {'empirical_sensitivity_not_interval': True})
