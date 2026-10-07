"""Post-evaluation donor-level technical associations, never model selection.

Reads group isotypes/count metadata and optional prepared cell metadata only.
It never opens raw counts or materializes the hidden protein/RNA group arrays.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import complete_stage, require_receipt, sha256, validate_receipt, write_json

METHODS = ('abundance_response_shrinkage', 'two_penalty_ridge')
TARGETS = ('CD25', 'CD69')
RNA_QC = ('rna_counts', 'rna_genes', 'rna_mito_fraction')


def _read(path):
    return json.loads(Path(path).read_text())


def _control(frame):
    return frame.is_control.astype(str).str.lower().isin(('true', '1'))


def _verify_seal(run):
    seal = _read(run / 'prediction_seal.json')
    for name, digest in seal['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha256(run / rel) != digest:
            raise ValueError('Prediction seal is invalid before technical diagnostics')
    return seal


def _cell_metadata(run):
    lock = run / 'cell_input_lock.json'
    if not lock.exists():
        return None, {'status': 'unavailable_no_cell_input_lock'}
    prepared = Path(_read(lock)['prepared'])
    path = prepared / 'obs.csv'
    if not path.exists():
        return None, {'status': 'unavailable_no_obs_csv', 'prepared': str(prepared)}
    digest = sha256(path)
    manifest_path = prepared / 'manifest.json'
    if not manifest_path.exists():
        return None, {'status': 'unavailable_unverified_obs_metadata'}
    expected = _read(manifest_path).get('files', {}).get('obs.csv', {}).get('sha256')
    if expected is None:
        return None, {'status': 'unavailable_obs_checksum_not_in_manifest'}
    if digest != expected:
        raise ValueError('Prepared obs metadata checksum changed')
    allowed = {'cell_id', 'donor', 'lineage', 'context', 'perturbation', 'condition', 'is_control', *RNA_QC}
    obs = pd.read_csv(path, usecols=lambda c: c in allowed, dtype={'donor': str}, keep_default_na=False)
    needed = {'donor', 'lineage', 'perturbation', 'is_control'}
    if needed - set(obs):
        return None, {'status': 'unavailable_missing_obs_fields', 'missing': sorted(needed-set(obs))}
    if 'cell_id' in obs and obs.cell_id.duplicated().any():
        raise ValueError('Duplicate cell identity in prepared metadata')
    return obs, {'status': 'available', 'obs_sha256': digest, 'manifest_sha256': sha256(manifest_path),
        'population': 'Already prepared retained cells only; fractions are not raw PBMC composition'}


def _technical_features(groups, lock, donors, protocol, obs):
    for name, key in (('groups.npz', 'groups_sha256'), ('groups.csv', 'metadata_sha256')):
        if sha256(groups/name) != lock[key]:
            raise ValueError('Frozen group input changed: '+name)
    meta = pd.read_csv(groups/'groups.csv', dtype={'donor': str}, keep_default_na=False)
    with np.load(groups/'groups.npz', allow_pickle=False) as values:
        # Deliberately do not call GroupData.load, which loads hidden y and RNA x.
        technical = np.asarray(values['technical']) if 'technical' in values else np.empty((len(meta), 0))
        names = values['technical_names'].astype(str).tolist() if 'technical_names' in values else []
    if technical.shape != (len(meta), len(names)) or len(set(names)) != len(names):
        raise ValueError('Technical group dimensions/names disagree')
    rows = []
    numeric = ['control_cells', 'stimulated_cells', 'stimulated_control_cell_ratio']
    for name in names:
        numeric += [f'isotype__{name}__{arm}' for arm in ('control', 'stimulated', 'delta')]
    for key in RNA_QC:
        numeric += [f'{key}_median_{arm}' for arm in ('control', 'stimulated', 'delta')]
    lineages = sorted(obs.lineage.astype(str).unique()) if obs is not None else []
    for name in lineages:
        numeric += [f'prepared_fraction__{name}__{arm}' for arm in ('control', 'stimulated', 'delta')]
    for donor in sorted(donors):
        row = {'donor': donor, 'metadata_status': 'available', 'isotype_status': 'available' if names else 'unavailable_no_isotypes',
               'cell_metadata_status': 'available' if obs is not None else 'unavailable'}
        selected = (meta.donor == donor) & (meta.lineage.astype(str) == protocol['cohort_lineage'])
        stimulated = selected & ~_control(meta) & (meta.perturbation.astype(str) == protocol['primary_stimulation'])
        indices = np.flatnonzero(stimulated)
        if len(indices) != 1:
            row.update({name: np.nan for name in numeric})
            row['metadata_status'] = 'unavailable_missing_or_ambiguous_stimulation_group'
            rows.append(row)
            continue
        stim_index = indices[0]
        context = str(meta.iloc[stim_index]['context'])
        control = selected & _control(meta) & (meta.context.astype(str) == context)
        controls = np.flatnonzero(control)
        weights = pd.to_numeric(meta.loc[control, 'n_cells'], errors='coerce').to_numpy(float)
        nstim = float(meta.iloc[stim_index]['n_cells'])
        if not len(controls) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isfinite(nstim) or nstim <= 0:
            raise ValueError('Invalid matched control/stimulation cell counts')
        row.update(control_cells=float(weights.sum()), stimulated_cells=nstim, stimulated_control_cell_ratio=nstim/weights.sum())
        for j, name in enumerate(names):
            base = float(np.average(technical[controls, j], weights=weights))
            active = float(technical[stim_index, j])
            for arm, value in [('control', base), ('stimulated', active), ('delta', active-base)]:
                row[f'isotype__{name}__{arm}'] = value
        for key in RNA_QC:
            medians = {}
            for arm in ('control', 'stimulated'):
                mask = None if obs is None else ((obs.donor == donor) & (obs.lineage.astype(str) == protocol['cohort_lineage']))
                if obs is not None:
                    mask &= _control(obs) if arm == 'control' else (~_control(obs) & (obs.perturbation.astype(str) == protocol['primary_stimulation']))
                    if 'context' in obs: mask &= obs.context.astype(str) == context
                values = pd.to_numeric(obs.loc[mask, key], errors='coerce') if obs is not None and key in obs else pd.Series(dtype=float)
                values = values[np.isfinite(values)]
                medians[arm] = float(values.median()) if len(values) else np.nan
                row[f'{key}_median_{arm}'] = medians[arm]
            row[f'{key}_median_delta'] = medians['stimulated']-medians['control']
        if obs is not None:
            for name in lineages:
                fractions = {}
                for arm in ('control', 'stimulated'):
                    mask = obs.donor == donor
                    mask &= _control(obs) if arm == 'control' else (~_control(obs) & (obs.perturbation.astype(str) == protocol['primary_stimulation']))
                    cells = obs.loc[mask]
                    fractions[arm] = float((cells.lineage.astype(str) == name).mean()) if len(cells) else np.nan
                    row[f'prepared_fraction__{name}__{arm}'] = fractions[arm]
                row[f'prepared_fraction__{name}__delta'] = fractions['stimulated']-fractions['control']
        rows.append(row)
    return pd.DataFrame(rows), numeric


def run_diagnostics(run, protocol):
    """Write immutable diagnostics only after sealed evaluation has completed."""
    run = Path(run)
    require_receipt(run/'evaluation')  # Check before reading any donor metadata.
    if _read(run/'protocol.json') != protocol:
        raise ValueError('Diagnostics protocol differs from frozen run protocol')
    seal = _verify_seal(run)
    independent = protocol['evaluation_mode'] == 'independent_test'
    if protocol['evaluation_mode'] not in ('independent_test', 'development_lodo'):
        raise ValueError('Unknown diagnostics evaluation mode')
    dest = run/'technical_diagnostics'
    if dest.exists():
        if validate_receipt(dest): return dest
        raise FileExistsError('Incomplete diagnostics output is preserved; use a new declared run')
    effects = pd.read_csv(run/'evaluation/effects.csv', dtype={'donor': str})
    effects = effects.loc[effects.method.isin(METHODS) & effects.protein.isin(TARGETS)].copy()
    if effects.empty: raise ValueError('No primary new-model/comparator evaluation rows')
    roles = {}
    for fold in sorted(effects.fold.unique()):
        name = f'{fold}/roles.json'
        if independent and name not in seal['files']:
            raise ValueError('Independent donor roles are not sealed')
        split = _read(run/name)
        parts = [set(map(str, split[key])) for key in ('train', 'calibration', 'test')]
        if any(parts[i] & parts[j] for i in range(3) for j in range(i)):
            raise ValueError('Donor roles overlap')
        roles[str(fold)] = parts[2]
    if any(str(row.donor) not in roles[str(row.fold)] for row in effects.itertuples()):
        raise ValueError('Technical diagnostics may report evaluated test donors only')
    keys = ['fold', 'panel', 'donor', 'protein', 'method']
    if effects.duplicated(keys).any():raise ValueError('Duplicate donor/target error rows')
    donors = set.union(*roles.values())
    cell, cell_status = _cell_metadata(run)
    if cell is not None:cell = cell.loc[cell.donor.isin(donors)].copy()
    lock = _read(run/'input_lock.json')
    features, columns = _technical_features(Path(lock['groups']), lock, donors, protocol, cell)
    role_name = 'independent_test' if independent else 'development_held_out'
    features['evaluation_role'] = role_name
    # Reindex onto every evaluated donor, panel, target and method, so absent
    # errors remain explicit and never disappear via an available-case mean.
    roster = [(fold, panel, donor, protein, method)
        for fold, test in roles.items() for panel in sorted(effects.loc[effects.fold==fold, 'panel'].unique())
        for donor in sorted(test) for protein in TARGETS for method in METHODS]
    index = pd.MultiIndex.from_tuples(roster, names=keys)
    error = effects.set_index(keys).reindex(index).reset_index()
    keep = keys + ['standardized_abs_error']
    error = error[keep]
    error['error_status'] = np.where(np.isfinite(error.standardized_abs_error), 'available', 'unavailable')
    joined = error.merge(features, on='donor', how='left', validate='many_to_one')
    correlations = []
    for (panel, method, protein), frame in joined.groupby(['panel', 'method', 'protein'], sort=True):
        # Collapse any repeated evaluation appearance to one donor; never count
        # folds/cells as independent observations for a correlation denominator.
        by_donor = frame.groupby('donor')[['standardized_abs_error', *columns]].mean()
        for column in columns:
            selected = by_donor[['standardized_abs_error', column]].replace([np.inf,-np.inf], np.nan).dropna()
            usable = len(selected) >= 3 and (selected.nunique() > 1).all()
            correlations.append({'panel':panel,'method':method,'protein':protein,'technical_feature':column,
                'donors_total':len(by_donor),'donors_complete':len(selected),
                'pearson_r':float(selected.corr().iloc[0,1]) if usable else np.nan,
                'status':'descriptive_only' if usable else 'unavailable_insufficient_or_constant_metadata',
                'evaluation_role':role_name})
    availability = [{'technical_feature':column,'donors_total':len(features),
        'donors_available':int(np.isfinite(pd.to_numeric(features[column],errors='coerce')).sum()),
        'status':'available' if np.isfinite(pd.to_numeric(features[column],errors='coerce')).any() else 'unavailable'} for column in columns]
    dest.mkdir(parents=True, exist_ok=False)
    features.to_csv(dest/'technical_counts.csv', index=False)
    joined.to_csv(dest/'donor_errors_with_metadata.csv', index=False)
    pd.DataFrame(correlations).to_csv(dest/'descriptive_correlations.csv', index=False)
    pd.DataFrame(availability).to_csv(dest/'metadata_availability.csv', index=False)
    write_json(dest/'scope.json', {'evaluation_mode':protocol['evaluation_mode'],'evaluation_role':role_name,
        'evaluation_receipt_sha256':sha256(run/'evaluation/complete.json'),'prediction_seal_sha256':sha256(run/'prediction_seal.json'),
        'group_input_lock_sha256':sha256(run/'input_lock.json'),'cell_metadata':cell_status,
        'donors':sorted(donors),'isotype_scale':'Cell-count weighted group mean cellwise log1p counts; stimulation minus matched pooled control',
        'hidden_ADT_totals_used':False,'raw_counts_opened':False,'model_selection_or_adjustment':False,
        'p_values_or_independent_fold_tests':False,'causal_or_cell_intrinsic_inference':False})
    (dest/'report.md').write_text('# Descriptive technical diagnostics\n\n'
        f'Evaluated scope: {role_name}; {len(donors)} donors. Every missing error/metadata item remains explicit.\n\n'
        'The tables relate donor prediction error to cell coverage, RNA QC medians, available isotype response contrasts, '
        'and broad lineage fractions among already prepared cells. These fractions do not measure unfiltered PBMC composition; '
        'T-only prepared inputs cannot establish changes between T-cell subtypes.\n\n'
        'Correlations are descriptive associations with complete-pair and full donor counts. No p-values, model selection, '
        'isotype subtraction, hidden-ADT total normalization or causal adjustment is performed. An association cannot show '
        'that technical noise caused an error, or that a biological pattern survives technical adjustment.\n\n'
        f'Optional cell metadata status: {cell_status["status"]}. Missing or constant features have unavailable correlations.\n')
    complete_stage(dest, {'purpose':'Post-evaluation technical associations only; no tuning'})
    return dest
