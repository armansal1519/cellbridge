"""Version 0.4 abundance-response experiments with explicitly separated outcome vaults."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .artifacts import (complete_stage, require_receipt, validate_receipt, write_json,
                        fingerprint, sha256)
from .benchmark_data import GroupData, contrast_matrix, abundance_group_weights
from .response_baselines import LinearResponse, tune_linear, response_scale
from .rna import RNARegressor
from .shrinkage import ResponseShrinkage, select_shrinkage


def load_arrays(path):
    with np.load(path, allow_pickle=False) as v:
        return {k:v[k] for k in v.files}


def arrays(path, **value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value={key:np.asarray(v) for key,v in value.items()}
    for key,v in value.items():
        if v.dtype.kind=='O':
            if not all(isinstance(item,str) for item in v.flat):
                raise ValueError(f'Object array is not permitted in {key}')
            value[key]=v.astype(str)
    with path.with_suffix('.partial').open('wb') as f: np.savez_compressed(f, **value)
    path.with_suffix('.partial').replace(path)


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def index_names(proteins,names):
    lookup={str(p):i for i,p in enumerate(proteins)}
    if set(names)-set(lookup): raise ValueError(f'Missing required antigens: {set(names)-set(lookup)}')
    return np.array([lookup[n] for n in names],dtype=int)


def _same_donors(actual, expected, label):
    if not np.array_equal(np.asarray(actual,dtype=str),np.asarray(expected,dtype=str)):
        raise ValueError(f'{label} donor order does not match frozen roles/outcomes')


def _aligned_residuals(model, donors):
    """Recover input order from the core's canonical order; one row per donor."""
    donors=np.asarray(donors,dtype=str)
    if len(np.unique(donors))!=len(donors) or len(np.unique(model.train_donors))!=len(model.train_donors):
        raise ValueError('Response experiment requires one matched response per donor')
    lookup={d:i for i,d in enumerate(model.train_donors)}
    if set(donors)!=set(lookup):raise ValueError('Residual donor identities do not match training inputs')
    return model.oof_residuals[[lookup[d] for d in donors]]


def cohort(groups, protocol):
    """One matched response per donor; only the prespecified lineage/stimulus."""
    data=GroupData.load(groups)
    ctrl=data.obs.is_control.astype(str).str.lower().isin(['true','1']).to_numpy()
    retain=(data.obs.lineage.astype(str)==protocol['cohort_lineage']).to_numpy() & (
        ctrl | (data.obs.perturbation.astype(str)==protocol['primary_stimulation']).to_numpy())
    data=data.subset(np.flatnonzero(retain))
    c, obs=contrast_matrix(data.obs,protocol['minimum_cells_per_arm'])
    order=np.argsort(obs.donor.astype(str).to_numpy(),kind='stable'); c=c[order]; obs=obs.iloc[order].reset_index(drop=True)
    if obs.donor.astype(str).duplicated().any():
        raise ValueError('One stimulation/context response per donor is required; do not pool times or conditions')
    if len(obs)<protocol['minimum_development_donors']: raise ValueError('Insufficient paired development donors')
    return data, {'x':c@data.x, 'y':c@data.y, 'variance':(c*c)@data.y_var,
        'donors':obs.donor.astype(str).to_numpy(), 'proteins':data.proteins.astype(str),
        'genes':data.genes.astype(str)}, obs



# The old implementation is a separately recorded competitive refit. Its
# artifacts never share mutable directories with the new fitted model.
def initialize(groups, run, protocol, donor_split=None, limit=None):
    from . import response_experiment as previous
    previous.initialize(groups, run, protocol, donor_split=donor_split, limit=limit)
    old = copy.deepcopy(protocol)
    old.update(version='0.3.0', model='response-shrinkage', primary_panel='fixed_4',
               primary_baseline='joint', status='competitive_refit_of_v030_algorithm')
    reference = Path(run) / 'reference_v030'
    write_json(reference / 'protocol.json', old, immutable=True)
    previous.initialize(groups, reference, old, donor_split=donor_split, limit=limit)
    for fold in sorted(Path(run).glob('fold_*')):
        roles=json.loads((fold/'roles.json').read_text())
        write_json(fold/'cell_roles.json', {**roles,
            'cohort_lineage':protocol['cohort_lineage'],
            'control_condition':protocol.get('control_condition','Baseline'),
            'stimulated_condition':protocol['primary_stimulation']}, immutable=True)


def _abundance_kwargs(a, keep=None):
    if keep is None: keep=np.ones(len(a['donors']),dtype=bool)
    return {'abundance_x':a['x'][keep], 'abundance_y':a['y'][keep],
            'abundance_donors':a['donors'][keep], 'abundance_weights':a['weights'][keep]}


def fit_stage(run, protocol, guard=None):
    from . import response_experiment as previous
    from .abundance_response import AbundanceResponseShrinkage, select_abundance_shrinkage
    from .two_penalty import tune_two_penalty
    run=Path(run)
    old=json.loads((run/'reference_v030/protocol.json').read_text())
    previous.fit_stage(run/'reference_v030',old,guard)
    roster=json.loads((run/'target_roster.json').read_text())
    targets=np.asarray(roster['primary'],dtype=int)
    for fold in sorted(run.glob('fold_*')):
        require_receipt(fold/'development'); dest=fold/'fit'
        if validate_receipt(dest): continue
        tr=load_arrays(fold/'development/responses.npz')
        ab=load_arrays(fold/'development/abundance.npz')
        x,y,donors=tr['x'],tr['y'],tr['donors']
        panels=json.loads((fold/'panels.json').read_text())
        model, selection=select_abundance_shrinkage(x,y,donors,
            np.asarray(panels['fixed_8']),targets,**_abundance_kwargs(ab),
            alphas=protocol['ridge_alphas'],shrinkages=protocol['shrinkages'],
            n_hvg=protocol['rna_hvg'],guard=guard)
        dest.mkdir(parents=True,exist_ok=True)
        model.save(dest/'abundance_response_shrinkage.npz')
        write_json(dest/'selection.json',clean(selection))
        matched=AbundanceResponseShrinkage().fit(x,y,donors,**_abundance_kwargs(ab),
            alpha=model.alpha,shrinkage=model.shrinkage,n_hvg=protocol['rna_hvg'],
            formulation='matched-response',guard=guard)
        matched.save(dest/'matched_response.npz')
        arrays(dest/'reference.npz',cohort_mean=y.mean(axis=0),scales=response_scale(y,donors),
            proteins=tr['proteins'],donors=donors)
        details={}
        for panel, aa in panels.items():
            a=np.asarray(aa,dtype=int)
            bm,info,oof=tune_two_penalty(x,y,donors,y[:,a],targets,
                alphas=protocol['ridge_block_alphas'],n_hvg=protocol['rna_hvg'],guard=guard)
            bm.save(dest/f'{panel}_two_penalty_ridge.npz')
            details[panel]=info
            arrays(dest/f'{panel}_two_penalty_development.npz',oof=oof,
                error_sd=np.maximum(np.sqrt(np.mean((oof-y)**2,axis=0)),1e-6))
        write_json(dest/'baselines.json',clean(details))
        write_json(dest/'uncertainty_contract.json',{
            'conformal_scale':'Same training response SD for every method',
            'covariance':'Observed donor OOF response errors including sampling noise; no extra ADT variance',
            'reference_v030':'Separately nested competitive refit with original fixed4 selection'})
        complete_stage(dest,{'training_donors':donors.tolist(),'model':'abundance-response-shrinkage'})


def predict_stage(run, protocol, guard=None):
    from . import response_experiment as previous
    from .abundance_response import AbundanceResponseShrinkage
    from .two_penalty import TwoPenaltyResponse
    run=Path(run); reference=run/'reference_v030'
    previous.predict_stage(reference,json.loads((reference/'protocol.json').read_text()),guard)
    for fold in sorted(run.glob('fold_*')):
        require_receipt(fold/'fit')
        for role in ('test','calibration'): require_receipt(fold/'queries'/role)
        oldfold=reference/fold.name; require_receipt(oldfold/'predictions')
        dest=fold/'predictions'
        if validate_receipt(dest): continue
        model=AbundanceResponseShrinkage.load(fold/'fit/abundance_response_shrinkage.npz')
        matched=AbundanceResponseShrinkage.load(fold/'fit/matched_response.npz')
        for role in ('test','calibration'):
            for panel in json.loads((fold/'panels.json').read_text()):
                if guard: guard()
                q=load_arrays(fold/'queries'/role/f'{panel}.npz')
                old=load_arrays(oldfold/'predictions'/role/f'{panel}.npz')
                _same_donors(old['donors'],q['donors'],'v0.3 reference prediction')
                x,a,observed=q['x'],q['anchor_indices'],q['anchor_values']
                pred={}; sd={}
                rename={'response_shrinkage':'v030_response_shrinkage',
                        'abundance_shrinkage':'v030_abundance_shrinkage',
                        'no_anchor_correction':'v030_no_anchor_correction'}
                for key in old:
                    if key.startswith('sd__'):
                        name=key[4:]; new=rename.get(name,name)
                        pred[new]=old[name];sd[new]=old[key]
                result=model.predict(x,observed,a)
                mr=matched.predict(x,observed,a)
                nr=model.predict(x,np.empty((len(x),0)),np.array([],dtype=int))
                for name,value in [('abundance_response_shrinkage',result),
                                   ('matched_response',mr),('no_anchor_correction',nr)]:
                    pred[name]=value['prediction']
                    sd[name]=np.sqrt(np.maximum(value['model_variance'],1e-12))
                bm=TwoPenaltyResponse.load(fold/'fit'/f'{panel}_two_penalty_ridge.npz')
                pred['two_penalty_ridge']=bm.predict(x,observed)
                sd['two_penalty_ridge']=load_arrays(fold/'fit'/f'{panel}_two_penalty_development.npz')['error_sd']
                removed=[]
                for j in range(len(a)):
                    keep=np.arange(len(a))!=j
                    removed.append(model.predict(x,observed[:,keep],a[keep])['prediction'])
                arrays(dest/role/f'{panel}.npz',**pred,**{f'sd__{k}':v for k,v in sd.items()},
                    donors=q['donors'],anchor_indices=a,base_response=result['base_response'],
                    rna_contribution=result['rna_contribution'],anchor_contributions=result['anchor_contributions'],
                    anchor_operator=result['anchor_operator'],removed_anchor_predictions=np.stack(removed,axis=-1))
        complete_stage(dest,{'query_contract':'Only paired RNA contrast and measured anchor contrast; no hidden outcomes',
                             'reference_receipt_sha256':sha256(oldfold/'predictions/complete.json')})


def _cell_predictions(fold, role, donors, proteins):
    from .scvaeit_cells_v040 import validate_cell_receipt as validate_neural
    from .sclinear_cells import validate_cell_receipt as validate_linear
    result={}; status={'fold':Path(fold).name}
    for name,validator in [('scvaeit_cells',validate_neural),('sclinear_cells',validate_linear)]:
        path=Path(fold)/name
        if not (path/'receipt.json').exists():
            status[name]='not_executed';continue
        if not validator(path): raise ValueError(f'Invalid {name} receipt')
        status[name]='executed'
        if role=='calibration' and not len(donors): continue
        data=load_arrays(path/('predictions.npz' if role=='test' else 'calibration_predictions.npz'))
        _same_donors(data['donor_ids'],donors,f'{name} {role}')
        if not np.array_equal(data['protein_names'].astype(str),np.asarray(proteins,dtype=str)):
            raise ValueError(f'{name} protein order mismatch')
        result[name]=data
    status['status']='executed' if all(status[n]=='executed' for n in ('scvaeit_cells','sclinear_cells')) else 'incomplete'
    return result,status


def _cell_methods(data,panel):
    result={}
    for name,d in data.items():
        values=np.asarray(d[panel]); seeds=d['seeds']
        if values.shape!=(len(seeds),len(d['donor_ids']),len(d['protein_names'])):
            raise ValueError(f'{name} prediction axes mismatch')
        if name=='sclinear_cells':
            if len(seeds)!=1: raise ValueError('scLinear expects one declared deterministic fit')
            result['sclinear_cell']=values[0]
        else:
            if len(np.unique(seeds))!=len(seeds):raise ValueError('Duplicate neural seed')
            result.update({f'scvaeit_cell_seed_{int(s)}':values[i] for i,s in enumerate(seeds)})
            result['scvaeit_cell_ensemble']=values.mean(axis=0)
    return result


def training_sensitivity(run,protocol,guard=None):
    from .abundance_response import AbundanceResponseShrinkage
    root=Path(run);rows=[]
    roster=json.loads((root/'target_roster.json').read_text())
    for fold in sorted(root.glob('fold_*')):
        require_receipt(fold/'fit');require_receipt(fold/'development')
        for role in ('test','calibration'):require_receipt(fold/'queries'/role)
        dest=fold/'training_sensitivity'
        if not validate_receipt(dest):
            model=AbundanceResponseShrinkage.load(fold/'fit/abundance_response_shrinkage.npz')
            tr=load_arrays(fold/'development/responses.npz')
            ab=load_arrays(fold/'development/abundance.npz')
            queries={p:load_arrays(fold/'queries/test'/f'{p}.npz') for p in json.loads((fold/'panels.json').read_text())}
            predictions={p:[] for p in queries};coefficients={p:[] for p in queries}
            donors=sorted(set(tr['donors']))
            for d in donors:
                keep=tr['donors']!=d
                obj=AbundanceResponseShrinkage().fit(tr['x'][keep],tr['y'][keep],tr['donors'][keep],
                    **_abundance_kwargs(ab,ab['donors']!=d),alpha=model.alpha,shrinkage=model.shrinkage,
                    n_hvg=protocol['rna_hvg'],guard=guard)
                for panel,q in queries.items():
                    p=obj.predict(q['x'],q['anchor_values'],q['anchor_indices'])
                    predictions[panel].append(p['prediction']);coefficients[panel].append(p['anchor_operator'])
            for panel,q in queries.items():
                arrays(dest/f'{panel}.npz',predictions=np.stack(predictions[panel]),coefficients=np.stack(coefficients[panel]),
                    deleted_donors=np.asarray(donors),query_donors=q['donors'])
            write_json(dest/'interpretation.json',{'conditional_on':'Fixed selected alpha/shrinkage; all RNA transforms and residual folds refitted',
                'meaning':'Training donor deletion sensitivity, not confidence or calibrated intervals','test_outcomes_accessed':False})
            complete_stage(dest)
        for panel in json.loads((fold/'panels.json').read_text()):
            saved=load_arrays(dest/f'{panel}.npz');p=saved['predictions']
            for j,d in enumerate(saved['query_donors']):
                for k in roster['secondary']:
                    rows.append({'fold':fold.name,'panel':panel,'donor':str(d),'protein':roster['proteins'][k],
                        'training_deletion_sd':float(p[:,j,k].std(ddof=1)),
                        'training_deletion_min':float(p[:,j,k].min()),'training_deletion_max':float(p[:,j,k].max())})
    dest=root/'training_sensitivity'
    if not validate_receipt(dest):
        dest.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(dest/'sensitivity.csv',index=False)
        complete_stage(dest,{'empirical_sensitivity_not_interval':True})

def calibrate_stage(run, protocol):
    from .reliability import (conditional_gaussian_intervals,fit_joint_calibration,apply_calibration,validate_donor_splits)
    roster=json.loads((Path(run)/'target_roster.json').read_text()); targets=np.array(roster['primary'])
    names=[roster['proteins'][i] for i in targets]
    for fold in sorted(Path(run).glob('fold_*')):
        require_receipt(fold/'predictions');require_receipt(fold/'fit');dest=fold/'calibration'
        roles=json.loads((fold/'roles.json').read_text());records={}
        validate_donor_splits(roles['train'],roles['calibration'],roles['test'])
        reference=load_arrays(fold/'fit/reference.npz')
        neural,_=_cell_predictions(fold,'test',roles['test'],roster['proteins'])
        neural_cal={}
        if roles['calibration']:
            require_receipt(fold/'outcomes/calibration')
            neural_cal,_=_cell_predictions(fold,'calibration',roles['calibration'],roster['proteins'])
        inputs={'predictions_receipt':sha256(fold/'predictions/complete.json'),
            'cell_receipts':{name:sha256(fold/name/'receipt.json') for name in ('scvaeit_cells','sclinear_cells') if (fold/name/'receipt.json').exists()},
            'calibration_truth_receipt':sha256(fold/'outcomes/calibration/complete.json') if roles['calibration'] else None,
            'roles':roles}
        write_json(dest/'inputs.json',inputs,immutable=True)
        if validate_receipt(dest): continue
        for panel in json.loads((fold/'panels.json').read_text()):
            test=load_arrays(fold/'predictions/test'/f'{panel}.npz'); output={}
            _same_donors(test['donors'],roles['test'],'Prediction test')
            methods={k[4:]:test[k[4:]] for k in test if k.startswith('sd__')}
            neural_methods=_cell_methods(neural,panel);methods.update(neural_methods)
            cal_methods=_cell_methods(neural_cal,panel)
            if roles['calibration']:
                c=load_arrays(fold/'predictions/calibration'/f'{panel}.npz')
                truth=load_arrays(fold/'outcomes/calibration/truth.npz')
                _same_donors(c['donors'],roles['calibration'],'Prediction calibration')
                _same_donors(truth['donors'],c['donors'],'Calibration truth')
                cal_methods.update({k[4:]:c[k[4:]] for k in c if k.startswith('sd__')})
            for method,pred in methods.items():
                is_neural=method in neural_methods
                sd=reference['scales'] if is_neural else test[f'sd__{method}']
                if is_neural:
                    # A training response SD is a valid fixed conformal score
                    # scale, but is not a neural-model posterior variance.
                    output[f'model_lower__{method}']=np.full_like(pred,np.nan)
                    output[f'model_upper__{method}']=np.full_like(pred,np.nan)
                else:
                    ci=conditional_gaussian_intervals(pred,sd,protocol['uncertainty_alpha'])
                    output[f'model_lower__{method}']=ci['lower']; output[f'model_upper__{method}']=ci['upper']
                if roles['calibration']:
                    cal_sd=reference['scales']
                    cal=fit_joint_calibration(truth['y'][:,targets],cal_methods[method][:,targets],
                        cal_sd[...,targets],truth['donors'],names,protocol['uncertainty_alpha'])
                    applied=apply_calibration(pred[:,targets],reference['scales'][...,targets],cal,names)
                    lo=np.full_like(pred,np.nan); hi=np.full_like(pred,np.nan)
                    lo[:,targets]=applied['lower']; hi[:,targets]=applied['upper']
                    output[f'lower__{method}']=lo;output[f'upper__{method}']=hi
                    cal['score_scale']='fixed_training_response_sd'
                    records[f'{panel}/{method}']=clean(cal)
                else:
                    output[f'lower__{method}']=np.full_like(pred,np.nan)
                    output[f'upper__{method}']=np.full_like(pred,np.nan)
                    records[f'{panel}/{method}']={'status':'no_independent_calibration_donors',
                        'guarantee':None,'model_interval_only':not is_neural,
                        'score_scale':'fixed_training_response_sd'}
            arrays(dest/f'{panel}.npz',**output)
        write_json(dest/'status.json',records); complete_stage(dest)
    write_json(Path(run)/'prediction_seal.json',{'files':_prediction_seal_files(run),
        'purpose':'Frozen models, predictions, explanations and calibrated intervals before test outcome evaluation'},immutable=True)


def _prediction_seal_files(run):
    """Include the exact currently present set of model/comparator receipts."""
    run=Path(run);files={}
    for fold in sorted(run.glob('fold_*')):
        for relative in ('fit/complete.json','predictions/complete.json','calibration/complete.json',
                         'roles.json','scvaeit_cells/receipt.json','sclinear_cells/receipt.json'):
            path=fold/relative
            if path.exists():files[str(path.relative_to(run))]=sha256(path)
    return files


def explain_stage(run,protocol):
    run=Path(run); roster=json.loads((run/'target_roster.json').read_text()); rows=[]; operator_rows=[]
    dest=run/'explanation'
    if validate_receipt(dest): return
    for fold in sorted(run.glob('fold_*')):
        require_receipt(fold/'predictions')
        for panel in json.loads((fold/'panels.json').read_text()):
            pred=load_arrays(fold/'predictions/test'/f'{panel}.npz')
            check=pred['base_response']+pred['rna_contribution']+pred['anchor_contributions'].sum(-1)
            if not np.allclose(check,pred['abundance_response_shrinkage'],rtol=1e-11,atol=1e-11): raise ValueError('Explanation sum mismatch')
            for p in roster['secondary']:
                for d,donor in enumerate(pred['donors']):
                    rows.append({'fold':fold.name,'panel':panel,'donor':donor,'target':roster['proteins'][p],
                        'component':'base_response','contribution':pred['base_response'][d,p]})
                    rows.append({'fold':fold.name,'panel':panel,'donor':donor,'target':roster['proteins'][p],
                        'component':'RNA','contribution':pred['rna_contribution'][d,p]})
                    for j,a in enumerate(pred['anchor_indices']):
                        rows.append({'fold':fold.name,'panel':panel,'donor':donor,'target':roster['proteins'][p],
                            'component':roster['proteins'][a],'contribution':pred['anchor_contributions'][d,p,j],
                            'removal_change':pred['removed_anchor_predictions'][d,p,j]-pred['abundance_response_shrinkage'][d,p]})
                for j,a in enumerate(pred['anchor_indices']):
                    operator_rows.append({'fold':fold.name,'panel':panel,'target':roster['proteins'][p],
                        'anchor':roster['proteins'][a],'coefficient':pred['anchor_operator'][p,j]})
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'contributions.csv',index=False)
    op=pd.DataFrame(operator_rows);op.to_csv(dest/'operators.csv',index=False)
    op.groupby(['panel','target','anchor']).coefficient.agg(['mean','std','min','max']).to_csv(dest/'donor_deletion_stability.csv')
    write_json(dest/'interpretation.json',{'sum_exact':True,'causal':False,
        'correlated_anchors':'Additive fitted-model associations; not uniquely allocated biological information',
        'stability':'Variation across donor-deleted development fits; not an independent-sample confidence interval'})
    complete_stage(dest)


def evaluate_stage(run,protocol):
    from .reliability import paired_gain_bootstrap, interval_score
    run=Path(run); dest=run/'evaluation'
    seal=json.loads((run/'prediction_seal.json').read_text())
    current=_prediction_seal_files(run)
    if set(current)!=set(seal['files']):
        raise ValueError('Prediction seal receipt set changed before evaluation; unsealed or removed comparator/model receipt')
    if current!=seal['files']:
        raise ValueError('Prediction seal changed before evaluation')
    for fold in sorted(run.glob('fold_*')):
        for path in ['fit','predictions','calibration','outcomes/test']:require_receipt(fold/path)
    if validate_receipt(dest): return
    roster=json.loads((run/'target_roster.json').read_text()); rows=[]; cell_status=[]
    for fold in sorted(run.glob('fold_*')):
        for path in ['predictions','calibration','outcomes/test']:require_receipt(fold/path)
        truth=load_arrays(fold/'outcomes/test/truth.npz');ref=load_arrays(fold/'fit/reference.npz')
        cstatus=json.loads((fold/'calibration/status.json').read_text())
        roles=json.loads((fold/'roles.json').read_text())
        _same_donors(truth['donors'],roles['test'],'Evaluation truth')
        cell_pred,status=_cell_predictions(fold,'test',truth['donors'],ref['proteins'])
        cell_status.append(status or {'fold':fold.name,'status':'not_executed'})
        for panel in json.loads((fold/'panels.json').read_text()):
            pp=load_arrays(fold/'predictions/test'/f'{panel}.npz')
            _same_donors(pp['donors'],truth['donors'],'Evaluation prediction')
            intervals=load_arrays(fold/'calibration'/f'{panel}.npz')
            methods={k[4:]:pp[k[4:]] for k in pp if k.startswith('sd__')}
            methods.update(_cell_methods(cell_pred,panel))
            for method,pred in methods.items():
                for i,donor in enumerate(truth['donors']):
                    for p in roster['secondary']:
                        y=float(truth['y'][i,p]);yh=float(pred[i,p]);scale=float(ref['scales'][p]); mu=float(ref['cohort_mean'][p])
                        lo=intervals.get(f'lower__{method}',np.full_like(pred,np.nan))[i,p]
                        hi=intervals.get(f'upper__{method}',np.full_like(pred,np.nan))[i,p]
                        ml=intervals.get(f'model_lower__{method}',np.full_like(pred,np.nan))[i,p]
                        mh=intervals.get(f'model_upper__{method}',np.full_like(pred,np.nan))[i,p]
                        rows.append({'fold':fold.name,'panel':panel,'method':method,'donor':str(donor),
                            'protein':roster['proteins'][p],'primary':p in roster['primary'],
                            'observed':y,'prediction':yh,'training_mean':mu,'scale':scale,'available':np.isfinite(yh),
                            'standardized_abs_error':abs(yh-y)/scale,'standardized_squared_error':((yh-y)/scale)**2,
                            'template_squared_error':(mu-y)**2,'squared_error':(yh-y)**2,
                            'direction_error':float(np.sign(yh)!=np.sign(y)) if np.isfinite(yh) else np.nan,
                            'donor_deviation_direction_error':float(np.sign(yh-mu)!=np.sign(y-mu)) if np.isfinite(yh) else np.nan,
                            'model_lower':ml,'model_upper':mh,'lower':lo,'upper':hi,
                            'calibration_status':cstatus.get(f'{panel}/{method}',{}).get('status','not_calibrated')})
    frame=pd.DataFrame(rows); dest.mkdir(parents=True,exist_ok=True);frame.to_csv(dest/'effects.csv',index=False)
    summaries=[]
    for scope,sub in [('primary',frame.loc[frame.primary]),('all_hidden',frame)]:
        for (panel,method),g in sub.groupby(['panel','method'],sort=True):
            per=g.groupby('donor',sort=True).agg(mae=('standardized_abs_error','mean'),
                mse=('standardized_squared_error','mean'),direction=('direction_error','mean'),
                availability=('available','mean'))
            available=float(per.availability.mean());full=available==1.0
            cal_finite=np.isfinite(g.lower)&np.isfinite(g.upper)&g.available
            model_finite=np.isfinite(g.model_lower)&np.isfinite(g.model_upper)&g.available
            scored=interval_score(g.observed.to_numpy(),g.lower.to_numpy(),g.upper.to_numpy(),protocol['uncertainty_alpha'])
            summaries.append({'scope':scope,'panel':panel,'method':method,'donors':len(per),
                'point_availability':available,'standardized_mae_available':float(per.mae.mean()),
                'standardized_mae_full':float(per.mae.mean()) if full else None,
                'standardized_rmse':float(np.sqrt(per.mse.mean())),'direction_error':float(per.direction.mean()),
                'skill_vs_template':float(1-g.squared_error.sum()/g.template_squared_error.sum()) if full and g.template_squared_error.sum()>0 else None,
                'calibrated_interval_availability':float(cal_finite.mean()),
                'calibrated_coverage':float(((g.observed>=g.lower)&(g.observed<=g.upper))[cal_finite].mean()) if cal_finite.any() else None,
                'calibrated_mean_width':float((g.upper-g.lower)[cal_finite].mean()) if cal_finite.any() else None,
                'calibrated_interval_score':scored['mean'],
                'model_conditional_coverage':float(((g.observed>=g.model_lower)&(g.observed<=g.model_upper))[model_finite].mean()) if model_finite.any() else None,
                'model_conditional_width':float((g.model_upper-g.model_lower)[model_finite].mean()) if model_finite.any() else None})
    summary=pd.DataFrame(summaries);summary.to_csv(dest/'summary.csv',index=False)
    per_target=[]
    for (panel,method,protein),g in frame.groupby(['panel','method','protein'],sort=True):
        finite=g.available.all();denom=g.template_squared_error.sum()
        yy=g.observed-g.training_mean; pp=g.prediction-g.training_mean
        corr=float(np.corrcoef(yy,pp)[0,1]) if finite and yy.std()>0 and pp.std()>0 else None
        per_target.append({'panel':panel,'method':method,'protein':protein,'primary':bool(g.primary.iloc[0]),
            'standardized_mae':float(g.standardized_abs_error.mean()),'point_availability':float(g.available.mean()),
            'skill_vs_template':float(1-g.squared_error.sum()/denom) if finite and denom>0 else None,
            'donor_deviation_correlation':corr,
            'calibration_slope':float(np.cov(pp,yy,ddof=0)[0,1]/np.var(pp)) if finite and np.var(pp)>0 else None})
    pd.DataFrame(per_target).to_csv(dest/'per_target.csv',index=False)
    comparisons={};paired=[];primary=frame.loc[frame.primary]
    for panel in sorted(primary.panel.unique()):
        part=primary.loc[primary.panel==panel]
        # Every protected target must be present for a donor to enter a paired
        # comparison; groupby.mean alone would silently remove missing targets.
        donorloss=part.groupby(['donor','method']).standardized_abs_error.agg(
            lambda v:float(np.mean(v.to_numpy())) if len(v)==len(roster['primary']) and np.isfinite(v).all() else np.nan).unstack('method')
        independent=protocol['evaluation_mode']=='independent_test'
        result=paired_gain_bootstrap(donorloss['two_penalty_ridge'].to_numpy(),donorloss['abundance_response_shrinkage'].to_numpy(),
            donorloss.index.to_numpy(),n_bootstrap=10000,seed=protocol['seed'],independent_test=independent,
            models_frozen=independent,model_metadata={'implementation_lock':sha256(run/'implementation_lock.json'),
                'protocol_lock':sha256(run/'protocol.json')})
        comparisons[panel]=result
        for donor,row in donorloss.iterrows():
            paired.append({'panel':panel,'donor':donor,'new_loss':row['abundance_response_shrinkage'],
                'two_penalty_ridge_loss':row['two_penalty_ridge'],'mean_loss':row['mean'],'anchor_only_loss':row['anchor'],
                'difference_vs_two_penalty':row['two_penalty_ridge']-row['abundance_response_shrinkage']})
    pd.DataFrame(paired).to_csv(dest/'paired_donors.csv',index=False)
    write_json(dest/'paired_inference.json',clean(comparisons))
    primary_gain=comparisons['fixed_8'];ci=primary_gain.get('difference_ci95')
    main=summary.loc[(summary.scope=='primary')&(summary.panel=='fixed_8')&
                     (summary.method=='abundance_response_shrinkage')].iloc[0]
    full=bool(main.point_availability==1)
    skill=main.skill_vs_template
    skill_supported=bool(skill is not None and pd.notna(skill) and np.isfinite(float(skill)) and float(skill)>0)
    decision={'version':'0.4.0','evaluation_mode':protocol['evaluation_mode'],
        'primary_panel':'fixed_8','primary_baseline':'two_penalty_ridge','primary_targets':protocol['primary_targets'],'fixed_percentage_threshold':None,
        'primary_descriptive_gain':primary_gain['relative_gain'],
        'primary_full_denominator':full,'primary_skill_vs_template':skill,
        'independent_accuracy_supported':bool(full and skill_supported and ci is not None and ci[0]>0),
        'independent_calibration_executed':bool(protocol['evaluation_mode']=='independent_test' and main.calibrated_interval_availability==1),
        'cell_comparator_folds':cell_status,'publication_ready':False,
        'reason':('Development results cannot establish a replicated independent accuracy/reliability claim'
            if protocol['evaluation_mode']=='development_lodo' else
            'Independent test evidence requires interpretation of effect size, interval precision and replication; execution alone is insufficient'),
        'old_results_reclassified':False}
    write_json(dest/'decision.json',clean(decision));complete_stage(dest)


def diagnose_stage(run,protocol):
    """Post-evaluation descriptive diagnostics; never selects a model."""
    from scipy.stats import beta
    run=Path(run); require_receipt(run/'evaluation');dest=run/'diagnostics'
    if validate_receipt(dest):return
    effects=pd.read_csv(run/'evaluation/effects.csv')
    primary=effects.loc[effects.primary].copy();rows=[]
    for (panel,method),g in primary.groupby(['panel','method'],sort=True):
        total=g.donor.nunique(); finite_count=covered=0
        widths=[]
        for _,d in g.groupby('donor'):
            finite=len(d)==len(protocol['primary_targets']) and np.isfinite(d[['lower','upper','prediction']].to_numpy()).all()
            finite_count+=int(finite)
            covered+=int(finite and ((d.observed>=d.lower)&(d.observed<=d.upper)).all())
            if finite:widths.append(float(((d.upper-d.lower)/d.scale).mean()))
        # Descriptive exact binomial bounds only for a fixed independent test;
        # overlapping development folds do not meet that independence premise.
        independent=protocol['evaluation_mode']=='independent_test'
        lo=(0. if covered==0 else beta.ppf(.025,covered,total-covered+1)) if independent else None
        hi=(1. if covered==total else beta.ppf(.975,covered+1,total-covered)) if independent else None
        rows.append({'panel':panel,'method':method,'donors':total,'finite_joint_intervals':finite_count,
            'joint_covered_donors':covered,'full_roster_finite_joint_coverage':covered/total,
            'joint_coverage_finite_subset':covered/finite_count if finite_count else None,
            'mean_standardized_width_finite_subset':np.mean(widths) if widths else None,
            'coverage_ci95_low':lo,'coverage_ci95_high':hi,
            'coverage_ci_scope':'Independent fixed-model descriptive binomial; not subgroup guarantee' if independent else 'Unavailable for overlapping development folds'})
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'joint_reliability.csv',index=False)
    primary.to_csv(dest/'primary_donor_target.csv',index=False)
    # Component wins/losses are retained for every donor, with no exclusion.
    loss=primary.groupby(['panel','donor','method']).standardized_abs_error.agg(
        lambda x:float(x.mean()) if len(x)==len(protocol['primary_targets']) and np.isfinite(x).all() else np.nan).unstack('method')
    for name in ('two_penalty_ridge','joint','v030_response_shrinkage','v030_abundance_shrinkage','matched_response','no_anchor_correction','mean'):
        loss[f'gain_vs_{name}']=loss[name]-loss['abundance_response_shrinkage']
    loss.to_csv(dest/'paired_all_comparators.csv')
    selections=[]
    for fold in sorted(run.glob('fold_*')):
        selections.append({'fold':fold.name,'selection':json.loads((fold/'fit/selection.json').read_text())})
    write_json(dest/'selection_stability.json',clean(selections))
    write_json(dest/'claims.json',{'primary':json.loads((run/'evaluation/decision.json').read_text()),
        'causal_or_cell_intrinsic_claim':False,'independent_biological_replication':False,
        'technical_adjustment':'No isotype subtraction or hidden-ADT total normalization',
        'selection_on_diagnostics':False})
    complete_stage(dest,{'purpose':'All-donor error and calibration diagnostics; no post-hoc retuning'})


def report_stage(run,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);require_receipt(run/'evaluation');require_receipt(run/'explanation');dest=run/'report'
    if validate_receipt(dest):return dest/'report.md'
    dest.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(run/'evaluation/summary.csv');primary=summary.loc[summary.scope=='primary']
    paired=pd.read_csv(run/'evaluation/paired_donors.csv');decision=json.loads((run/'evaluation/decision.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    labels={'abundance_response_shrinkage':'ResponseBridge 0.4','two_penalty_ridge':'Two-penalty RNA + anchors ridge','joint':'Original joint ridge','v030_response_shrinkage':'ResponseBridge 0.3','v030_abundance_shrinkage':'Original abundance ablation','matched_response':'Matched response-objective ablation','mean':'Cohort mean',
        'rna':'RNA response ridge','anchor':'Anchor response ridge','residual':'Residual ridge',
        'no_anchor_correction':'Matched no-anchor ablation','legacy_responsebridge':'Previous architecture, fixed panel',
        'hard_rank':'Hard rank ablation','abundance_shrinkage':'Abundance RNA ablation',
        'sclinear_cell':'scLinear published core','scvaeit_cell_ensemble':'scVAEIT cell ensemble'}
    for ax,panel in zip(axes,['fixed_4','fixed_8']):
        ss=primary.loc[(primary.panel==panel)&primary.method.isin(labels)].copy()
        values=ss.standardized_mae_full.to_numpy(); xx=np.arange(len(ss))
        ax.bar(xx,values,color=['#d85c3d' if m=='abundance_response_shrinkage' else '#39769f' for m in ss.method])
        for j,v in enumerate(values):
            if not np.isfinite(v):ax.text(j,0,'unavailable',rotation=90,va='bottom',fontsize=8)
        ax.set_xticks(xx,[labels[m] for m in ss.method],rotation=45,ha='right',fontsize=8)
        ax.set_title(panel.replace('_',' ')+' / donor response MAE');ax.set_ylabel('Training-response-SD standardized MAE')
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=170);fig.savefig(dest/'comparison.pdf');plt.close(fig)
    e=pd.read_csv(run/'explanation/contributions.csv');e=e[(e.panel=='fixed_8')&e.target.isin(protocol['primary_targets'])]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,target in zip(axes,protocol['primary_targets']):
        table=e[e.target==target].pivot_table(index='donor',columns='component',values='contribution',aggfunc='sum')
        table.plot.bar(stacked=True,ax=ax,legend=False);ax.set_title(target+' exact additive components');ax.set_xlabel('Held-out donor');ax.set_ylabel('Response: mean cellwise log1p ADT change')
        ax.set_xticklabels([f'D{i+1}' for i in range(len(table))],rotation=0)
    handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(dest/'explanations.png',dpi=170);fig.savefig(dest/'explanations.pdf');plt.close(fig)
    fixed=paired[paired.panel=='fixed_8'];wins=int((fixed.difference_vs_two_penalty>0).sum())
    gain=decision['primary_descriptive_gain']
    gain_text='unavailable' if gain is None else f'{100*gain:.2f}%'
    development=protocol['evaluation_mode']=='development_lodo'
    evidence_text=('These previously inspected data are development evidence. No confidence interval treating overlapping cross-validation fits as independent is reported.'
        if development else 'This run uses frozen independent training, calibration and test donor roles. Paired bootstrap intervals condition on the fitted models and remain approximate.')
    calibration_text=('This development run has no independent calibration donors; absent calibrated intervals are not replaced by a guarantee.'
        if development else 'Calibrated intervals use the independent calibration donors. Insufficient or unavailable calibration scores retain uninformative intervals and the full test denominator.')
    lines=['# ResponseBridge 0.4: abundance-trained paired donor response recovery','',
        f"Version 0.4.0; mode **{protocol['evaluation_mode']}**. Primary panel: eight fixed measured proteins; primary targets: {', '.join(protocol['primary_targets'])}.",'',
        'The model predicts observed stimulation-minus-control responses from paired RNA and observed anchors. It does not forecast from baseline alone or prove a post-transcriptional mechanism.','',
        f"Primary relative MAE gain versus independently penalized RNA + anchor ridge: **{gain_text}**. Positive means lower error. Donor wins: **{wins}/{len(fixed)}**.",'',
        evidence_text+' Historical results and their previous project gate remain unchanged.','',
        '| Panel | Method | Full-cohort standardized MAE | Point availability | Skill versus cohort mean |',
        '|---|---|---:|---:|---:|']
    def fmt(v):return 'unavailable' if pd.isna(v) else f'{v:.4f}'
    for row in primary.itertuples(index=False):
        lines.append(f'| {row.panel} | {row.method} | {fmt(row.standardized_mae_full)} | {row.point_availability:.0%} | {fmt(row.skill_vs_template)} |')
    lines+=['','![Response error](comparison.png)','','## Reliability and explanations','',
        'Gaussian residual variances concern observed responses and already include sampling noise. They omit full parameter uncertainty and are not calibrated coverage guarantees. Donor-deletion coefficient variability is a sensitivity analysis.','',
        'Independent donor calibration, when supplied through frozen donor roles, uses a joint maximum-residual conformal score for the protected targets. '+calibration_text+' Every method uses the same fixed training response SDs for conformal scores; conditional variances and donor-deletion spreads remain separate diagnostics. All methods retain the same denominator.','',
        'The additive component sum exactly equals each prediction. Anchor coefficients and removal effects describe the fitted model; correlated markers prevent causal or unique-information interpretations.','',
        '![Exact contributions](explanations.png)','','## Comparator and scope','',
        f"Cell-level scVAEIT completed in {sum(x.get('scvaeit_cells')=='executed' for x in decision['cell_comparator_folds'])}/{len(decision['cell_comparator_folds'])} folds; scLinear completed in {sum(x.get('sclinear_cells')=='executed' for x in decision['cell_comparator_folds'])}/{len(decision['cell_comparator_folds'])} folds. Epochs are chosen only in donor-separated development folds; seed results and their ensemble are separately identified. Completion does not establish convergence.",'',
        'The previous architecture is also refitted on the same donor cohort and fixed panels: abundance-trained RNA with nested penalty selection, uncentered low-rank residuals and bootstrap refusal. Matched hard-rank, abundance-RNA and no-anchor ablations isolate individual changes. Historical v0.2.1 results remain separate. Unavailable low-rank predictions are retained.','',
        'Novelty rests on a useful donor-response/panel-extension application and replicated evidence, not on Gaussian conditioning or shrinkage as new mathematics. A fresh vaccine study is a separate stimulation/time-specific task and requires assay, metadata, resource and donor-role gates.','',
        'Machine-readable evidence: evaluation/effects.csv, summary.csv, per_target.csv, paired_donors.csv, paired_inference.json; explanation/contributions.csv, operators.csv and donor_deletion_stability.csv.','',
        '**Publication readiness requires independent accuracy, useful interval precision and biological replication; execution alone does not establish these claims.**']
    (dest/'report.md').write_text('\n'.join(lines)+'\n');complete_stage(dest)
    return dest/'report.md'
