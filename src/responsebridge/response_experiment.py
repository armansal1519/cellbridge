"""Version 0.3 response-native experiments with explicitly separated outcome vaults."""
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


def _cell_predictions(fold, role, donors, proteins):
    cell=Path(fold)/'scvaeit_cells'
    if not (cell/'receipt.json').exists():return {},None
    from .scvaeit_cells import validate_cell_receipt
    if not validate_cell_receipt(cell):raise ValueError('Invalid cell comparator receipt')
    if role=='calibration' and not len(donors):return {},None
    data=load_arrays(cell/('predictions.npz' if role=='test' else 'calibration_predictions.npz'))
    _same_donors(data['donor_ids'],donors,f'Cell comparator {role}')
    if not np.array_equal(data['protein_names'].astype(str),np.asarray(proteins,dtype=str)):
        raise ValueError('Cell comparator protein order mismatch')
    return data,{'fold':Path(fold).name,'status':'executed','selected_epochs':int(data['selected_epochs'])}


def _cell_methods(data,panel):
    if not data:return {}
    values=np.asarray(data[panel])
    expected=(len(data['seeds']),len(data['donor_ids']),len(data['protein_names']))
    if values.shape!=expected or len(np.unique(data['seeds']))!=len(data['seeds']):
        raise ValueError('Cell comparator predictions must be seeds by donors by proteins')
    result={f'scvaeit_cell_seed_{int(seed)}':values[i] for i,seed in enumerate(data['seeds'])}
    result['scvaeit_cell_ensemble']=values.mean(axis=0)
    return result


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


def initialize(groups, run, protocol, donor_split=None, limit=None):
    run=Path(run); dest=run/'cohort'
    require_receipt(groups)
    data,response,obs=cohort(groups,protocol)
    source={'groups':str(Path(groups).resolve()),'groups_sha256':sha256(Path(groups)/'groups.npz'),
        'metadata_sha256':sha256(Path(groups)/'groups.csv'),'cohort_lineage':protocol['cohort_lineage'],
        'primary_stimulation':protocol['primary_stimulation']}
    write_json(run/'input_lock.json',source,immutable=True)
    if not validate_receipt(dest):
        arrays(dest/'responses.npz',**response); obs.to_csv(dest/'responses.csv',index=False)
        complete_stage(dest,{'estimand':protocol['estimand'],'source':source})
    panels={f'fixed_{size}':index_names(response['proteins'],protocol[f'fixed_panel_{size}']) for size in [4,8]}
    target=index_names(response['proteins'],protocol['primary_targets'])
    if any(np.intersect1d(target,a).size for a in panels.values()): raise ValueError('Protected target is an anchor')
    secondary=np.setdiff1d(np.arange(len(response['proteins'])), panels['fixed_8'])
    write_json(run/'target_roster.json',{'proteins':response['proteins'].tolist(),'primary':target.tolist(),
        'secondary':secondary.tolist(),'policy':'All biological proteins outside fixed8, including primary sentinels; no outcome ranking'},immutable=True)
    ids=response['donors']; folds=[]
    if donor_split:
        from .reliability import validate_donor_splits
        split=json.loads(Path(donor_split).read_text())
        split=split.get('splits',split)
        if 'fit' in split and 'train' not in split:split={**split,'train':split['fit']}
        role={key:[str(d) for d in split[key]] for key in ['train','calibration','test']}
        if any(len(v)!=len(set(v)) for v in role.values()):raise ValueError('Donor roles must not contain duplicate identities')
        validate_donor_splits(role['train'],role['calibration'],role['test'])
        if set(sum(role.values(),[])) != set(ids): raise ValueError('Every eligible donor must have exactly one external role')
        if protocol['evaluation_mode']!='independent_test': raise ValueError('External roles require an independent-test protocol')
        if len(role['train'])<8 or len(role['calibration'])<9 or len(role['test'])<10:
            raise ValueError('External allocation does not meet frozen training/calibration/test minima')
        folds=[{'train':np.flatnonzero(np.isin(ids,role['train'])).tolist(),
            'calibration':np.flatnonzero(np.isin(ids,role['calibration'])).tolist(),
            'test':np.flatnonzero(np.isin(ids,role['test'])).tolist()}]
    else:
        if protocol['evaluation_mode']!='development_lodo': raise ValueError('Independent test requires frozen donor roles')
        folds=[{'train':np.flatnonzero(ids!=d).tolist(),'calibration':[],
                'test':np.flatnonzero(ids==d).tolist()} for d in ids]
    if limit is not None:
        if limit<1: raise ValueError('Positive pilot fold limit required')
        folds=folds[:limit]
    write_json(run/'splits.json',{'mode':protocol['evaluation_mode'],'donors':ids.tolist(),
        'folds':folds,'pilot_fold_limit':limit,'previous_exposure':protocol['previous_exposure']},immutable=True)
    for i,split in enumerate(folds):
        fold=run/f'fold_{i:02d}'; tr=np.asarray(split['train'],int)
        write_json(fold/'roles.json',{k:ids[v].tolist() for k,v in split.items()},immutable=True)
        train=fold/'development'
        if not validate_receipt(train):
            arrays(train/'responses.npz',**{k:(v[tr] if k in ['x','y','variance','donors'] else v) for k,v in response.items()})
            group_idx=np.flatnonzero(data.obs.donor.astype(str).isin(ids[tr]).to_numpy())
            gd=data.subset(group_idx)
            arrays(train/'abundance.npz',x=gd.x,y=gd.y,y_var=gd.y_var,genes=gd.genes,proteins=gd.proteins,donors=gd.obs.donor.astype(str).to_numpy(),
                weights=abundance_group_weights(gd.obs,'condition_balanced'))
            gd.obs.to_csv(train/'abundance.csv',index=False)
            complete_stage(train)
        for role in ['test','calibration']:
            chosen=np.asarray(split[role],int); out=fold/'outcomes'/role
            if not validate_receipt(out):
                arrays(out/'truth.npz', y=response['y'][chosen],donors=ids[chosen],variance=response['variance'][chosen])
                # Privileged masking step: target columns never enter query artifacts.
                query=fold/'queries'/role
                for name,a in panels.items():
                    arrays(query/f'{name}.npz',x=response['x'][chosen],anchor_values=response['y'][chosen][:,a],
                        anchor_indices=a,donors=ids[chosen])
                complete_stage(query,{'hidden_targets_present':False,'role':role})
                complete_stage(out,{'access':'evaluation/calibration only'})
        write_json(fold/'panels.json',{k:v.tolist() for k,v in panels.items()},immutable=True)
    return folds


def _abundance_ablation(base, train, abundance, guard=None):
    """Hold alpha/shrinkage/panel fixed; change only the RNA training objective."""
    x,y,donors=train['x'],train['y'],train['donors']
    ad=abundance['donors']; alpha=base.alpha
    def fit_for(mask):
        if alpha is None: return None
        weights=abundance['weights'][mask].copy()
        # For donor LODO, slicing equal donor/condition weights preserves each
        # retained donor's condition masses; normalize in RNARegressor.fit.
        m=RNARegressor(alpha,base.n_hvg).fit(abundance['x'][mask],abundance['y'][mask],weights)
        m.x_mean=np.zeros_like(m.x_mean); m.y_mean=np.zeros_like(m.y_mean); m.intercept=np.zeros_like(m.intercept)
        return m
    oof=np.zeros_like(y)
    for d in sorted(set(donors)):
        if guard: guard()
        m=fit_for(ad!=d)
        if m is not None: oof[donors==d]=m.predict(x[donors==d])
    final=fit_for(np.ones(len(ad),bool))
    # Mean-only abundance predicts a zero difference, represented by a constant
    # linear model rather than the donor-response mean used by the direct fit.
    if final is None:
        final=RNARegressor(1,base.n_hvg).fit(np.zeros_like(x),np.zeros_like(y))
    provenance={'ablation':'abundance_RNA_same_alpha_shrinkage_panel',
        'matched_response_alpha':alpha,'abundance_rows':len(ad),'abundance_donors':sorted(set(ad)),
        'mean_only_representation':'Constant zero response mapping is persisted as an explicit fitted affine model',
        'residual_folds':[{'validation_donor':str(d),'training_donors':sorted(set(ad)-{d})} for d in sorted(set(donors))]}
    return ResponseShrinkage.from_components(x,y,donors,rna_model=final,oof_residuals=y-oof,
        shrinkage=base.shrinkage,provenance=provenance)


def fit_stage(run, protocol, guard=None):
    from .model import ResponseBasis, ResponseBridge, ObservableInput
    from .experiment import save_basis
    from .legacy_response import fit_legacy
    run=Path(run); roster=json.loads((run/'target_roster.json').read_text()); targets=np.array(roster['primary'])
    for fold in sorted(run.glob('fold_*')):
        dest=fold/'fit'; require_receipt(fold/'development')
        if validate_receipt(dest):
            fit_legacy(fold,protocol,guard)
            continue
        dest.mkdir(parents=True,exist_ok=True)
        tr=load_arrays(fold/'development/responses.npz'); x,y,donors=tr['x'],tr['y'],tr['donors']
        panels=json.loads((fold/'panels.json').read_text()); a4=np.array(panels['fixed_4'])
        model, selection=select_shrinkage(x,y,donors,a4,targets,alphas=protocol['ridge_alphas'],
            shrinkages=protocol['shrinkages'],n_hvg=protocol['rna_hvg'],guard=guard)
        model.save(dest/'response_shrinkage.npz'); write_json(dest/'selection.json',selection)
        abundance=_abundance_ablation(model,tr,load_arrays(fold/'development/abundance.npz'),guard)
        abundance.save(dest/'abundance_shrinkage.npz')
        baseline_meta={}; scales=response_scale(y)
        arrays(dest/'reference.npz',cohort_mean=y.mean(axis=0),scales=scales,proteins=tr['proteins'],donors=donors)
        # Independent baseline tuning shares the actual primary donor splits.
        for panel,anchor_list in panels.items():
            a=np.array(anchor_list,dtype=int); details={}; residual_scores=[]
            for kind in ['mean','rna','anchor','joint']:
                bm, info, oof=tune_linear(x,y,donors,y[:,a],targets,kind,protocol['ridge_alphas'],
                    n_hvg=protocol['rna_hvg'],guard=guard)
                bm.save(dest/f'{panel}_{kind}.npz'); details[kind]=info
                arrays(dest/f'{panel}_{kind}_development.npz',oof=oof,
                    error_sd=np.maximum(np.sqrt(np.mean((oof-y)**2,axis=0)),1e-6))
            # Residual ridge gets its own joint RNA/anchor penalty search, with
            # RNA residuals rebuilt wholly inside every inner training split.
            rank_scores=[]
            for alpha in protocol['ridge_alphas']:
                folds=[]
                for donor in sorted(set(donors)):
                    if guard: guard()
                    ii=donors!=donor; vv=~ii
                    cm=ResponseShrinkage().fit(x[ii],y[ii],donors[ii],alpha=alpha,
                        shrinkage=1,n_hvg=protocol['rna_hvg'],guard=guard)
                    res=_aligned_residuals(cm,donors[ii])
                    fval=cm.predict_rna(x[vv]); va=y[vv][:,a]-fval[:,a]
                    folds.append((cm,res,fval,va,ii,vv))
                for gamma in protocol['ridge_alphas']:
                    losses=[]
                    for cm,res,fval,va,ii,vv in folds:
                        rr=LinearResponse('residual',gamma,protocol['rna_hvg']).fit(x[ii],res,res[:,a])
                        pred=fval+rr.predict(x[vv],va)
                        losses.append(float(np.mean(np.abs((pred-y[vv])/response_scale(y[ii]))[:,targets])))
                    residual_scores.append({'rna_alpha':alpha,'anchor_alpha':gamma,'mae':float(np.mean(losses))})
                if alpha==model.alpha:
                    for rank in protocol['ablation_ranks']:
                        losses=[]
                        for cm,res,fval,va,ii,vv in folds:
                            try:
                                b=ResponseBasis.fit(res,rank,scales=response_scale(y[ii]))
                                pp=ResponseBridge(b).predict(ObservableInput(fval,va,a)).prediction
                                loss=float(np.mean(np.abs((pp-y[vv])/response_scale(y[ii]))[:,targets]))
                            except ValueError: loss=float('inf')
                            losses.append(loss)
                        rank_scores.append({'rank':rank,'mae':float(np.mean(losses))})
            chosen=min(residual_scores,key=lambda v:v['mae'])
            rm=ResponseShrinkage().fit(x,y,donors,alpha=chosen['rna_alpha'],shrinkage=1,n_hvg=protocol['rna_hvg'],guard=guard)
            rm.save(dest/f'{panel}_residual_rna.npz')
            aligned=_aligned_residuals(rm,donors)
            rr=LinearResponse('residual',chosen['anchor_alpha'],protocol['rna_hvg']).fit(x,aligned,aligned[:,a])
            rr.save(dest/f'{panel}_residual.npz')
            details['residual']={'selected':chosen,'scores':residual_scores}
            valid=[s for s in rank_scores if np.isfinite(s['mae'])]
            if valid:
                rank=min(valid,key=lambda v:(v['mae'],v['rank']))['rank']
                basis=ResponseBasis.fit(model.oof_residuals,rank,scales=scales)
                save_basis(basis,dest/f'{panel}_hard_rank.npz')
                details['hard_rank']={'selected_rank':rank,'scores':clean(rank_scores)}
            else: details['hard_rank']={'status':'no_identified_development_rank','scores':clean(rank_scores)}
            baseline_meta[panel]=details
        write_json(dest/'baselines.json',baseline_meta)
        write_json(dest/'uncertainty_contract.json',{'model':'Observed residual covariance; no added ADT sampling variance',
            'development':'Outer development LODO errors are descriptive, not independent conformal calibration',
            'old_version':'Original results preserved; fixed-alpha hard-rank and abundance-objective ablations match current data/panels'})
        fit_legacy(fold,protocol,guard)
        complete_stage(dest,{'training_donors':donors.tolist(),'model':'response-shrinkage'})


def predict_stage(run, protocol, guard=None):
    from .model import ResponseBridge, ObservableInput
    from .experiment import load_basis
    from .legacy_response import predict_legacy
    for fold in sorted(Path(run).glob('fold_*')):
        require_receipt(fold/'fit'); require_receipt(fold/'legacy_fit');dest=fold/'predictions'
        for role in ['test','calibration']:require_receipt(fold/'queries'/role)
        if validate_receipt(dest): continue
        model=ResponseShrinkage.load(fold/'fit/response_shrinkage.npz')
        abundance=ResponseShrinkage.load(fold/'fit/abundance_shrinkage.npz')
        panels=json.loads((fold/'panels.json').read_text())
        for role in ['test','calibration']:
            require_receipt(fold/'queries'/role)
            for panel in panels:
                if guard: guard()
                q=load_arrays(fold/'queries'/role/f'{panel}.npz'); x,anchors,a=q['x'],q['anchor_values'],q['anchor_indices']
                result=model.predict(x,anchors,a); pred={'response_shrinkage':result['prediction']}
                sd={'response_shrinkage':np.sqrt(np.maximum(result['model_variance'],1e-12))}
                no_anchor=model.predict(x,np.empty((len(x),0)),np.array([],dtype=int))
                pred['no_anchor_correction']=no_anchor['prediction']
                sd['no_anchor_correction']=np.sqrt(np.maximum(no_anchor['model_variance'],1e-12))
                pred['legacy_responsebridge'],sd['legacy_responsebridge']=predict_legacy(fold,panel,q)
                for kind in ['mean','rna','anchor','joint']:
                    bm=LinearResponse.load(fold/'fit'/f'{panel}_{kind}.npz')
                    pred[kind]=bm.predict(x,anchors)
                    sd[kind]=load_arrays(fold/'fit'/f'{panel}_{kind}_development.npz')['error_sd']
                rm=ResponseShrinkage.load(fold/'fit'/f'{panel}_residual_rna.npz')
                rr=LinearResponse.load(fold/'fit'/f'{panel}_residual.npz'); rp=rm.predict_rna(x)
                pred['residual']=rp+rr.predict(x,anchors-rp[:,a])
                sd['residual']=np.sqrt(np.maximum(np.diag(rm.covariance),1e-12))
                ar=abundance.predict(x,anchors,a); pred['abundance_shrinkage']=ar['prediction']
                sd['abundance_shrinkage']=np.sqrt(np.maximum(ar['model_variance'],1e-12))
                basis_path=fold/'fit'/f'{panel}_hard_rank.npz'
                pred['hard_rank']=(ResponseBridge(load_basis(basis_path)).predict(
                    ObservableInput(model.predict_rna(x), anchors-model.predict_rna(x)[:,a],a)).prediction
                    if basis_path.exists() else np.full_like(result['prediction'],np.nan))
                sd['hard_rank']=np.sqrt(np.maximum(np.diag(model.covariance),1e-12))
                removed=[]
                for j in range(len(a)):
                    keep=np.arange(len(a))!=j
                    removed.append(model.predict(x,anchors[:,keep],a[keep])['prediction'])
                arrays(dest/role/f'{panel}.npz',**pred,**{f'sd__{k}':v for k,v in sd.items()},
                    donors=q['donors'],anchor_indices=a,base_response=result['base_response'],
                    rna_contribution=result['rna_contribution'],anchor_contributions=result['anchor_contributions'],
                    anchor_operator=result['anchor_operator'],removed_anchor_predictions=np.stack(removed,axis=-1))
        complete_stage(dest,{'query_contract':'Only x,anchor_values,anchor_indices,donors; outcome vault never read'})


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
            'cell_receipt':sha256(fold/'scvaeit_cells/receipt.json') if neural else None,
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
                    cal_sd=reference['scales'] if is_neural else c[f'sd__{method}']
                    cal=fit_joint_calibration(truth['y'][:,targets],cal_methods[method][:,targets],
                        cal_sd[...,targets],truth['donors'],names,protocol['uncertainty_alpha'])
                    applied=apply_calibration(pred[:,targets],sd[...,targets],cal,names)
                    lo=np.full_like(pred,np.nan); hi=np.full_like(pred,np.nan)
                    lo[:,targets]=applied['lower']; hi[:,targets]=applied['upper']
                    output[f'lower__{method}']=lo;output[f'upper__{method}']=hi
                    cal['score_scale']='fixed_training_response_sd' if is_neural else 'fixed_training_model_or_development_error_sd'
                    records[f'{panel}/{method}']=clean(cal)
                else:
                    output[f'lower__{method}']=np.full_like(pred,np.nan)
                    output[f'upper__{method}']=np.full_like(pred,np.nan)
                    records[f'{panel}/{method}']={'status':'no_independent_calibration_donors',
                        'guarantee':None,'model_interval_only':not is_neural,
                        'score_scale':'fixed_training_response_sd' if is_neural else 'fixed_training_model_or_development_error_sd'}
            arrays(dest/f'{panel}.npz',**output)
        write_json(dest/'status.json',records); complete_stage(dest)


def explain_stage(run,protocol):
    run=Path(run); roster=json.loads((run/'target_roster.json').read_text()); rows=[]; operator_rows=[]
    dest=run/'explanation'
    if validate_receipt(dest): return
    for fold in sorted(run.glob('fold_*')):
        require_receipt(fold/'predictions')
        for panel in json.loads((fold/'panels.json').read_text()):
            pred=load_arrays(fold/'predictions/test'/f'{panel}.npz')
            check=pred['base_response']+pred['rna_contribution']+pred['anchor_contributions'].sum(-1)
            if not np.allclose(check,pred['response_shrinkage'],rtol=1e-11,atol=1e-11): raise ValueError('Explanation sum mismatch')
            for p in roster['secondary']:
                for d,donor in enumerate(pred['donors']):
                    rows.append({'fold':fold.name,'panel':panel,'donor':donor,'target':roster['proteins'][p],
                        'component':'base_response','contribution':pred['base_response'][d,p]})
                    rows.append({'fold':fold.name,'panel':panel,'donor':donor,'target':roster['proteins'][p],
                        'component':'RNA','contribution':pred['rna_contribution'][d,p]})
                    for j,a in enumerate(pred['anchor_indices']):
                        rows.append({'fold':fold.name,'panel':panel,'donor':donor,'target':roster['proteins'][p],
                            'component':roster['proteins'][a],'contribution':pred['anchor_contributions'][d,p,j],
                            'removal_change':pred['removed_anchor_predictions'][d,p,j]-pred['response_shrinkage'][d,p]})
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
        result=paired_gain_bootstrap(donorloss['joint'].to_numpy(),donorloss['response_shrinkage'].to_numpy(),
            donorloss.index.to_numpy(),n_bootstrap=10000,seed=protocol['seed'],independent_test=independent,
            models_frozen=independent,model_metadata={'implementation_lock':sha256(run/'implementation_lock.json'),
                'protocol_lock':sha256(run/'protocol.json')})
        comparisons[panel]=result
        for donor,row in donorloss.iterrows():
            paired.append({'panel':panel,'donor':donor,'new_loss':row['response_shrinkage'],
                'joint_ridge_loss':row['joint'],'mean_loss':row['mean'],'anchor_only_loss':row['anchor'],
                'difference_vs_joint':row['joint']-row['response_shrinkage']})
    pd.DataFrame(paired).to_csv(dest/'paired_donors.csv',index=False)
    write_json(dest/'paired_inference.json',clean(comparisons))
    primary_gain=comparisons['fixed_4'];ci=primary_gain.get('difference_ci95')
    decision={'version':'0.3.0','evaluation_mode':protocol['evaluation_mode'],
        'primary_panel':'fixed_4','primary_targets':protocol['primary_targets'],'fixed_percentage_threshold':None,
        'primary_descriptive_gain':primary_gain['relative_gain'],
        'independent_accuracy_supported':bool(ci is not None and ci[0]>0),
        'independent_calibration_executed':protocol['evaluation_mode']=='independent_test',
        'cell_comparator_folds':cell_status,'publication_ready':False,
        'reason':('Development results cannot establish a replicated independent accuracy/reliability claim'
            if protocol['evaluation_mode']=='development_lodo' else
            'Independent test evidence requires interpretation of effect size, interval precision and replication; execution alone is insufficient'),
        'old_results_reclassified':False}
    write_json(dest/'decision.json',clean(decision));complete_stage(dest)


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
    labels={'response_shrinkage':'ResponseBridge 2','joint':'RNA + anchors ridge','mean':'Cohort mean',
        'rna':'RNA response ridge','anchor':'Anchor response ridge','residual':'Residual ridge',
        'no_anchor_correction':'Matched no-anchor ablation','legacy_responsebridge':'Previous architecture, fixed panel',
        'hard_rank':'Hard rank ablation','abundance_shrinkage':'Abundance RNA ablation','scvaeit_cell_ensemble':'scVAEIT cell ensemble'}
    for ax,panel in zip(axes,['fixed_4','fixed_8']):
        ss=primary.loc[(primary.panel==panel)&primary.method.isin(labels)].copy()
        values=ss.standardized_mae_full.to_numpy(); xx=np.arange(len(ss))
        ax.bar(xx,values,color=['#d85c3d' if m=='response_shrinkage' else '#39769f' for m in ss.method])
        for j,v in enumerate(values):
            if not np.isfinite(v):ax.text(j,0,'unavailable',rotation=90,va='bottom',fontsize=8)
        ax.set_xticks(xx,[labels[m] for m in ss.method],rotation=45,ha='right',fontsize=8)
        ax.set_title(panel.replace('_',' ')+' / donor response MAE');ax.set_ylabel('Training-response-SD standardized MAE')
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=170);fig.savefig(dest/'comparison.pdf');plt.close(fig)
    e=pd.read_csv(run/'explanation/contributions.csv');e=e[(e.panel=='fixed_4')&e.target.isin(protocol['primary_targets'])]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,target in zip(axes,protocol['primary_targets']):
        table=e[e.target==target].pivot_table(index='donor',columns='component',values='contribution',aggfunc='sum')
        table.plot.bar(stacked=True,ax=ax,legend=False);ax.set_title(target+' exact additive components');ax.set_xlabel('Held-out donor');ax.set_ylabel('Response: mean cellwise log1p ADT change')
        ax.set_xticklabels([f'D{i+1}' for i in range(len(table))],rotation=0)
    handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(dest/'explanations.png',dpi=170);fig.savefig(dest/'explanations.pdf');plt.close(fig)
    fixed=paired[paired.panel=='fixed_4'];wins=int((fixed.difference_vs_joint>0).sum())
    gain=decision['primary_descriptive_gain']
    gain_text='unavailable' if gain is None else f'{100*gain:.2f}%'
    development=protocol['evaluation_mode']=='development_lodo'
    evidence_text=('These previously inspected data are development evidence. No confidence interval treating overlapping cross-validation fits as independent is reported.'
        if development else 'This run uses frozen independent training, calibration and test donor roles. Paired bootstrap intervals condition on the fitted models and remain approximate.')
    calibration_text=('This development run has no independent calibration donors; absent calibrated intervals are not replaced by a guarantee.'
        if development else 'Calibrated intervals use the independent calibration donors. Insufficient or unavailable calibration scores retain uninformative intervals and the full test denominator.')
    lines=['# ResponseBridge 2: response-native donor validation','',
        f"Version 0.3.0; mode **{protocol['evaluation_mode']}**. Primary panel: four fixed measured proteins; primary targets: {', '.join(protocol['primary_targets'])}.",'',
        'The model predicts observed stimulation-minus-control responses from paired RNA and observed anchors. It does not forecast from baseline alone or prove a post-transcriptional mechanism.','',
        f"Primary relative MAE gain versus direct RNA + anchor response ridge: **{gain_text}**. Positive means lower error. Donor wins: **{wins}/{len(fixed)}**.",'',
        evidence_text+' Historical results and their previous project gate remain unchanged.','',
        '| Panel | Method | Full-cohort standardized MAE | Point availability | Skill versus cohort mean |',
        '|---|---|---:|---:|---:|']
    def fmt(v):return 'unavailable' if pd.isna(v) else f'{v:.4f}'
    for row in primary.itertuples(index=False):
        lines.append(f'| {row.panel} | {row.method} | {fmt(row.standardized_mae_full)} | {row.point_availability:.0%} | {fmt(row.skill_vs_template)} |')
    lines+=['','![Response error](comparison.png)','','## Reliability and explanations','',
        'Gaussian residual variances concern observed responses and already include sampling noise. They omit full parameter uncertainty and are not calibrated coverage guarantees. Donor-deletion coefficient variability is a sensitivity analysis.','',
        'Independent donor calibration, when supplied through frozen donor roles, uses a joint maximum-residual conformal score for the protected targets. '+calibration_text+' Neural score normalization uses fixed training response SDs; these are not neural posterior variances. All methods retain the same denominator.','',
        'The additive component sum exactly equals each prediction. Anchor coefficients and removal effects describe the fitted model; correlated markers prevent causal or unique-information interpretations.','',
        '![Exact contributions](explanations.png)','','## Comparator and scope','',
        f"Cell-level scVAEIT completed in {sum(x['status']=='executed' for x in decision['cell_comparator_folds'])}/{len(decision['cell_comparator_folds'])} folds. Epochs are chosen only in donor-separated development folds; seed results and their ensemble are separately identified.",'',
        'The previous architecture is also refitted on the same donor cohort and fixed panels: abundance-trained RNA with nested penalty selection, uncentered low-rank residuals and bootstrap refusal. Matched hard-rank, abundance-RNA and no-anchor ablations isolate individual changes. Historical v0.2.1 results remain separate. Unavailable low-rank predictions are retained.','',
        'Novelty rests on a useful donor-response/panel-extension application and replicated evidence, not on Gaussian conditioning or shrinkage as new mathematics. A fresh vaccine study is a separate stimulation/time-specific task and requires assay, metadata, resource and donor-role gates.','',
        'Machine-readable evidence: evaluation/effects.csv, summary.csv, per_target.csv, paired_donors.csv, paired_inference.json; explanation/contributions.csv, operators.csv and donor_deletion_stability.csv.','',
        '**Publication readiness requires independent accuracy, useful interval precision and biological replication; execution alone does not establish these claims.**']
    (dest/'report.md').write_text('\n'.join(lines)+'\n');complete_stage(dest)
    return dest/'report.md'
