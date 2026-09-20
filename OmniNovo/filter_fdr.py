#!/usr/bin/env python3
"""One-command PSM/composition FDR and all-modification PTMProphet FLR filtering."""
import argparse
import csv
import glob
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'fdr'))
import numpy as np
import ptmprophet_adapter as adapter
from sequence_schema import parse_prediction, OMNI_TOKENS
from per_seed_engine import single_seed_curve, map_target_qvalues, single_seed_composition_competition
from io_utils import sha256, write_tsv


def read_predictions(patterns, registry, origin):
    files=sorted({Path(p).resolve() for pattern in patterns for p in glob.glob(pattern)})
    if not files:
        raise ValueError(f'no {origin} prediction TSV files matched')
    title_key='title' if origin=='target' else 'decoy_title'
    expected={row[title_key]:row for row in registry}
    found={}; fields=[]
    for path in files:
        with path.open(newline='',encoding='utf-8') as handle:
            reader=csv.DictReader(handle,delimiter='\t')
            if not {'title','prediction','confidence_score'} <= set(reader.fieldnames or []):
                raise ValueError(f'{path}: required columns title/prediction/confidence_score missing')
            fields=list(dict.fromkeys([*fields,*reader.fieldnames]))
            for raw in reader:
                title=raw['title']
                if title not in expected:
                    raise ValueError(f'{path}: unknown {origin} spectrum {title!r}; use the prepared dataset')
                if title in found:
                    raise ValueError(f'duplicate {origin} prediction: {title}')
                score=float(raw['confidence_score'])
                if not math.isfinite(score):
                    raise ValueError(f'non-finite score for {title}')
                row=expected[title]
                for col in ('precursor_mz','precursor_charge'):
                    if col in raw and raw[col]:
                        tol=0.002 if col=='precursor_mz' else 0
                        if abs(float(raw[col])-float(row[col])) > tol:
                            raise ValueError(f'{title}: {col} does not match the prepared spectrum')
                parsed=parse_prediction('OmniNovo',raw['prediction'])
                found[title]={**raw,'spectrum_index':int(row['spectrum_index']),'spectrum_id':row['title'],
                    'source_title':row['source_title'],'source_file':row['source_file'],'source_index':int(row['source_index']),
                    'precursor_mz':float(row['precursor_mz']),'precursor_charge':int(row['precursor_charge']),
                    'native_score':score,'score':score,'model':'OmniNovo','parsed_prediction':parsed,
                    'composition_key':parsed.composition_key_il_equivalent,'is_empty':parsed.is_empty}
    missing=set(expected)-set(found)
    if missing:
        raise ValueError(f'{origin} predictions incomplete: {len(missing)} missing; include every rank TSV (including empty predictions)')
    return sorted(found.values(),key=lambda row:row['spectrum_index']),files,fields


def curve_rows(curve):
    return [{'threshold':float(t),'targets':int(nt),'decoys':int(nd[0]),'estimated_fdr':float(e),'qvalue':float(q)}
            for t,nt,nd,e,q in zip(curve.threshold,curve.target_count,curve.decoy_count_by_seed,curve.plus_one_mean,curve.qvalue)]


def localization_qvalues(rows):
    """Group score ties and use cumulative mean(1-MBPr)."""
    groups={}
    for row in rows:
        p=float(row['mbpr'])
        if not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('invalid localization MBPr')
        groups.setdefault(p,[]).append(row)
    curve=[]; errors=0.; count=0
    for p,group in sorted(groups.items(),reverse=True):
        count+=len(group); errors+=(1-p)*len(group)
        curve.append({'threshold_mbpr':p,'psms':count,'expected_errors':errors,'estimated_flr':errors/count})
    minimum=1.
    for point in reversed(curve):
        minimum=min(minimum,point['estimated_flr']); point['q_loc']=minimum
        for row in groups[point['threshold_mbpr']]: row['q_loc']=minimum
    return curve


def run(args):
    data=args.dataset.resolve(); manifest=json.loads((data/'dataset.json').read_text())
    for name in ('target.mgf','spectrum_registry.tsv'):
        if sha256(data/name)!=manifest['files'][name]['sha256']:
            raise ValueError(f'prepared input changed: {name}')
    with (data/'spectrum_registry.tsv').open(newline='') as handle:
        registry=list(csv.DictReader(handle,delimiter='\t'))
    if len(registry)!=manifest['n_spectra'] or len({r['title'] for r in registry})!=len(registry):
        raise ValueError('invalid spectrum registry')
    target,tf,fields=read_predictions(args.target,registry,'target')
    decoy,df,_=read_predictions(args.decoy,registry,'decoy')
    targets=tuple(adapter.LocalizationTarget(row['name'],row['residues'],float(row['delta_mass'])) for row in json.loads(args.modifications.read_text()))
    adapter.validate_targets(targets)
    if {t.name for t in targets}!=set(OMNI_TOKENS.values()):
        raise ValueError('modification configuration must cover every OmniNovo modification token exactly once')
    all_names={t.name for t in targets}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    tvalid=[r for r in target if not r['is_empty']]; dvalid=[r for r in decoy if not r['is_empty']]
    psm_curve=single_seed_curve([r['score'] for r in tvalid],[r['score'] for r in dvalid])
    for r,q in zip(tvalid,map_target_qvalues([r['score'] for r in tvalid],psm_curve,'plus_one')): r['q_psm']=float(q)
    competition=single_seed_composition_competition(tvalid,dvalid,str(manifest['seed']))
    tw=competition['target_winners'];dw=competition['decoy_winners_by_seed'][str(manifest['seed'])]
    peptide_curve=single_seed_curve([r['score'] for r in tw],[r['score'] for r in dw])
    cq=dict(zip([r['composition_key'] for r in tw],map_target_qvalues([r['score'] for r in tw],peptide_curve,'plus_one')))
    eligible=[]
    for row in target:
        row.update(q_peptide=float(cq.get(row['composition_key'],1.)),q_loc=None,accepted=False,
                   localization_status='not_run',localization_reason='')
        row.setdefault('q_psm',1.)
        if row['is_empty']:
            row['localization_reason']='empty_prediction'
        elif row['q_psm']>args.psm_fdr or row['q_peptide']>args.peptide_fdr or row['composition_key'] not in cq:
            row['localization_reason']='failed_identification_gates'
        elif args.identification_only:
            row.update(accepted=True,localization_status='not_requested')
        elif not row['parsed_prediction'].modifications:
            row.update(accepted=True,localization_status='unmodified')
        else:
            parsed=row['parsed_prediction']
            if any(m.name not in all_names for m in parsed.modifications):
                raise ValueError('unhandled modification')
            reason=adapter.localization_exclusion_reason(parsed,targets)
            if reason:
                row.update(localization_status='incompatible',localization_reason=reason)
            else: eligible.append(row)
    site_rows=[]; loc_curve=[]
    if eligible:
        binary=shutil.which(str(args.ptmprophet))
        if not binary: raise ValueError('PTMProphetParser not found; set --ptmprophet /absolute/path/PTMProphetParser')
        adapter.PTMPROPHET=Path(binary).resolve();adapter.PYOPENMS_PYTHON=args.pyopenms_python.resolve()
        options=(*adapter.with_targets(adapter.PTMPROPHET_EM0_OPTIONS,targets),'MODPREC=4')
        options=tuple(f'MAXTHREADS={args.workers}' if o.startswith('MAXTHREADS=') else o for o in options)
        workers=min(args.workers,len(eligible))
        if workers>=2:
            result=adapter.run_ptmprophet_sharded_direct(eligible,data/'target.mgf',out/'ptmprophet',shards=workers,options=options,targets=targets)
            site_rows=result['parsed_rows']
        else:
            result=adapter.run_ptmprophet(eligible,data/'target.mgf',out/'ptmprophet',options=options,targets=targets)
            site_rows=adapter.parse_ptmprophet_output(result['output_pepxml'],result['mapping'],targets)
        if len({r['spectrum_index'] for r in site_rows})!=len(eligible):
            raise ValueError('localization coverage mismatch')
        loc_curve=localization_qvalues(site_rows)
        by_index={r['spectrum_index']:r for r in site_rows}
        for row in eligible:
            localized=by_index[row['spectrum_index']]
            for name in ('mbpr','mbpr_by_target','target_counts','site_positions_by_target','site_probabilities_by_target','assigned_target_sites','localized_site_count','expected_site_errors','q_loc'):
                row[name]=localized[name]
            row.update(localization_status='localized',accepted=row['q_loc']<=args.localization_flr)
            if not row['accepted']: row['localization_reason']='failed_localization_gate'
    extra=['spectrum_index','source_file','source_index','source_title','composition_key','q_psm','q_peptide','q_loc',
           'localization_status','localization_reason','mbpr','mbpr_by_target','target_counts','site_positions_by_target',
           'site_probabilities_by_target','assigned_target_sites','localized_site_count','expected_site_errors','accepted']
    fields=list(dict.fromkeys([*fields,*extra]))
    write_tsv(out/'all_psms.tsv',target,fields)
    accepted=[r for r in target if r['accepted']]
    write_tsv(out/'filtered_psms.tsv',accepted,fields)
    representatives={}
    for row in sorted(accepted,key=lambda r:(-r['score'],r['spectrum_index'])): representatives.setdefault(row['composition_key'],row)
    write_tsv(out/'filtered_peptides.tsv',representatives.values(),fields)
    write_tsv(out/'excluded_psms.tsv',[r for r in target if not r['accepted']],fields)
    for name,curve in [('psm',psm_curve),('peptide',peptide_curve)]:
        write_tsv(out/f'{name}_qvalue_curve.tsv',curve_rows(curve),['threshold','targets','decoys','estimated_fdr','qvalue'])
    write_tsv(out/'localization_qvalue_curve.tsv',loc_curve,['threshold_mbpr','psms','expected_errors','estimated_flr','q_loc'])
    summary={'status':'complete','estimator':'single_seed_plus_one_(D+1)/T','seed':manifest['seed'],'rho':manifest['rho'],
             'thresholds':{'psm_fdr':args.psm_fdr,'peptide_fdr':args.peptide_fdr,'localization_flr':args.localization_flr},
             'localization':'skipped_by_request' if args.identification_only else 'PTMProphet_EM0_MODPREC4_all_tokens',
             'localization_qvalue_definition':'tail-min cumulative PSM mean of (1 - site-weighted MBPr), conditional on both identification gates',
             'counts':{'target':len(target),'decoy':len(decoy),'nonempty_target':len(tvalid),'nonempty_decoy':len(dvalid),
                       'localized':len(site_rows),'accepted_psms':len(accepted),'accepted_compositions':len(representatives)},
             'excluded_reasons':dict(Counter(r['localization_reason'] for r in target if not r['accepted'])),
             'modifications':[{'name':t.name,'specification':t.option} for t in targets],
             'dataset_manifest_sha256':sha256(data/'dataset.json'),
             'prediction_files':{'target':[{'path':str(p),'sha256':sha256(p)} for p in tf],'decoy':[{'path':str(p),'sha256':sha256(p)} for p in df]}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['counts'],indent=2));print(f'Filtered results: {out / "filtered_psms.tsv"}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path,help='directory produced by prepare_decoys.py')
    p.add_argument('--target',required=True,nargs='+',help='target TSVs or quoted glob across all GPU ranks')
    p.add_argument('--decoy',required=True,nargs='+',help='decoy TSVs or quoted glob across all GPU ranks')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--psm-fdr',type=float,default=.01);p.add_argument('--peptide-fdr',type=float,default=.01)
    p.add_argument('--localization-flr',type=float,default=.01)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--ptmprophet',default=str(adapter.PTMPROPHET))
    p.add_argument('--pyopenms-python',type=Path,default=adapter.PYOPENMS_PYTHON)
    p.add_argument('--modifications',type=Path,default=Path(__file__).resolve().parent/'fdr/modifications.json')
    p.add_argument('--identification-only',action='store_true',help='explicitly omit PTM localization; outputs are identification-filtered only')
    args=p.parse_args()
    if not all(0<x<=1 for x in [args.psm_fdr,args.peptide_fdr,args.localization_flr]) or not 1<=args.workers<=16:
        p.error('thresholds must be in (0,1], workers in [1,16]')
    run(args)

if __name__=='__main__': main()
