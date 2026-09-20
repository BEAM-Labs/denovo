#!/usr/bin/env python3
"""Prepare paired target/peak-replaced-decoy MGF and LMDB before inference."""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'fdr'))
import lmdb
import numpy as np
from decoy_engine import generate_rate, read_targets, mz_keys, array_hash
from io_utils import read_spectra, write_mgf_record, write_tsv, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,nargs='+',type=Path,help='MGF/LMDB files; pooled as one FDR population')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--rho',type=float,default=0.5)
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    if not 0 < args.rho <= 1 or args.seed < 0:
        parser.error('rho must be in (0,1]; seed must be nonnegative')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=lmdb.open(str(out/'target.lmdb'),subdir=False,map_size=1<<40)
    registry=[]; removed=0; merged=0
    try:
        with env.begin(write=True) as txn, (out/'target.mgf').open('w') as mgf:
            for file_index,path in enumerate(args.input):
                for source_index,record in enumerate(read_spectra(path)):
                    row=dict(record)
                    mz=np.asarray(row['mz_array'],np.float32); intensity=np.asarray(row['intensity_array'],np.float32)
                    if len(mz)!=len(intensity) or not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensity)):
                        raise ValueError(f'invalid peaks: {path}, scan {source_index}')
                    keep=(mz>0)&(intensity>0); removed+=int((~keep).sum())
                    mz,intensity=mz[keep],intensity[keep]
                    # Same normalization is used for both inference arms and donor pool.
                    order=np.argsort(-intensity,kind='stable'); _,first=np.unique(mz_keys(mz[order]),return_index=True)
                    select=order[first]; merged+=len(mz)-len(select); select=select[np.argsort(mz[select],kind='stable')]
                    row['mz_array'],row['intensity_array']=mz[select],intensity[select]
                    if not len(select):
                        raise ValueError(f'no positive peaks: {path}, scan {source_index}')
                    if not math.isfinite(float(row['precursor_mz'])) or float(row['precursor_mz'])<=0 or int(row['precursor_charge'])<=0:
                        raise ValueError(f'invalid precursor: {path}, scan {source_index}')
                    index=len(registry); title=f'target|{index:09d}'
                    registry.append({'spectrum_index':index,'title':title,'source_file':str(path.resolve()),'source_index':source_index,'source_title':row.get('title',''),'precursor_mz':row['precursor_mz'],'precursor_charge':row['precursor_charge']})
                    row['title']=title; row['pep']=''
                    txn.put(str(index).encode(),pickle.dumps(row,protocol=pickle.HIGHEST_PROTOCOL))
                    write_mgf_record(mgf,row,index)
            if not registry:
                raise ValueError('input contains no spectra')
            txn.put(b'n_spectra',str(len(registry)).encode()); txn.put(b'ms_level',b'2')
    finally:
        env.close()
    target=out/'target.lmdb'
    spectra,pool_mz,pool_intensity=read_targets(target)
    pool_keys=mz_keys(pool_mz)
    # One global set, then bounded per-spectrum checks before rejection sampling.
    distinct=set(map(int,pool_keys))
    for i,spectrum in enumerate(spectra):
        n=len(spectrum['mz']); needed=n-math.floor(n*(1-args.rho))
        if len(distinct)-len(set(map(int,mz_keys(spectrum['mz'])))) < needed:
            raise ValueError(f'not enough distinct donor peaks for spectrum {i}; include more spectra')
    generate_rate('paired',target,spectra,pool_mz,pool_intensity,pool_keys,sha256(target),
                  array_hash(np.column_stack((pool_mz,pool_intensity)).astype(np.float32)),
                  out,str(args.seed),args.seed,args.rho,len(spectra))
    generated=out/'paired'/f'seed_{args.seed}'/f'rho_{args.rho:.2f}'
    for name in ('decoy.lmdb','decoy.mgf','generation_trace.parquet','generation.json'):
        (generated/name).rename(out/name)
    for row in registry:
        row['decoy_title']=f"seed{args.seed}|r{args.rho:.2f}|{row['title']}"
    write_tsv(out/'spectrum_registry.tsv',registry,list(registry[0]))
    manifest={'schema_version':1,'seed':args.seed,'rho':args.rho,'n_spectra':len(registry),
              'sources':[{'path':str(p.resolve()),'sha256':sha256(p)} for p in args.input],
              'normalization':{'nonpositive_peaks_removed':removed,'duplicate_1e5_mz_peaks_removed':merged,'duplicate_policy':'keep highest intensity; sort m/z'},
              'files':{name:{'sha256':sha256(out/name)} for name in ('target.lmdb','target.mgf','decoy.lmdb','decoy.mgf','spectrum_registry.tsv')}}
    (out/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared {len(registry)} target + {len(registry)} decoy spectra in {out}')

if __name__=='__main__':
    main()
