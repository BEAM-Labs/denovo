"""Strict spectrum and TSV I/O shared by preparation and filtering."""
import csv
import hashlib
import json
import pickle
import re
from pathlib import Path

import lmdb
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_spectra(path):
    path = Path(path)
    if path.suffix.lower() == '.lmdb':
        env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
        try:
            with env.begin() as txn:
                for i in range(int(txn.get(b'n_spectra'))):
                    value = txn.get(str(i).encode())
                    if value is None:
                        raise ValueError(f'missing LMDB spectrum {i}')
                    yield pickle.loads(value)
        finally:
            env.close()
    elif path.suffix.lower() == '.mgf':
        from ptmprophet_adapter import iter_mgf
        for i, (meta, peaks) in enumerate(iter_mgf(path)):
            charge = meta.get('CHARGE', '')
            if not re.fullmatch(r'[1-9][0-9]*\+?', charge):
                raise ValueError(f'{path}: spectrum {i} needs one positive CHARGE, got {charge!r}')
            pairs = np.array([list(map(float, p.split()[:2])) for p in peaks], dtype=np.float32)
            if not len(pairs):
                raise ValueError(f'{path}: spectrum {i} has no peaks')
            yield {'title':meta.get('TITLE',''), 'precursor_mz':float(meta['PEPMASS'].split()[0]),
                   'precursor_charge':int(charge.rstrip('+')), 'mz_array':pairs[:,0],
                   'intensity_array':pairs[:,1], 'retention_time':float(meta.get('RTINSECONDS',-1)),
                   'pep':meta.get('SEQ',meta.get('PEP',''))}
    else:
        raise ValueError('input must be .mgf or single-file .lmdb')


def write_mgf_record(handle, row, index):
    handle.write(f"BEGIN IONS\nTITLE={row['title']}\nPEPMASS={row['precursor_mz']:.12g}\nCHARGE={row['precursor_charge']}+\nSCANS={index}\n")
    if float(row.get('retention_time',-1)) >= 0:
        handle.write(f"RTINSECONDS={row['retention_time']:.12g}\n")
    for mz, intensity in zip(row['mz_array'],row['intensity_array']):
        handle.write(f'{float(mz):.9g} {float(intensity):.9g}\n')
    handle.write('END IONS\n\n')


def write_tsv(path, rows, fields):
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t',extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items() if k in fields})
