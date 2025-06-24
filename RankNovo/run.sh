#!/bin/bash
conda activate ranknovo


test_path=./test_data/bacillus_V1.mgf
msa_test_path=./test_data/bacillus_6m_V1.pkl
model="./ckpt/model_0.660.ckpt"
srun python -m RankNovo.RankNovo --mode=eval --peak_path=${test_path} --model=${model} --msa_test_path=${msa_test_path}

