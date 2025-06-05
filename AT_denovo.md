### 1. Environment for AT

```bash
conda create -n ATdenovo python=3.10
conda env create -f environment.yml
conda activate ATdenovo
```

### 2. Run ContraNovo

Run ContraNovo test on bacillus.10k.mgf:

```bash
python -m ContraNovo.ContraNovo  --mode=eval --peak_path=./ContraNovo/bacillus.10k.mgf --model=./ContraNovo/ContraNovo.ckpt
```