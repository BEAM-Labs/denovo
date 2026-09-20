# OmniNovo

Inference and target/decoy filtering for *Accurate de novo sequencing of the modified proteome with OmniNovo*.

## 1. Install the environment

The supported runtime is Linux with an NVIDIA Ampere-or-newer GPU and compatible CUDA 11.x/12.x drivers. Inference uses Flash Attention and bf16; decoy preparation and FDR filtering run on CPU. macOS and CPU-only inference are not supported.

The recommended route is the pre-packed environment:

```bash
mkdir -p ~/miniconda3/envs/OmniNovo
tar -xzf /path/to/OmniNovo.tar.gz -C ~/miniconda3/envs/OmniNovo
conda activate ~/miniconda3/envs/OmniNovo
conda-unpack
pip install -r fdr/requirements.txt
```

The environment archive is available from [Google Drive](https://drive.google.com/file/d/1GgmrywunIhPAywXIdPqu7zSUN1Yd7F40/view?usp=drive_link). It is external and is not included in this ZIP.

As an alternative, create the environment from the included definition:

```bash
conda env create -f environment.yml
conda activate OmniNovo
pip install -r fdr/requirements.txt
```

For a pip-based installation in an existing Python 3.10 environment, use `requirements.txt`. The file assumes CUDA 12.1 wheels; for CUDA 11.x, install the matching PyTorch and CuPy wheels instead (`cupy-cuda11x`). Flash Attention must be built after PyTorch is installed.

If CuPy reports a missing CUDA runtime library, expose the CUDA libraries in the active environment and retry:

```bash
export LD_LIBRARY_PATH="$(python -c 'import glob,sysconfig; print(":".join(glob.glob(sysconfig.get_paths()["purelib"]+"/nvidia/*/lib")))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Download the [OmniNovo checkpoint](https://drive.google.com/file/d/1chNcOOnHfoJKh7VBfO8C_l5bQg_pVb-g/view?usp=drive_link) and save it to a known path. The checkpoint is also an external download and is not bundled.

For PTM localization, install **PTMProphetParser (TPP 6.3.2, tested)** from the [Trans-Proteomic Pipeline](https://tools.proteomecenter.org/software.php). Install pyOpenMS in a separate Python 3.10 environment:

```bash
conda create -n omninovo-ptmprophet python=3.10 -y
conda run -n omninovo-ptmprophet pip install -r fdr/requirements-localization.txt
export PYOPENMS_PYTHON="$(conda run -n omninovo-ptmprophet python -c 'import sys; print(sys.executable)')"
```

Alternatively pass an existing pyOpenMS interpreter with `--pyopenms-python /path/to/python`.

## 2. Run the complete workflow

Run commands from this code folder with the OmniNovo environment active:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh \
  example/sample_100.mgf /path/to/omninovo.ckpt /path/to/new_run \
  --ptmprophet /path/to/PTMProphetParser
```

Replace the example MGF with an MGF or single-file LMDB input. The output directory must not already exist. Omit `CUDA_VISIBLE_DEVICES=0` to use all visible GPUs. `RHO=0.50`, `SEED=7`, `CONFIG=/path/to/config.yaml` and `PYTHON=/path/to/python` can override the defaults.

The 100-spectrum example checks execution. At the default 1% thresholds it can produce no accepted rows; use a larger dataset for FDR analysis.

## 3. Run the steps separately

Prepare paired target/decoy spectra:

```bash
python prepare_decoys.py --input /path/to/input.mgf \
  --output /path/to/dataset --rho 0.50 --seed 7
```

Run inference on both arms with the same checkpoint and configuration:

```bash
for ARM in target decoy; do
  CUDA_VISIBLE_DEVICES=0 python -m OmniNovo.run --mode denovo \
    --input_path="/path/to/dataset/$ARM.lmdb" \
    --model=/path/to/omninovo.ckpt \
    --config=OmniNovo/configs/default_rules.yaml \
    --log_base="/path/to/predictions/$ARM"
done
```

Filter the prediction sets:

```bash
python filter_fdr.py --dataset /path/to/dataset \
  --target '/path/to/predictions/target/*rank*.tsv' \
  --decoy '/path/to/predictions/decoy/*rank*.tsv' \
  --output /path/to/filtered \
  --ptmprophet /path/to/PTMProphetParser
```

Include every GPU rank and retain all prediction rows. The filter accepts `--psm-fdr`, `--peptide-fdr`, `--localization-flr`, `--workers` and `--identification-only`.

## 4. Outputs and supported modifications

The filtered directory contains `filtered_psms.tsv`, `filtered_peptides.tsv`, `all_psms.tsv`, `excluded_psms.tsv`, `summary.json`, q-value curves and PTMProphet results. All 11 model modifications and candidate sites are defined in `fdr/modifications.json`; PTM-site constraints are enabled by `default_rules.yaml`.

## 5. FragPipe reference-search configuration

The configuration snapshots used for the ultradeep HeLa reference search are in [`config/fragpipe/`](config/fragpipe/). They include the FragPipe 22.0 workflow and the corresponding MSFragger parameter file. Machine-specific paths were replaced with portable placeholders; see [`config/fragpipe/README.md`](config/fragpipe/README.md) before running the search.
