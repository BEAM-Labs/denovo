### 1. Environment for NAT

```bash
conda create -n NATdenovo python=3.10
conda activate NATdenovo
pip install -r ./requirements.txt
```

installing gcc and g++:

```bash
conda install -c conda-forge gcc
conda install -c conda-forge cxx-compiler
```

then install ctcdecode, which is the package for ctc-beamsearch decoding: 
```bash
git clone --recursive https://github.com/WayenVan/ctcdecode.git
cd ctcdecode
pip install .
cd ..  #this is needed as ctcdecode can not be imported under the current directory
rm -rf ctcdecode
```

(if there are no errors, ignore the next line and proceed to CuPy install)

if you encountered issues with C++ (gxx and gcc) version errors in this step, install gcc with version specified as :
```bash
conda install -c conda-forge gcc_linux-64=9.3.0
```

lastly, install CuPy to use our CUDA-accelerated precise mass-control decoding:

**Please install the following Cupy package in a GPU available env, If you are using a slurm server, this means you have to enter a interative session with sbatch to install Cupy, If you are using a machine with GPU already on it (checking by nvidia-smi), then there's no problem**

**Check your CUDA version using command nvidia-smi, the CUDA version will be on the top-right corner**

| Cuda version |	Command |
|-------------------|--------|
|v10.2 (x86_64 / aarch64)	|pip install cupy-cuda102 |
|v11.0 (x86_64) |	pip install cupy-cuda110 |
|v11.1 (x86_64) |	pip install cupy-cuda111 |
|v11.2 ~ 11.8 (x86_64 / aarch64) |	pip install cupy-cuda11x |
|v12.x (x86_64 / aarch64) |	pip install cupy-cuda12x |


### 2. Model Settings

Some of the important settings in config.yaml under ./PrimeNovo

n_beam: number of CTC-paths (beams) considered during inference. We recommend a value of 40.

mass_control_tol: This setting is only useful when PMC_enable is True. The tolerance of PMC-decoded mass from the measured mass by MS, when mass control algorithm (PMC) is used. For example, if this is set to 0.1, we will only obtain peptides that fall under the mass range [measured_mass-0.1, measured_mass+0.1]. Measured mass is calculated by : (pepMass - 1.007276) * charge - 18.01. pepMass and charge are given by input spectrum file (MGF).

PMC_enable: Weather use PMC decoding unit or not, either True or False.

n_peaks: Number of the most intense peaks to retain, any remaining peaks are discarded. We recommend a value of 800.

min_mz: Minimum peak m/z allowed, peaks with smaller m/z are discarded. We recommend a value of 1.

max_mz: Maximum peak m/z allowed, peaks with larger m/z are discarded. We recommend a value of 6500.

min_intensity: Min peak intensity allowed, less intense peaks are discarded. We recommend a value of 0.0.

### 3. Run Instructions

Note!!!!!!!!!!!!!!!!!!: All the following steps should be performed under the main directory: pi-PrimeNovo. Do not use cd PrimeNovo !!!!!!!!!!!!!!!!!!!

**step 1. Download Required Files**

**Note:** If you are using a remote server, you can use the `gdown` package to easily download the content from Google Drive to your server disk.

**Step 2: Choose the Mode**

The `--mode` argument can be set to either:

- `eval`: Use this mode when evaluating data with a labeled dataset.
- `denovo`: Use this mode for de novo analysis on unlabeled data.

**Important**: Select `eval` only if your data is labeled.

**Step 3: Run the Commands**

Execute the following command in the terminal:

```bash
python -m PrimeNovo.PrimeNovo --mode=eval --peak_path=./bacillus.10k.mgf --model=./model_massive.ckpt
```

This automatically uses all GPUs available in the current machine.

**Step 4: analyze the output**

We include a sample running output ```./output.txt```. The performance for evaluation will be reported at the end of the output file.

If you are using ```denovo``` mode, you will get a ```denovo.tsv``` file under the current directory. The file has the following structure:

| label | prediction | charge | score |
| --- | --- | --- | --- |
| Title in MGF document | Sequence in [ProForma](https://doi.org/10.1021/acs.jproteome.1c00771) notation| Charge, as a number | Confidence score as number in range 0 and 1 using scientific notation |

The example below contains two peptides predicted based on some given spectrum:

```tsv
label	prediction	charge	score
MS_19321_2024_02_DDA	ATTALP	2	0.99
MS_19326_2024_02_DDA	TAM[+15.995]TR	2	0.87
```