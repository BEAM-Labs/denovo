# RefineNovo

This is the official repo for the ICML paper: **[Curriculum Learning for Biological Sequence Prediction: The Case of De Novo Peptide Sequencing](https://arxiv.org/pdf/2506.13485)**

We will release the future model update here, please leave a **star** and **watching** if you want to get notified and follow up.
![prime](./assets/Glancing.png)



## Notes from Authors 

1. We developed our algorithm in CentOS Linux Version 7. For other OS systems, users need to check compability themselves.

2.  The MacOS users (with non-intel core) currently can not use this model due to un-supported CUDA drive.

3.  Machines that don't have Nvidia-GPUs can not use our algorithms as PMC is written directly with CUDA core, which is only supported by Nvidia-GPUs.

4.  We use lmdb for fast MS data reading during trianing and inference time. Once you provide your mgf and execute RefineNovo, an lmdb file will be automatically generated for you. You can save this lmdb file and use it directly next time during training/inference, and no processing time will needed second time you load it. This is very good for large MS file processing and saves hours to days spent on pre-processing data and loading data to memory each time you train/inference a neural network model. For detailed implementation refer to dataloader and dataset file in our code base. 

5. We notice that CuPy Sometimes gives random errors, which can be resolved simply by re-running code or switching a GPU node.
   
## Environment Setup



Create a new conda environment first:

```
conda create --name RefineNovo python=3.10
```

This will create an anaconda environment

Activate this environment by running:

```
conda activate RefineNovo
```

then install dependencies:

```
pip install -r ./requirements.txt
```

installing gcc and g++:

```bash
conda install -c conda-forge gcc
conda install -c conda-forge cxx-compiler
```

then install ctcdecode, which is the package for ctc-beamsearch decoding

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

then install pytorch imputer for CTC-curriculum sampling

```bash
cd imputer-pytorch
pip install -e .
cd ..
```

lastly, install CuPy to use our CUDA-accelerated precise mass-control decoding:

**_Please install the following Cupy package in a GPU available env, If you are using a slurm server, this means you have to enter a interative session with sbatch to install Cupy, If you are using a machine with GPU already on it (checking by nvidia-smi), then there's no problem_**

**Check your CUDA version using command nvidia-smi, the CUDA version will be on the top-right corner**

| cuda version | command |
|-------|-------|
|v10.2 (x86_64 / aarch64)| pip install cupy-cuda102 |
|v11.0 (x86_64)| pip install cupy-cuda110 |
|v11.1 (x86_64)| pip install cupy-cuda111 |
|v11.2 ~ 11.8 (x86_64 / aarch64)| pip install cupy-cuda11x |
|v12.x (x86_64 / aarch64)| pip install cupy-cuda12x |




## Model Settings

Some of the important settings in config.yaml under ./RefineNovo 

**n_beam**: number of CTC-paths (beams) considered during inference. We recommend a value of 40.

**mass_control_tol**: This setting is only useful when **PMC_enable** is ```True```. The tolerance of PMC-decoded mass from the measured mass by MS, when mass control algorithm (PMC) is used. For example, if this is set to 0.1, we will only obtain peptides that fall under the mass range [measured_mass-0.1, measured_mass+0.1]. ```Measured mass``` is calculated by : (pepMass - 1.007276) * charge - 18.01. pepMass and charge are given by input spectrum file (MGF).

**PMC_enable**: Weather use PMC decoding unit or not, either ```True``` or ```False```.

**n_peaks**: Number of the most intense peaks to retain, any remaining peaks are discarded. We recommend a value of 800.

**min_mz**: Minimum peak m/z allowed, peaks with smaller m/z are discarded. We recommend a value of 1.

**max_mz**: Maximum peak m/z allowed, peaks with larger m/z are discarded. We recommend a value of 6500.

**min_intensity**: Min peak intensity allowed, less intense peaks are discarded. We recommend a value of 0.0.

## Run Instructions

**Note!!!!!!!!!!!!!!!!!!:** All the following steps should be performed under the main directory: `RefineNovo`. Do **not** use `cd RefineNovo` !!!!!!!!!!!!!!!!!!!

### Step 1: Download Required Files

To evaluate the provided test MGF file (you can replace this MGF file with your own), download the following files:

1. **Model Checkpoint**: [refineNovo_massivekb.ckpt](https://drive.google.com/file/d/1NtEIdrm1lccZRWOeO20-2c3Ekop-BhCJ/view?usp=sharing)
2. **Test MGF File**: [Bacillus.10k.mgf](https://drive.google.com/file/d/1HqfCETZLV9ZB-byU0pqNNRXbaPbTAceT/view?usp=drive_link)

**Note:** If you are using a remote server, you can use the `gdown` package to easily download the content from Google Drive to your server disk.

### Step 2: Choose the Mode

The `--mode` argument can be set to either:

- `eval`: Use this mode when evaluating data with a labeled dataset.
- `denovo`: Use this mode for de novo analysis on unlabeled data.

**Important**: Select `eval` only if your data is labeled.

### Step 3: Run the Commands

Execute the following command in the terminal:

```bash
python -m RefineNovo.RefineNovo --mode=eval --peak_path=./bacillus.10k.mgf --model=./refineNovo_massivekb.ckpt
```

This automatically uses all GPUs available in the current machine.

## Citation

