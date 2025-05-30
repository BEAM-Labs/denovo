<h1 align="center">De Novo Peptide Sequencing</h1>

<img width="1301" alt="Clipboard_Screenshot_1748418034" src="https://github.com/user-attachments/assets/5e194446-04ed-4f39-b1bd-1dccb4de155a" />


---

## 📃 Overview

This is repo containing all advanced De Novo peptide sequencing models developed by Beam Lab.

It includes:

| Model | Model Checkpoint | Category | Brief Introduction |
|-------------------|--------|--------|-----------------------------------------------------------------------|
| **ContraNovo** | [ContraNovo](https://drive.google.com/file/d/1knNUqSwPf98j388Ds2E6bG8tAXx8voWR/view?usp=drive_link) | AT |  Autoregressive multimodal contrastive learning model for de novo sequencing. | 
| **PrimeNovo** | [PrimeNovo](https://drive.google.com/file/d/12IZgeGP3ae3KksI5_82yuSTbk_M9sKNY/view?usp=share_link) | NAT | First NAT biological sequences model for fast sequencing. |
| **RefineNovo** | coming soon | NAT | An ultra-stable NAT model framework that can adapt to any data distributions. (most stable training so far, guaranteed successful training). |
| **RankNovo** | coming soon | - | A framework for combining any set of de novo models for combined power of accurate predictions. |

(N)AT refers to (Non)-Autoregressive Transformer.

Test MGF File: [Bacillus.10k.mgf](https://drive.google.com/file/d/1HqfCETZLV9ZB-byU0pqNNRXbaPbTAceT/view?usp=drive_link)

Feel free to open Issues or start a Discussion to share your results!


## 🎉 News

- **[2025-05]** RefineNovo and RankNovo have been accepted by ICML'2025. 🎉

- **[2024-11]** PrimeNovo has been accepted by Nature Communications. 🎉

- **[2023-12]** ContraNovo has been accepted by AAAI'2024. 🎉


## 🌟 Get Started for AT De Novo

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

## 🌟 Get Started for NAT De Novo

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



## 🎈 Citations

If you use this project, please cite:

```bibtex
@inproceedings{jin2024contranovo,
  title={Contranovo: A contrastive learning approach to enhance de novo peptide sequencing},
  author={Jin, Zhi and Xu, Sheng and Zhang, Xiang and Ling, Tianze and Dong, Nanqing and Ouyang, Wanli and Gao, Zhiqiang and Chang, Cheng and Sun, Siqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={144--152},
  year={2024}
}

@article{zhang2025pi,
  title={$\pi$-PrimeNovo: an accurate and efficient non-autoregressive deep learning model for de novo peptide sequencing},
  author={Zhang, Xiang and Ling, Tianze and Jin, Zhi and Xu, Sheng and Gao, Zhiqiang and Sun, Boyan and Qiu, Zijie and Wei, Jiaqi and Dong, Nanqing and Wang, Guangshuai and others},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={267},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

@article{qiu2025universal,
  title={Universal Biological Sequence Reranking for Improved De Novo Peptide Sequencing},
  author={Qiu, Zijie and Wei, Jiaqi and Zhang, Xiang and Xu, Sheng and Zou, Kai and Jin, Zhi and Gao, Zhiqiang and Dong, Nanqing and Sun, Siqi},
  journal={arXiv preprint arXiv:2505.17552},
  year={2025}
}
