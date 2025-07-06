<h1 align="center">De Novo Peptide Sequencing</h1>

<img width="1301" alt="Clipboard_Screenshot_1748418034" src="https://github.com/user-attachments/assets/5e194446-04ed-4f39-b1bd-1dccb4de155a" />

---

## 📃 Overview

This is repo containing all advanced De Novo peptide sequencing models developed by Beam Lab.

It includes:

| Model                | Model Checkpoint                                                                                 | Category | Brief Introduction                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **ContraNovo** | [ContraNovo](https://drive.google.com/file/d/1knNUqSwPf98j388Ds2E6bG8tAXx8voWR/view?usp=drive_link) | AT       | Autoregressive multimodal contrastive learning model for de novo sequencing.                                                                 |
| **PrimeNovo**  | [PrimeNovo](https://drive.google.com/file/d/12IZgeGP3ae3KksI5_82yuSTbk_M9sKNY/view?usp=share_link)  | NAT      | First NAT biological sequences model for fast sequencing.                                                                                    |
| **RefineNovo** | coming soon                                                                                      | NAT      | An ultra-stable NAT model framework that can adapt to any data distributions. (most stable training so far, guaranteed successful training). |
| **RankNovo**   | [RankNovo](https://drive.google.com/file/d/1Zfzpu5JHUvMXfvNPA-QVGzXMyF499vFL/view?usp=sharing)      | -        | A framework for combining any set of de novo models for combined power of accurate predictions.                                              |

(N)AT refers to (Non)-Autoregressive Transformer.

Test MGF File: [Bacillus.10k.mgf](https://drive.google.com/file/d/1HqfCETZLV9ZB-byU0pqNNRXbaPbTAceT/view?usp=drive_link)

Feel free to open Issues or start a Discussion to share your results!

## 🎉 News

- **[2025-05]** RefineNovo and RankNovo have been accepted by ICML'2025. 🎉
- **[2024-11]** PrimeNovo has been accepted by Nature Communications. 🎉
- **[2023-12]** ContraNovo has been accepted by AAAI'2024. 🎉

## 🌟 Get Started

### 1. Run AT denovo

Refer to [AT Denovo](./AT_denovo.md) for AT denovo environment preparation.

### 2. Run NAT denovo

Refer to [NAT Denovo](./NAT_denovo.md) for NAT denovo environment preparation.

### 2. Run RankNovo

Refer to [NAT Denovo](./RankNovo/readme.md) for NAT denovo environment preparation.

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

@article{zhang2025curriculum,
  title={Curriculum Learning for Biological Sequence Prediction: The Case of De Novo Peptide Sequencing},
  author={Zhang, Xiang and Wei, Jiaqi and Qiu, Zijie and Xu, Sheng and Dong, Nanqing and Gao, Zhiqiang and Sun, Siqi},
  journal={arXiv preprint arXiv:2506.13485},
  year={2025}
}

@article{qiu2025universal,
  title={Universal Biological Sequence Reranking for Improved De Novo Peptide Sequencing},
  author={Qiu, Zijie and Wei, Jiaqi and Zhang, Xiang and Xu, Sheng and Zou, Kai and Jin, Zhi and Gao, Zhiqiang and Dong, Nanqing and Sun, Siqi},
  journal={arXiv preprint arXiv:2505.17552},
  year={2025}
}
```
