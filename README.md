<h1 align="center">De Novo Peptide Sequencing</h1>

<img width="1301" alt="Clipboard_Screenshot_1748418034" src="https://github.com/user-attachments/assets/5e194446-04ed-4f39-b1bd-1dccb4de155a" />


---

## 📃 Overview

This is repo containing all advanced De Novo peptide sequencing models developed by Beam Lab.

It includes:

| Model | Model Checkpoint | Category | Brief Introduction |
|-------------------|--------|-----------------------------------------------------------------------|
| ContraNovo | ContraNovo | AT |  Autoregressive multimodal contrastive learning model for de novo sequencing. | 
| PrimeNovo | PrimeNovo| NAT | First NAT biological sequences model for fast sequencing. |
| RefineNovo | RefineNovo | NAT | An ultra-stable NAT model framework that can adapt to any data distributions. (most stable training so far, guaranteed successful training). |
| RankNovo | RankNovo | NAT | A framework for combining any set of de novo models for combined power of accurate predictions. |
| CrossNovo | come soon | AT | A unified model combining knowledge learnt from both NAT and AT model (best single-model performance so far). |
| Reflection based model | come soon | AT | A model that allows for CoT and self-correction (human interferability). |

(N)AT refers to (Non)-Autoregressive Transformer.

Feel free to open Issues or start a Discussion to share your results! 🎉


## 🎉 News

- **[2025-05]** RefineNovo and RankNovo have been accepted by ICML'2025.🎉

- **[2024-11]** PrimeNovo has been accepted by Nature Communications.🎉

- **[2023-12]** ContraNovo has been accepted by AAAI'2024.🎉


## 🌟 Get Started


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

@misc{qiu2025universalbiologicalsequencereranking,
      title={Universal Biological Sequence Reranking for Improved De Novo Peptide Sequencing}, 
      author={Zijie Qiu and Jiaqi Wei and Xiang Zhang and Sheng Xu and Kai Zou and Zhi Jin and Zhiqiang Gao and Nanqing Dong and Siqi Sun},
      year={2025},
      eprint={2505.17552},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17552}, 
}
