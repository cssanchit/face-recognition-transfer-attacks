# face-recognition-transfer-attacks

A public repository of transfer-based adversarial attack implementations adapted for **CNN-based face recognition / face verification systems**.

This repository focuses on **attack code, usage instructions, paper references, and contributor attribution**. It does **not** aim to be a public benchmark leaderboard. The implementations here were adapted in a shared face-verification pipeline and are intended to help researchers and students quickly reuse, inspect, and extend transfer attacks in the face-recognition setting.

## Scope
- CNN-oriented face-recognition attack adaptations
- transfer-based adversarial attacks for embedding-based face verification
- student-contributed implementations consolidated into one reusable codebase
- per-attack documentation with original paper references and contributor credit

## Important note
Some attacks in this repository were originally proposed for **generic image classification** or **Vision Transformer settings**. The code included here reflects **CNN-side face-verification adaptations**, not necessarily exact official reproductions of the original paper implementations.

To avoid ambiguity, this repository distinguishes between:
- **original paper authors**, who proposed the method in the literature, and
- **repository implementation contributors**, who adapted or integrated a version of the attack for this CNN face-verification codebase.

## Repository layout
- `core/transfer_attack_core.py`: shared attack implementations and registry
- `experiments/generate_adversarial_examples.py`: generic batch generation script
- `attacks/`: per-attack notes, references, contributor information, and student slide PDFs
- `docs/usage.md`: how to run the code
- `CONTRIBUTORS.md`: attack-wise contributor list

## Attack catalog

| Attack | Original paper / source | Original authors | Repository implementation / adaptation |
|---|---|---|---|
| `PGD` | *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018) | Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| `MI_FGSM` | *Boosting Adversarial Attacks with Momentum* (CVPR 2018) | Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, Jianguo Li | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| `TI_FGSM` | *Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks* (CVPR 2019) | Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| `SI_NI_FGSM` | *Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks* (2020) | Jiadong Lin, Chuanbai Xiao, Lei Yang, Xiaoyuan Mao, Tian Xia, Tong He, Wei Yang | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| `MI_ADMIX_DI_TI` | *Admix: Enhancing the Transferability of Adversarial Attacks* (ICCV 2021), combined with DI and TI style transfer components | Xiaosen Wang, Jiefeng Chen, Si'wei Lyu, Yisen Wang | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| `BPA_CNN` | *Rethinking the Backward Propagation for Adversarial Transferability* (NeurIPS 2023) | Xiaosen Wang, Kangheng Tong, Kun He | CNN-side face-verification adaptation contributed by **Om Singh Rawat** from **IIT Delhi** |
| `BSR` | *Boosting Adversarial Transferability by Block Shuffle and Rotation* (CVPR 2024) | Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang | Face-verification adaptation contributed by **Chirag Sharma** from **IIIT Vadodara** |
| `DECOWA` | *Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping* (AAAI 2024) | Jiayi Lin, Chuanbai Xiao, Chao Ma, Jie Zhang, Qiong Cao, Xiaosen Wang | Face-verification adaptation contributed by **Om Singh Rawat** from **IIT Delhi** |
| `SIA_MI_TI` | *Structure Invariant Transformation for better Adversarial Transferability* (ICCV 2023) | Xiaosen Wang, Zeliang Zhang, Jianping Zhang | CNN face-verification adaptation contributed by **Janhavi Kishor** from **SRM University** |
| `OPS` | *Boosting Adversarial Transferability through Augmentation in Hypothesis Space* (CVPR 2025) | Yu Guo, Weiquan Liu, Qingshan Xu, Shijun Zheng, Shujun Huang, Yu Zang, Siqi Shen, Chenglu Wen, Cheng Wang | Face-verification adaptation contributed by **Kkartik Aggarwal** from **Delhi Technological University (DTU)** |
| `ATT_CNN` | *Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning* (NeurIPS 2024) | Di Ming, Peng Ren, Yunlong Wang, Xin Feng | CNN-side adaptation inspired by the original ATT method, contributed by **Keshav Raj** from **IIIT Delhi** |
| `LI_BOOST_MI` | Student-submitted MI-style logarithmic-shift boosting adaptation basis | No single canonical upstream paper reference was finalized in the submission materials | CNN face-verification adaptation contributed by **Charushi** from **IGDTUW** |
| `DPA_HMA` | *Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation* (CVPR 2025) | Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang | Face-recognition adaptation contributed by **Kushal Khemka** from **Delhi Technological University (DTU)** |
| `DPA_HMA_ENSEMBLE` | Extension built on the DPA-based face-recognition adaptation in this repository | Based on the DPA source above | Ensemble extension integrated in this repository on top of the shared DPA_HMA adaptation |

## Quick start
Install dependencies:

```bash
pip install -r requirements.txt
```

Generate adversarial images from a CSV of source-target pairs:

```bash
python experiments/generate_adversarial_examples.py \
  --input-csv /path/to/input_pairs.csv \
  --dataset-root /path/to/dataset_extractedfaces \
  --output-root /path/to/adv_outputs \
  --attacker-model Facenet512 \
  --attacks BSR,DPA_HMA,LI_BOOST_MI
```

For input CSV format and field descriptions, see [`docs/usage.md`](docs/usage.md).

## Acknowledgment
This repository consolidates student implementations created in a guided research setting. Each attack folder includes the contributor name, college affiliation, original paper reference, and an implementation note describing the adaptation.
