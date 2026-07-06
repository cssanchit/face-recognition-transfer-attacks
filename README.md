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
| [`PGD`](attacks/README.md) | *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018) | Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| [`MI_FGSM`](attacks/README.md) | *Boosting Adversarial Attacks with Momentum* (CVPR 2018) | Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, Jianguo Li | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| [`TI_FGSM`](attacks/README.md) | *Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks* (CVPR 2019) | Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu | Shared repository implementation integrated in the CNN face-verification pipeline by **Ananya Jain** from **Jaypee Institute of Information Technology, Noida** |
| [`SI_NI_FGSM`](attacks/README.md) | *Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks* (2020) | Jiadong Lin, Chuanbai Xiao, Lei Yang, Xiaoyuan Mao, Tian Xia, Tong He, Wei Yang | Shared repository implementation integrated in the CNN face-verification pipeline by **Vishaka** from **Delhi University** |
| [`MI_ADMIX_DI_TI`](attacks/README.md) | *Admix* (ICCV 2021) with DI/TI-style transfer components | Xiaosen Wang, Jiefeng Chen, Si'wei Lyu, Yisen Wang | Shared repository implementation integrated in the CNN face-verification pipeline by **Ananya Jain** from **Jaypee Institute of Information Technology, Noida** |
| [`BPA_CNN`](attacks/bpa_cnn) | *Rethinking the Backward Propagation for Adversarial Transferability* (NeurIPS 2023) | Xiaosen Wang, Kangheng Tong, Kun He | CNN-side face-verification adaptation contributed by [**Om Singh Rawat**](https://github.com/Om-Singh-Rawat) from **IIT Delhi** |
| [`BSR`](attacks/bsr) | *Boosting Adversarial Transferability by Block Shuffle and Rotation* (CVPR 2024) | Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang | Face-verification adaptation contributed by [**Chirag Sharma**](https://github.com/chiraagsharma24) from **IIIT Vadodara** |
| [`DECOWA`](attacks/decowa) | *Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping* (AAAI 2024) | Jiayi Lin, Chuanbai Xiao, Chao Ma, Jie Zhang, Qiong Cao, Xiaosen Wang | Face-verification adaptation contributed by [**Om Singh Rawat**](https://github.com/Om-Singh-Rawat) from **IIT Delhi** |
| [`SIA_MI_TI`](attacks/sia_mi_ti) | *Structure Invariant Transformation for better Adversarial Transferability* (ICCV 2023) | Xiaosen Wang, Zeliang Zhang, Jianping Zhang | CNN face-verification adaptation contributed by [**Janhavi Kishor**](https://github.com/Janhavi187) from **SRM University** |
| [`SIA`](attacks/sia) | *Structure Invariant Transformation for better Adversarial Transferability* (ICCV 2023) | Xiaosen Wang, Zeliang Zhang, Jianping Zhang | CNN face-verification adaptation contributed by [**Aditi Raj**](https://github.com/aditiraj2006) from **IGDTUW** |
| [`OPS`](attacks/ops) | *Boosting Adversarial Transferability through Augmentation in Hypothesis Space* (CVPR 2025) | Yu Guo, Weiquan Liu, Qingshan Xu, Shijun Zheng, Shujun Huang, Yu Zang, Siqi Shen, Chenglu Wen, Cheng Wang | Face-verification adaptation contributed by [**Kkartik Aggarwal**](https://github.com/skigeek16) from **Delhi Technological University (DTU)** |
| [`ATT_CNN`](attacks/att_cnn) | *Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning* (NeurIPS 2024) | Di Ming, Peng Ren, Yunlong Wang, Xin Feng | CNN-side adaptation inspired by the original ATT method, contributed by [**Keshav Raj**](https://github.com/keshavraj220507) from **IIIT Delhi** |
| [`LI_BOOST_MI`](attacks/li_boost_mi) | *Boosting the Local Invariance for Better Adversarial Transferability* (arXiv 2025) | Bohan Liu, Xiaosen Wang | CNN face-verification adaptation contributed by [**Charushi**](https://github.com/Charushi06) from **IGDTUW** |
| [`GRA`](attacks/gra) | *Boosting Adversarial Transferability via Gradient Relevance Attack* (ICCV 2023) | Yingwen Wu, Yinpeng Dong, Qin Wang, Jun Zhu, Xiaolin Hu | CNN face-verification adaptation contributed by [**Krish Bansal**](https://github.com/KrishBnsl) from **Delhi Technological University (DTU)** |
| [`IDAA`](attacks/idaa) | Student-submitted IDAA basis for CNN face verification | Not independently standardized in the submitted materials | CNN face-verification adaptation contributed by [**Arnav Asthana**](https://github.com/Arnav1730) from **IIIT Delhi** |
| [`DPA_HMA`](attacks/dpa_hma) | *Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation* (CVPR 2025) | Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang | Face-recognition adaptation contributed by [**Kushal Khemka**](https://github.com/Kushalkhemka) from **Delhi Technological University (DTU)** |
| [`DYNAMIC_MORPH`](attacks/dynamic_morph) | Student-submitted morph-style semantic face-region mixing basis for CNN face verification | Not independently standardized in the submitted materials | CNN face-verification adaptation contributed by [**Puneet Kumar**](https://github.com/puneetkumar-2005) from **Indian Institute of Information Technology, Senapati, Manipur** |

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
