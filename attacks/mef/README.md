# MEF

## Original paper

**Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**  
Chunlin Qiu, Ang Li, Yiheng Duan, Shenyi Zhang, Yuanjie Zhang, Lingchen Zhao, and Qian Wang  
IEEE Transactions on Information Forensics and Security, 2024  

## Face-verification adaptation

MEF samples local neighborhoods around the current adversarial image and performs an inner flatness-oriented gradient update before applying the outer perturbation update. This adaptation replaces the original classification loss with the shared embedding cosine objective and targets CNN-based face-verification models.

This is a student adaptation for the shared face-verification pipeline, not an official implementation from the paper authors.

## Repository contributor

- **Hiya Trehan**, IGDTUW

## Usage

Use the registry name `MEF` with the shared generation script:

```bash
python experiments/generate_adversarial_examples.py \
  --input-csv /path/to/input_pairs.csv \
  --dataset-root /path/to/dataset_extractedfaces \
  --output-root /path/to/adv_outputs \
  --attacker-model Facenet512 \
  --attacks MEF
```

The shared implementation is in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py).

## Verification note

A limited smoke run confirmed that the implementation executes and produces valid adversarial outputs. The full result reported in the supplied presentation should be reproduced with the complete common evaluation protocol before being treated as a verified benchmark result.

## Presentation

- [`student_presentation.pdf`](student_presentation.pdf)
