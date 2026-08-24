# VMI-FGSM

## Original paper

**Enhancing the Transferability of Adversarial Attacks through Variance Tuning**  
Xiaosen Wang and Kun He  
CVPR 2021  
[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html)

## Face-verification adaptation

This implementation applies VMI-FGSM to the shared CNN face-verification pipeline. It estimates local gradient variation around the current source image and combines the variance-tuned direction with momentum. The classification objective is replaced by the pipeline's embedding cosine objective for impersonation or dodging.

The implementation is intended for CNN-based face-recognition models and is an adaptation for this repository. It is not an official implementation from the original paper authors.

## Repository contributor

- **Khushi**, IGDTUW
- GitHub: [Khushi250321](https://github.com/Khushi250321)

## Usage

Use the registry name `VMI_FGSM` with the shared generation script:

```bash
python experiments/generate_adversarial_examples.py \
  --input-csv /path/to/input_pairs.csv \
  --dataset-root /path/to/dataset_extractedfaces \
  --output-root /path/to/adv_outputs \
  --attacker-model Facenet512 \
  --attacks VMI_FGSM
```

The shared implementation is in [`core/transfer_attack_core.py`](../../core/transfer_attack_core.py).

## Presentation

- [`student_presentation.pdf`](student_presentation.pdf)
