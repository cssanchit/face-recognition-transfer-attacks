# Usage

## 1. Core entry points
- Shared implementations: `core/transfer_attack_core.py`
- Batch generation script: `experiments/generate_adversarial_examples.py`

## 2. Expected input CSV
The generation script expects a CSV with at least the following columns:
- `row_id`
- `img1`
- `img2`
- `dataset`
- `attack_type`

`attack_type` should be one of:
- `impersonation_attack`
- `dodging_attack`

Meaning of `attack_type` in the input CSV:
- `impersonation_attack`: `img1` and `img2` should belong to different identities. The goal is to make the source image (`img1`) match the target identity (`img2`).
- `dodging_attack`: `img1` and `img2` should belong to the same identity. The goal is to make a genuine pair fail verification.

## 3. Example command
```bash
python experiments/generate_adversarial_examples.py \
  --input-csv /path/to/input_pairs.csv \
  --dataset-root /path/to/dataset_extractedfaces \
  --output-root /path/to/adv_outputs \
  --attacker-model ArcFace \
  --attacks MI_FGSM,BSR,DPA_HMA
```

## 4. Supported attacker models
- `Facenet512`
- `ArcFace`
- `GhostFaceNet`
- `VGG-Face`

## 5. Programmatic usage
You may also import the shared core directly:

```python
from core.transfer_attack_core import build_attacker, load_and_preprocess, run_attack
```

## 6. Data note
This repository does not redistribute face datasets. Users should supply their own aligned face crops and pair lists.

## 7. Reproducibility note
Some contributed attacks are faithful face-verification adaptations of the original methods, while others are CNN-oriented reinterpretations inspired by the original papers. Please read the attack-specific README before using a method in a paper or comparison study.
