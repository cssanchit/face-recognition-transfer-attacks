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

## 8. FaceSM objective selection

Add `--objective facesm` to use FaceSM, or `--objective both` to generate vanilla and FaceSM outputs. Use base attack names such as `PGD,MI_FGSM,BSR`; the runner adds `_SM` output labels automatically. `--source-lambda` defaults to `0.20`; zero disables source separation while retaining mirror fusion.

Vanilla keeps `<model>_adv_paths.csv`; FaceSM and combined runs write `<model>_facesm_adv_paths.csv` and `<model>_both_adv_paths.csv`. CSVs record the objective and source weight. Use separate output roots for different weights or reruns: the path CSV is rewritten on each run.

```python
adv = run_attack(
    'MI_FGSM', model, source, target, 'impersonation_attack', (112, 112),
    objective='facesm', source_lambda=0.20,
)
```

Inputs are single-pair NHWC tensors scaled to `[-1, 1]`. Source and target references are mirror-fused; the source reference is fixed throughout optimization. Model weights are not trained. Each optimized embedding requires two model evaluations, so an unchanged iteration count does not imply equal compute cost.

The CLI processes one pair at a time. IDAA is excluded because its implementation is missing; DYNAMIC_MORPH is available only with vanilla. Selecting an unsupported combination fails before model loading.

See [FaceSM integration notes](facesm.md) for paper reproduction limits.
