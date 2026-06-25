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

## Repository layout
- `core/transfer_attack_core.py`: shared attack implementations and registry
- `experiments/generate_adversarial_examples.py`: generic batch generation script
- `attacks/`: per-attack notes, references, contributor information, and student slide PDFs
- `docs/usage.md`: how to run the code
- `CONTRIBUTORS.md`: attack-wise contributor list

## Implemented attacks
- `PGD`
- `MI_FGSM`
- `TI_FGSM`
- `SI_NI_FGSM`
- `MI_ADMIX_DI_TI`
- `BPA_CNN`
- `BSR`
- `DECOWA`
- `SIA_MI_TI`
- `OPS`
- `ATT_CNN`
- `LI_BOOST_MI`
- `DPA_HMA`
- `DPA_HMA_ENSEMBLE`

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
