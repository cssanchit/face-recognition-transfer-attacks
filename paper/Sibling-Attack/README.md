# Sibling-Attack + FaceSM

This repo contains two attack scripts:

- `attack.py`: vanilla Sibling-Attack
- `attack_facesm.py`: Sibling-Attack with the FaceSM objective

Both scripts read image pairs from `./datasets/pairs.csv`, generate adversarial
images, and write output paths back into the same CSV. `evaluate.py` then
computes breach rate and impact for the generated images.

For the anonymous FaceSM review package, prefer the wrapper:

```bash
python experiments/run_sibling_facesm_experiment.py \
  --weights-dir /path/to/extracted/weight \
  --dataset-root /path/to/dataset_extractedfaces \
  --dataset lfw_pairs \
  --limit 300
```

The wrapper maps local image paths, selects a fixed pair subset, supports CPU
smoke tests, and avoids editing the hard-coded Colab-style paths below.

## Requirements

Use a Python environment with a CUDA-enabled PyTorch install for the full
experiment. The wrapper also supports `--device cpu` for small smoke tests.

Install the Python packages used by the repo:

```bash
pip install opencv-python omegaconf pandas deepface numpy
```

Install PyTorch separately for your CUDA version from the official PyTorch
instructions. On Google Colab, PyTorch is usually already available.

Quick GPU check:

```python
import torch
print(torch.cuda.is_available())
```

This should print `True`.

## Model Weights

The model weights are provided as a zip file (weight.zip - https://drive.google.com/file/d/1Ug8g_0TlAV9YgIzQDgJH9iDAo_4Hs8Fd/view?usp=sharing). Before running either attack,
extract the weights zip and place the `.pth` files in:

```text
Sibling-Attack/models/
```

After extraction, the `models/` folder should contain:

- `models/ir152.pth`
- `models/ir152_ar.pth`

These are loaded by both attack scripts. You do not need to train them.

## 1. Prepare `datasets/pairs.csv`

Create or copy your pair list to:

```text
./datasets/pairs.csv
```

Required columns:

| column | meaning |
| --- | --- |
| `img1` | source/attacker image filename or path |
| `img2` | target/victim image filename or path |
| `dataset` | dataset key used for evaluation thresholds, for example `celeba_pairs` |
| `attack_type` | only `impersonation_attack` rows are supported |

If your original CSV contains other attack types, filter it first:

```python
import os
import pandas as pd

SOURCE_CSV = '/content/add_correct_path_to_your_pairs_csv.csv'

df = pd.read_csv(SOURCE_CSV)
mask = df['attack_type'].astype(str).str.strip().str.lower() == 'impersonation_attack'
df_imp = df[mask].reset_index(drop=True)

os.makedirs('./datasets', exist_ok=True)
df_imp.to_csv('./datasets/pairs.csv', index=False)
print(f"Saved {len(df_imp)} impersonation pairs")
```

Path note:

- If `img1` and `img2` contain only filenames or relative paths, set `IMG_DIR`
  in the scripts to the folder containing those files.
- If `img1` and `img2` already contain full absolute paths, set `IMG_DIR` to
  the correct image root anyway for compatibility, but the evaluation script
  will use absolute paths as-is.

## 2. Edit Paths Before Running

You must update the hard-coded paths for your system.

In `attack.py`, near the bottom:

```python
CSV_PATH = './datasets/pairs.csv'
IMG_DIR = '/content/add_correct_path_to_your_face_images'
```

In `attack_facesm.py`, near the bottom:

```python
CSV_PATH = './datasets/pairs.csv'
IMG_DIR = '/content/add_correct_path_to_your_face_images'
```

In `evaluate.py`, near the top:

```python
CSV_PATH = './datasets/pairs.csv'
THRESHOLD_PATH = '/content/add_correct_path_to_thresholds.json'
IMG_DIR = '/content/add_correct_path_to_your_face_images'
```

In `configs/config.yaml`, check the GPU id:

```yaml
attack:
  gpu: 0
```

`dataset_dir` and `dataset_txt` in `configs/config.yaml` are not used by the
current CSV-based attack runners, but keeping them correct is still helpful if
you reuse older helper functions from the original repo.

## 3. Run the Vanilla Attack

From the repo folder:

```bash
cd Sibling-Attack
python attack.py
```

This writes adversarial images to:

```text
./results_adv_images_vanilla/
```

It also adds or fills this column in `./datasets/pairs.csv`:

```text
sibling_attack_path
```

## 4. Run the FaceSM Attack

```bash
python attack_facesm.py
```

This writes adversarial images to:

```text
./results_adv_images_facesm/
```

It also adds or fills this column in `./datasets/pairs.csv`:

```text
sibling_attack_facesm
```

## 5. Prepare Thresholds for Evaluation

`evaluate.py` requires a JSON file with per-model, per-dataset thresholds.
The dataset keys must match the values in `df['dataset'].unique()` from
`./datasets/pairs.csv`.

Expected shape:

```json
{
  "VGG-Face": {
    "celeba_pairs": {
      "threshold": 0.5
    }
  },
  "Facenet512": {
    "celeba_pairs": {
      "threshold": 0.5
    }
  },
  "ArcFace": {
    "celeba_pairs": {
      "threshold": 0.5
    }
  }
}
```

If your CSV contains multiple datasets, include each one under every victim
model:

```json
{
  "VGG-Face": {
    "celeba_pairs": {
      "threshold": 0.5
    },
    "lfw_pairs": {
      "threshold": 0.5
    },
    "vggface2_pairs": {
      "threshold": 0.5
    }
  }
}
```

Use your calibrated FAR = 0.1% thresholds instead of the placeholder values.

## 6. Run Evaluation

After both attack columns are filled in `./datasets/pairs.csv`, run:

```bash
python evaluate.py
```

The evaluator:

- reads dataset keys from `df['dataset'].unique()`
- selects the matching threshold for each row's dataset
- evaluates `sibling_attack_path` and `sibling_attack_facesm`
- writes per-row scores, thresholds, breach labels, and impacts back to
  `./datasets/pairs.csv`
- prints a summary table

DeepFace may download victim model weights the first time you run evaluation,
so the first run may require internet access.

## Runtime Notes

- By default, each attack uses `200` outer PGD iterations and `5` inner
  iterations from `configs/config.yaml`.
- Runtime can be long for large CSV files.
- The attack scripts save the CSV after each pair, so completed rows are kept
  if the run stops partway through.
- Images should already be aligned/cropped to faces. Evaluation uses
  `detector_backend='skip'`. If your images are not aligned, edit
  `evaluate.py` and remove that argument so DeepFace performs detection.

## Troubleshooting

`torch.cuda.is_available()` is `False`

Install a CUDA-enabled PyTorch build or switch to a GPU runtime.

`FileNotFoundError` for images

Check `IMG_DIR` and the values in `img1` / `img2`. If the CSV has relative
filenames, `IMG_DIR` must point to their folder. If the CSV has absolute paths,
make sure those files exist on the current system.

Missing threshold error

Add the missing dataset key to the threshold JSON. The required keys are the
values printed from `df['dataset'].unique()`.

DeepFace face-detection or embedding errors

The evaluator assumes aligned face crops. For unaligned images, remove
`detector_backend='skip'` in `evaluate.py`.
