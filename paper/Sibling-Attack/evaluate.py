import os, json
import numpy as np
import pandas as pd
from deepface import DeepFace

# Paths to update before running.
CSV_PATH = './datasets/pairs.csv'
THRESHOLD_PATH = '/content/add_correct_path_to_thresholds.json'
IMG_DIR = '/content/add_correct_path_to_your_face_images'

# Set to an integer for a quick smoke test, or leave as None to evaluate all rows.
EVAL_LIMIT = None


def resolve_image_path(img_value):
    img_value = str(img_value)
    if os.path.isabs(img_value) or img_value.startswith('/'):
        return img_value
    return os.path.join(IMG_DIR, img_value)


def get_embedding(img_path, model_name):
    result = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=False,
        detector_backend='skip',
    )
    return np.array(result[0]['embedding'])


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def threshold_for_row(thresholds_by_dataset, row):
    dataset_key = str(row['dataset'])
    return thresholds_by_dataset[dataset_key]


def evaluate_column(df, adv_col, model_name, thresholds_by_dataset, src_embeds, tgt_embeds):
    """
    Impersonation:
      breach = s_adv >= threshold
      impact = s_adv - s_clean
    """
    breaches, impacts, s_clean_list, s_adv_list, threshold_list = [], [], [], [], []

    for idx, row in df.iterrows():
        adv_path = row.get(adv_col, '')
        threshold = threshold_for_row(thresholds_by_dataset, row)
        threshold_list.append(threshold)

        if pd.isna(adv_path) or str(adv_path).strip() == '':
            breaches.append(np.nan)
            impacts.append(np.nan)
            s_clean_list.append(np.nan)
            s_adv_list.append(np.nan)
            continue

        try:
            adv_emb = get_embedding(str(adv_path), model_name)
            s_clean = cosine_sim(src_embeds[idx], tgt_embeds[idx])
            s_adv = cosine_sim(adv_emb, tgt_embeds[idx])
            breaches.append(int(s_adv >= threshold))
            impacts.append(s_adv - s_clean)
            s_clean_list.append(s_clean)
            s_adv_list.append(s_adv)
        except Exception as e:
            print(f"  row {idx} [{adv_col}] failed: {e}")
            breaches.append(np.nan)
            impacts.append(np.nan)
            s_clean_list.append(np.nan)
            s_adv_list.append(np.nan)

    return breaches, impacts, s_clean_list, s_adv_list, threshold_list


df = pd.read_csv(CSV_PATH)
if EVAL_LIMIT is not None:
    df = df.iloc[:EVAL_LIMIT].copy()

if 'dataset' not in df.columns:
    raise ValueError("pairs CSV must include a 'dataset' column for per-dataset thresholds.")

dataset_keys = sorted(df['dataset'].dropna().astype(str).unique())
print(f"\nLoaded {len(df)} pairs")
print("Columns:", df.columns.tolist())
print("Datasets:", dataset_keys)

with open(THRESHOLD_PATH, 'r') as f:
    thresh_raw = json.load(f)

victim_model_names = ['VGG-Face', 'Facenet512', 'ArcFace']
victim_thresholds = {}
for model_name in victim_model_names:
    if model_name not in thresh_raw:
        raise KeyError(f"Missing model '{model_name}' in threshold file.")

    victim_thresholds[model_name] = {}
    for dataset_key in dataset_keys:
        try:
            victim_thresholds[model_name][dataset_key] = thresh_raw[model_name][dataset_key]['threshold']
        except KeyError as exc:
            raise KeyError(
                f"Missing threshold for model '{model_name}' and dataset '{dataset_key}'."
            ) from exc

print("Thresholds in use:")
for model_name, thresholds_by_dataset in victim_thresholds.items():
    joined = ', '.join(
        f"{dataset_key}: {thresholds_by_dataset[dataset_key]}"
        for dataset_key in dataset_keys
    )
    print(f"  {model_name}: {joined}")

for col in ['sibling_attack_path', 'sibling_attack_facesm']:
    n_filled = df[col].notna().sum() if col in df.columns else 0
    print(f"  {col}: {n_filled} / {len(df)} filled")

results = {}

for model_name, thresholds_by_dataset in victim_thresholds.items():
    print(f"\n{'=' * 60}")
    print(f"Victim: {model_name} | thresholds selected from each row's dataset")

    print("  Computing clean embeddings...")
    src_embeds, tgt_embeds = {}, {}
    for idx, row in df.iterrows():
        src_embeds[idx] = get_embedding(resolve_image_path(row['img1']), model_name)
        tgt_embeds[idx] = get_embedding(resolve_image_path(row['img2']), model_name)

    print("  Evaluating Sibling (vanilla)...")
    v_breach, v_impact, v_sclean, v_sadv, thresholds_used = evaluate_column(
        df, 'sibling_attack_path', model_name, thresholds_by_dataset, src_embeds, tgt_embeds)

    print("  Evaluating Sibling + FaceSM...")
    f_breach, f_impact, f_sclean, f_sadv, _ = evaluate_column(
        df, 'sibling_attack_facesm', model_name, thresholds_by_dataset, src_embeds, tgt_embeds)

    v_br = np.nanmean(v_breach) * 100
    f_br = np.nanmean(f_breach) * 100
    v_imp = np.nanmean(v_impact)
    f_imp = np.nanmean(f_impact)

    results[model_name] = {
        'vanilla_breach_%': round(v_br, 2),
        'facesm_breach_%': round(f_br, 2),
        'delta_breach_pp': round(f_br - v_br, 2),
        'vanilla_impact': round(v_imp, 4),
        'facesm_impact': round(f_imp, 4),
        'delta_impact': round(f_imp - v_imp, 4),
    }

    col_prefix = model_name.replace('-', '').replace(' ', '')
    df[f'{col_prefix}_threshold'] = thresholds_used
    df[f'{col_prefix}_s_clean'] = v_sclean
    df[f'{col_prefix}_vanilla_s_adv'] = v_sadv
    df[f'{col_prefix}_facesm_s_adv'] = f_sadv
    df[f'{col_prefix}_vanilla_breach'] = v_breach
    df[f'{col_prefix}_vanilla_impact'] = v_impact
    df[f'{col_prefix}_facesm_breach'] = f_breach
    df[f'{col_prefix}_facesm_impact'] = f_impact

df.to_csv(CSV_PATH, index=False)
print(f"\nAnnotated CSV saved to {CSV_PATH}")

print("\n" + "=" * 64)
print(f"{'Model':<14} {'V-B%':>6} {'SM-B%':>7} {'DeltaB':>8} {'V-Imp':>8} {'SM-Imp':>8} {'DeltaImp':>9}")
print("-" * 64)
for model_name, r in results.items():
    print(f"{model_name:<14} "
          f"{r['vanilla_breach_%']:>6.2f} "
          f"{r['facesm_breach_%']:>7.2f} "
          f"{r['delta_breach_pp']:>+8.2f} "
          f"{r['vanilla_impact']:>8.4f} "
          f"{r['facesm_impact']:>8.4f} "
          f"{r['delta_impact']:>+9.4f}")
print("=" * 64)

all_v_b = [r['vanilla_breach_%'] for r in results.values()]
all_f_b = [r['facesm_breach_%'] for r in results.values()]
all_v_i = [r['vanilla_impact'] for r in results.values()]
all_f_i = [r['facesm_impact'] for r in results.values()]
print(f"{'AVERAGE':<14} "
      f"{np.mean(all_v_b):>6.2f} "
      f"{np.mean(all_f_b):>7.2f} "
      f"{np.mean(all_f_b) - np.mean(all_v_b):>+8.2f} "
      f"{np.mean(all_v_i):>8.4f} "
      f"{np.mean(all_f_i):>8.4f} "
      f"{np.mean(all_f_i) - np.mean(all_v_i):>+9.4f}")
print("=" * 64)