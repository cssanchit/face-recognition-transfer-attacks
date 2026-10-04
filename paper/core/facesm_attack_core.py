import argparse
import json
import os
import random
import shutil
import traceback
import uuid
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Force TensorFlow to remain CPU-only before importing tf/deepface.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from deepface import DeepFace

from adv_output_cleanup import cleanup_orphan_adv_images

ATTACKER_MODELS = {
    "Facenet": (160, 160),
    "Facenet512": (160, 160),
    "GhostFaceNet": (112, 112),
    "ArcFace": (112, 112),
    "VGG-Face": (224, 224),
}

ALL_ATTACKS = [
    "PGD",
    "MI_FGSM",
    "TI_FGSM",
    "SI_NI_FGSM",
    "MI_ADMIX_DI_TI",
    "RAP",
    "BPA_CNN",
    "BSR",
    "DECOWA",
    "PGD_SM",
    "MI_FGSM_SM",
    "TI_FGSM_SM",
    "MI_ADMIX_DI_TI_SM",
    "SI_NI_FGSM_SM",
    "RAP_SM",
    "BPA_CNN_SM",
    "BSR_SM",
    "DECOWA_SM",
]
ATTACK_PAIRS = [
    ("PGD", "PGD_SM"),
    ("MI_FGSM", "MI_FGSM_SM"),
    ("TI_FGSM", "TI_FGSM_SM"),
    ("SI_NI_FGSM", "SI_NI_FGSM_SM"),
    ("MI_ADMIX_DI_TI", "MI_ADMIX_DI_TI_SM"),
    ("RAP", "RAP_SM"),
    ("BPA_CNN", "BPA_CNN_SM"),
    ("BSR", "BSR_SM"),
    ("DECOWA", "DECOWA_SM"),
]
ATTACK_COLS = {
    "PGD": "pgd_path",
    "MI_FGSM": "mi_fgsm_path",
    "TI_FGSM": "ti_fgsm_path",
    "SI_NI_FGSM": "si_ni_fgsm_path",
    "MI_ADMIX_DI_TI": "mi_admix_di_ti_path",
    "RAP": "rap_path",
    "BPA_CNN": "bpa_cnn_path",
    "BSR": "bsr_path",
    "DECOWA": "decowa_path",
    "PGD_SM": "pgd_sm_path",
    "MI_FGSM_SM": "mi_fgsm_sm_path",
    "TI_FGSM_SM": "ti_fgsm_sm_path",
    "MI_ADMIX_DI_TI_SM": "mi_admix_di_ti_sm_path",
    "SI_NI_FGSM_SM": "si_ni_fgsm_sm_path",
    "RAP_SM": "rap_sm_path",
    "BPA_CNN_SM": "bpa_cnn_sm_path",
    "BSR_SM": "bsr_sm_path",
    "DECOWA_SM": "decowa_sm_path",
}
OUTPUT_COLUMNS = [
    "row_id",
    "attacker_model",
    "img1",
    "img2",
    "dataset",
    "attack_type",
    "pgd_path",
    "mi_fgsm_path",
    "ti_fgsm_path",
    "si_ni_fgsm_path",
    "mi_admix_di_ti_path",
    "rap_path",
    "bpa_cnn_path",
    "bsr_path",
    "decowa_path",
    "pgd_sm_path",
    "mi_fgsm_sm_path",
    "ti_fgsm_sm_path",
    "mi_admix_di_ti_sm_path",
    "si_ni_fgsm_sm_path",
    "rap_sm_path",
    "bpa_cnn_sm_path",
    "bsr_sm_path",
    "decowa_sm_path",
]
PERF_BASE_COLUMNS = [
    "row_id",
    "attacker_model",
    "victim_model",
    "dataset",
    "attack_type",
    "threshold",
    "clean_similarity",
]


def perf_prefix(attack_name):
    return ATTACK_COLS[attack_name].replace("_path", "")


def perf_metric_columns(attack_name):
    prefix = perf_prefix(attack_name)
    return {
        "adv_similarity": f"{prefix}_adv_similarity",
        "breach": f"{prefix}_breach",
        "impact": f"{prefix}_impact",
        "adv_image_path": f"{prefix}_adv_image_path",
    }


def build_perf_columns(attacks):
    cols = list(PERF_BASE_COLUMNS)
    for attack_name in attacks:
        metrics = perf_metric_columns(attack_name)
        cols.extend(
            [
                metrics["adv_similarity"],
                metrics["breach"],
                metrics["impact"],
                metrics["adv_image_path"],
            ]
        )
    return cols


PERF_COLUMNS = build_perf_columns(ALL_ATTACKS)

EPSILON = 0.062
NUM_ITER = 5
DECAY = 1.0
DECOWA_MESH = 3
DECOWA_NUM_WARPING = 20
DECOWA_NOISE_SCALE = 2.0
DECOWA_RHO = 0.01

WORKER_MODEL = None
WORKER_MODEL_NAME = ""
WORKER_INPUT_SIZE = (112, 112)
WORKER_BASE_PATH = ""
WORKER_ADV_ROOT = ""
WORKER_ATTACKS = []
WORKER_SOURCE_LAMBDA = 0.20
WORKER_PGD_RANDOM_START = True


def configure_cpu_runtime(tf_threads):
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    except Exception:
        pass


def ensure_output_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(csv_path, index=False)
        return
    df = pd.read_csv(csv_path)
    if list(df.columns) == OUTPUT_COLUMNS:
        return
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    tmp_path = f"{csv_path}.tmp"
    df[OUTPUT_COLUMNS].to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)


def ensure_perf_csv(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        pd.DataFrame(columns=PERF_COLUMNS).to_csv(csv_path, index=False)


def is_blank(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def empty_perf_row(row_id, attacker_model, victim_model, existing=None):
    row = {c: "" for c in PERF_COLUMNS}
    if existing:
        row.update(existing)
    row["row_id"] = int(row_id)
    row["attacker_model"] = str(attacker_model)
    row["victim_model"] = str(victim_model)
    return row


def perf_attack_complete(row, attack_name):
    metrics = perf_metric_columns(attack_name)
    required = [
        "threshold",
        "clean_similarity",
        metrics["adv_similarity"],
        metrics["breach"],
        metrics["impact"],
        metrics["adv_image_path"],
    ]
    return all(not is_blank(row.get(col, "")) for col in required)


def perf_attack_matches_path(row, attack_name, adv_path):
    metrics = perf_metric_columns(attack_name)
    stored = str(row.get(metrics["adv_image_path"], "")).strip()
    return stored == str(adv_path).strip()


def merge_perf_rows(perf_latest_map, perf_rows):
    changed = 0
    for row in perf_rows:
        key = (int(row["row_id"]), str(row["attacker_model"]), str(row["victim_model"]))
        perf_latest_map[key] = row
        changed += 1
    return changed


def write_perf_csv(csv_path, perf_latest_map):
    rows = sorted(
        perf_latest_map.values(),
        key=lambda r: (
            str(r.get("attacker_model", "")),
            str(r.get("victim_model", "")),
            int(r.get("row_id", -1)),
        ),
    )
    df = pd.DataFrame(rows, columns=PERF_COLUMNS)
    tmp_path = f"{csv_path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)


def recompute_perf_stats(perf_latest_map, attacks):
    stats = init_perf_stats(attacks)
    for row in perf_latest_map.values():
        for attack_name in attacks:
            if not perf_attack_complete(row, attack_name):
                continue
            metrics = perf_metric_columns(attack_name)
            add_perf_stat(
                stats,
                attack_name,
                int(row[metrics["breach"]]),
                float(row[metrics["impact"]]),
            )
    return stats


def legacy_perf_df_to_latest_map(df):
    perf_latest_map = {}
    for _, rec in df.iterrows():
        attack_name = str(rec.get("attack_name", "")).strip()
        if attack_name not in ATTACK_COLS:
            continue
        row_id = int(pd.to_numeric(rec.get("row_id"), errors="coerce"))
        attacker_model = str(rec.get("attacker_model", "")).strip()
        victim_model = str(rec.get("victim_model", "")).strip()
        key = (row_id, attacker_model, victim_model)
        row = empty_perf_row(
            row_id=row_id,
            attacker_model=attacker_model,
            victim_model=victim_model,
            existing=perf_latest_map.get(key),
        )
        row["dataset"] = str(rec.get("dataset", row.get("dataset", ""))).strip()
        row["attack_type"] = str(rec.get("attack_type", row.get("attack_type", ""))).strip()
        if not is_blank(rec.get("threshold", "")):
            row["threshold"] = float(rec["threshold"])
        if not is_blank(rec.get("clean_similarity", "")):
            row["clean_similarity"] = float(rec["clean_similarity"])
        metrics = perf_metric_columns(attack_name)
        if not is_blank(rec.get("adv_similarity", "")):
            row[metrics["adv_similarity"]] = float(rec["adv_similarity"])
        if not is_blank(rec.get("breach", "")):
            row[metrics["breach"]] = int(rec["breach"])
        if not is_blank(rec.get("impact", "")):
            row[metrics["impact"]] = float(rec["impact"])
        row[metrics["adv_image_path"]] = str(rec.get("adv_image_path", "")).strip()
        perf_latest_map[key] = row
    return perf_latest_map


def migrate_perf_csv_if_needed(perf_csv):
    if not os.path.exists(perf_csv) or os.path.getsize(perf_csv) == 0:
        return None
    df = pd.read_csv(perf_csv)
    if "attack_name" not in set(df.columns):
        return None
    perf_latest_map = legacy_perf_df_to_latest_map(df)
    backup_path = f"{perf_csv}.legacy_long.csv"
    if not os.path.exists(backup_path):
        shutil.copy2(perf_csv, backup_path)
    write_perf_csv(perf_csv, perf_latest_map)
    return backup_path


def load_processed_keys(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["row_id", "attacker_model"])
    return set((int(rid), str(model)) for rid, model in zip(df["row_id"], df["attacker_model"]))


def load_latest_output_map(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return {}
    df = pd.read_csv(csv_path)
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[OUTPUT_COLUMNS]
    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    df["attacker_model"] = df["attacker_model"].astype(str)
    df = df.drop_duplicates(subset=["row_id", "attacker_model"], keep="last")
    out = {}
    for _, rec in df.iterrows():
        key = (int(rec["row_id"]), str(rec["attacker_model"]))
        row = {}
        for c in OUTPUT_COLUMNS:
            val = rec[c]
            if pd.isna(val):
                val = ""
            row[c] = int(rec["row_id"]) if c == "row_id" else str(val)
        out[key] = row
    return out


def normalize_attack_type(attack_type):
    return str(attack_type).strip().lower()


def normalize_model_name(name):
    return str(name).strip().lower().replace("-", "").replace("_", "")


def equivalent_models(a, b):
    na = normalize_model_name(a)
    nb = normalize_model_name(b)
    if na == nb:
        return True
    facenet_group = {"facenet", "facenet512"}
    return na in facenet_group and nb in facenet_group


def is_impersonation_attack(attack_type):
    return normalize_attack_type(attack_type) == "impersonation_attack"


def success_from_threshold(sim, threshold, attack_type):
    if is_impersonation_attack(attack_type):
        return bool(sim >= threshold)
    return bool(sim < threshold)


def impact_value(clean_sim, adv_sim, attack_type):
    if is_impersonation_attack(attack_type):
        return float(adv_sim - clean_sim)
    return float(clean_sim - adv_sim)


def threshold_for(thresholds, model_name, dataset_name):
    obj = thresholds.get(str(model_name), {}).get(str(dataset_name), {})
    if "threshold" not in obj:
        return None
    return float(obj["threshold"])


def preprocess_uint8_for_model(img_uint8, input_size):
    arr = np.asarray(Image.fromarray(img_uint8).convert("RGB").resize(input_size)).astype(np.float32) / 255.0
    return ((arr - 0.5) * 2.0)[None, ...]


def init_perf_stats(attacks):
    return {a: {"success": 0, "total": 0, "impact_sum": 0.0} for a in attacks}


def add_perf_stat(stats, attack_name, breach, impact):
    if attack_name not in stats:
        stats[attack_name] = {"success": 0, "total": 0, "impact_sum": 0.0}
    stats[attack_name]["total"] += 1
    stats[attack_name]["success"] += int(breach)
    stats[attack_name]["impact_sum"] += float(impact)


def load_perf_state(perf_csv, attacks):
    perf_latest_map = {}
    if not os.path.exists(perf_csv) or os.path.getsize(perf_csv) == 0:
        return perf_latest_map, init_perf_stats(attacks)
    df = pd.read_csv(perf_csv)
    if "attack_name" in set(df.columns):
        perf_latest_map = legacy_perf_df_to_latest_map(df)
        return perf_latest_map, recompute_perf_stats(perf_latest_map, attacks)

    for col in PERF_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[PERF_COLUMNS]
    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype(int)
    df["attacker_model"] = df["attacker_model"].astype(str)
    df["victim_model"] = df["victim_model"].astype(str)
    df = df.drop_duplicates(subset=["row_id", "attacker_model", "victim_model"], keep="last")

    for _, rec in df.iterrows():
        key = (int(rec["row_id"]), str(rec["attacker_model"]), str(rec["victim_model"]))
        row = {}
        for col in PERF_COLUMNS:
            value = rec[col]
            row[col] = "" if is_blank(value) else value
        row["row_id"] = int(rec["row_id"])
        perf_latest_map[key] = row
    return perf_latest_map, recompute_perf_stats(perf_latest_map, attacks)


def mean_impact(stats, attack_name):
    t = int(stats.get(attack_name, {}).get("total", 0))
    return (float(stats[attack_name]["impact_sum"]) / t) if t else 0.0


def breach_rate(stats, attack_name):
    t = int(stats.get(attack_name, {}).get("total", 0))
    return (float(stats[attack_name]["success"]) / t) if t else 0.0


def print_pairwise_perf(stats, selected_attacks, prefix):
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    reset = "\033[0m"

    #print(prefix)
    for van, sm in ATTACK_PAIRS:
        if van not in selected_attacks or sm not in selected_attacks:
            continue
        van_t = int(stats.get(van, {}).get("total", 0))
        sm_t = int(stats.get(sm, {}).get("total", 0))
        van_s = int(stats.get(van, {}).get("success", 0))
        sm_s = int(stats.get(sm, {}).get("success", 0))
        van_br = breach_rate(stats, van)
        sm_br = breach_rate(stats, sm)
        van_im = mean_impact(stats, van)
        sm_im = mean_impact(stats, sm)

        if sm_br > van_br:
            winner = f"{green}{sm}{reset}"
        elif van_br > sm_br:
            winner = f"{red}{van}{reset}"
        else:
            if sm_im > van_im:
                winner = f"{green}{sm}{reset}"
            elif van_im > sm_im:
                winner = f"{red}{van}{reset}"
            else:
                winner = f"{yellow}TIE{reset}"

        print(
            f"  {sm} vs {van} | breach={sm_s}/{sm_t} ({sm_br:.4f}) vs {van_s}/{van_t} ({van_br:.4f}) | "
            f"delta_breach={sm_br - van_br:+.4f} delta_impact={sm_im - van_im:+.4f} | winner={winner}"
        )


def resolve_image_path(path, base_path):
    value = str(path)
    if os.path.exists(value):
        return value
    marker = "dataset_extractedfaces/"
    if marker in value:
        rel = value.split(marker, 1)[1]
        candidate = os.path.join(base_path, rel)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(base_path, value.lstrip("/"))


def load_and_preprocess(path, input_size):
    img = Image.open(path).convert("RGB").resize(input_size)
    arr = np.array(img).astype("float32") / 255.0
    return (arr - 0.5) * 2.0


def denormalize(x):
    x = (x + 1.0) / 2.0
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def compute_embedding(model, x, multi_view=False):
    out = model(x, training=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    emb = tf.nn.l2_normalize(out, axis=1)
    if not multi_view:
        return emb
    x_flip = tf.image.flip_left_right(x)
    out_flip = model(x_flip, training=False)
    if isinstance(out_flip, (tuple, list)):
        out_flip = out_flip[0]
    emb_flip = tf.nn.l2_normalize(out_flip, axis=1)
    return tf.nn.l2_normalize(0.5 * (emb + emb_flip), axis=1)


def attack_loss(cos, attack_type):
    normalized = str(attack_type).strip().lower()
    return tf.reduce_mean(cos if normalized == "impersonation_attack" else (1 - cos))


def attack_loss_sm(cos_t, cos_s, attack_type, source_lambda):
    normalized = str(attack_type).strip().lower()
    if normalized == "impersonation_attack":
        score = cos_t - source_lambda * cos_s
    else:
        score = (1 - cos_t) + source_lambda * (1 - cos_s)
    return tf.reduce_mean(score)


def save_adv(img, attack, src, tgt, attack_type, model_name, row_id):
    out_dir = os.path.join(WORKER_ADV_ROOT, model_name, attack)
    os.makedirs(out_dir, exist_ok=True)

    s = os.path.splitext(os.path.basename(src))[0].replace(" ", "_")
    t = os.path.splitext(os.path.basename(tgt))[0].replace(" ", "_")
    rand = uuid.uuid4().hex[:8]
    name = f"adv_r{row_id}_{s}_to_{t}_{attack_type}_{rand}.png"
    path = os.path.join(out_dir, name)

    Image.fromarray(img).save(path)
    return os.path.abspath(path)


def gaussian_kernel(k=15, sigma=3.0, ch=3):
    x = tf.range(-k // 2 + 1, k // 2 + 1, dtype=tf.float32)
    g = tf.exp(-tf.square(x) / (2 * sigma**2))
    g /= tf.reduce_sum(g)
    kernel = tf.tensordot(g, g, axes=0)
    kernel = kernel[:, :, None, None]
    return tf.tile(kernel, [1, 1, ch, 1])


def input_diversity(x, input_size, prob=0.7):
    if tf.random.uniform([]) > prob:
        return x
    img_size = input_size[0]
    rnd = tf.random.uniform([], int(0.9 * img_size), img_size, dtype=tf.int32)

    x_resized = tf.image.resize(x, (rnd, rnd))
    pad_total = img_size - rnd
    pad_top = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_bottom = pad_total - pad_top
    pad_left = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_right = pad_total - pad_left

    x_padded = tf.pad(x_resized, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.image.resize(x_padded, input_size)


def pgd_attack(model, x, tgt_emb, attack_type):
    if WORKER_PGD_RANDOM_START:
        noise = tf.random.uniform(tf.shape(x), minval=-EPSILON, maxval=EPSILON, dtype=x.dtype)
        adv = tf.clip_by_value(x + noise, -1.0, 1.0)
    else:
        adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def pgd_sm_attack(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda):
    if WORKER_PGD_RANDOM_START:
        noise = tf.random.uniform(tf.shape(x), minval=-EPSILON, maxval=EPSILON, dtype=x.dtype)
        adv = tf.clip_by_value(x + noise, -1.0, 1.0)
    else:
        adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv, multi_view=True)
            cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
            cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
            loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_fgsm_sm(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv, multi_view=True)
            cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
            cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
            loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def ti_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    kernel = gaussian_kernel()
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def ti_fgsm_sm(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda):
    adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    kernel = gaussian_kernel()
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv, multi_view=True)
            cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
            cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
            loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def si_ni_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)

    for _ in range(NUM_ITER):
        nes = adv + DECAY * alpha * g
        grad_sum = tf.zeros_like(x)
        for s in scales:
            with tf.GradientTape() as tape:
                tape.watch(nes)
                emb = compute_embedding(model, nes * s)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad_sum += tape.gradient(loss, nes)

        grad = grad_sum / len(scales)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def si_ni_fgsm_sm(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)

    for _ in range(NUM_ITER):
        nes = adv + DECAY * alpha * g
        grad_sum = tf.zeros_like(x)
        for s in scales:
            with tf.GradientTape() as tape:
                tape.watch(nes)
                emb = compute_embedding(model, nes * s, multi_view=True)
                cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
                cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
                loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
            grad_sum += tape.gradient(loss, nes)

        grad = grad_sum / len(scales)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_admix_di_ti(model, x, tgt_emb, attack_type, pool_imgs, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel()
    n_pool = tf.shape(pool_imgs)[0]

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)

            idx = tf.random.uniform([3], 0, n_pool, dtype=tf.int32)
            others = tf.gather(pool_imgs, idx)
            adv_rep = tf.repeat(adv, 3, axis=0)
            mixed = adv_rep + 0.2 * (others - adv_rep)
            batch = input_diversity(mixed, input_size)

            emb = compute_embedding(model, batch)
            tgt_rep = tf.repeat(tgt_emb, 3, axis=0)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)

        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_admix_di_ti_sm(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda, pool_imgs, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)
    kernel = gaussian_kernel()
    n_pool = tf.shape(pool_imgs)[0]

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)

            idx = tf.random.uniform([3], 0, n_pool, dtype=tf.int32)
            others = tf.gather(pool_imgs, idx)
            adv_rep = tf.repeat(adv, 3, axis=0)
            mixed = adv_rep + 0.2 * (others - adv_rep)
            batch = input_diversity(mixed, input_size)

            emb = compute_embedding(model, batch, multi_view=True)
            tgt_rep = tf.repeat(tgt_emb_mv, 3, axis=0)
            src_rep = tf.repeat(src_emb_mv, 3, axis=0)
            cos_t = tf.reduce_sum(emb * tgt_rep, axis=1)
            cos_s = tf.reduce_sum(emb * src_rep, axis=1)
            loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)

        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def rap_attack(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel()

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            delta = tf.zeros_like(adv)
            inner_alpha = (EPSILON / 2) / 3

            for _ in range(3):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(delta)
                    emb = compute_embedding(model, adv + delta)
                    cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                    loss = attack_loss(cos, attack_type)
                grad_inner = inner_tape.gradient(loss, delta)
                delta += inner_alpha * tf.sign(grad_inner)
                delta = tf.clip_by_value(delta, -EPSILON / 2, EPSILON / 2)

            mixed = adv + 0.2 * tf.stop_gradient(delta)
            emb = compute_embedding(model, mixed)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)

        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def rap_sm_attack(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)
    kernel = gaussian_kernel()

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            delta = tf.zeros_like(adv)
            inner_alpha = (EPSILON / 2) / 3

            for _ in range(3):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(delta)
                    emb = compute_embedding(model, adv + delta, multi_view=True)
                    cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
                    cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
                    loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
                grad_inner = inner_tape.gradient(loss, delta)
                delta += inner_alpha * tf.sign(grad_inner)
                delta = tf.clip_by_value(delta, -EPSILON / 2, EPSILON / 2)

            mixed = adv + 0.2 * tf.stop_gradient(delta)
            emb = compute_embedding(model, mixed, multi_view=True)
            cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
            cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
            loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)

        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def _bpa_smooth_gradient(grad):
    kernel = gaussian_kernel(k=5, sigma=1.0)
    temperature = 3.0
    scaled = temperature * grad
    sig = tf.sigmoid(scaled)
    silu_deriv = sig + scaled * sig * (1.0 - sig)
    grad = grad * silu_deriv
    return tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")


def bpa_cnn(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = _bpa_smooth_gradient(grad)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def bpa_cnn_sm(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv, multi_view=True)
            cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
            cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
            loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
        grad = tape.gradient(loss, adv)
        grad = _bpa_smooth_gradient(grad)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


_BSR_MIN_DIM = 4


def _bsr_get_lengths(total, num_block):
    rand = np.random.uniform(size=num_block).astype(np.float32)
    sizes = np.round(rand * total / rand.sum()).astype(np.int32)
    sizes = np.maximum(sizes, _BSR_MIN_DIM)
    while sizes.sum() > total:
        sizes[sizes.argmax()] -= 1
    while sizes.sum() < total:
        sizes[sizes.argmin()] += 1
    return sizes.tolist()


def _bsr_rotate(image, angle_rad):
    h_int = int(image.shape[1])
    w_int = int(image.shape[2])
    if h_int < _BSR_MIN_DIM or w_int < _BSR_MIN_DIM:
        return image
    cx, cy = float(w_int) / 2.0, float(h_int) / 2.0
    cos_a = tf.math.cos(-angle_rad)
    sin_a = tf.math.sin(-angle_rad)
    tx = cx - cx * cos_a + cy * sin_a
    ty = cy - cx * sin_a - cy * cos_a
    transform = tf.reshape(
        tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]), [1, 8]
    )
    out_shape = tf.cast(tf.stack([h_int, w_int]), dtype=tf.int32)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.cast(image, tf.float32),
        transforms=tf.cast(transform, tf.float32),
        output_shape=out_shape,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )


def _bsr_shuffle_rotate(x, num_block=2):
    h_val, w_val = int(x.shape[1]), int(x.shape[2])
    w_strips = tf.split(x, _bsr_get_lengths(w_val, num_block), axis=2)
    result_w = []
    for wi in np.random.permutation(num_block).tolist():
        h_strips = tf.split(w_strips[wi], _bsr_get_lengths(h_val, num_block), axis=1)
        result_h = []
        for hi in np.random.permutation(num_block).tolist():
            angle = tf.random.truncated_normal([], stddev=0.05)
            result_h.append(_bsr_rotate(h_strips[hi], angle))
        result_w.append(tf.concat(result_h, axis=1))
    return tf.concat(result_w, axis=2)


def bsr(model, x, tgt_emb, attack_type, num_copies=20, num_block=2):
    adv = tf.Variable(tf.identity(x), trainable=True, dtype=tf.float32)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            copies = [_bsr_shuffle_rotate(adv, num_block) for _ in range(num_copies)]
            x_batch = tf.concat(copies, axis=0)
            tgt_rep = tf.repeat(tgt_emb, num_copies, axis=0)
            emb = compute_embedding(model, x_batch)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv.assign(adv + alpha * tf.sign(g))
        adv.assign(tf.clip_by_value(adv, x - EPSILON, x + EPSILON))
        adv.assign(tf.clip_by_value(adv, -1.0, 1.0))
    return tf.identity(adv)


def bsr_sm(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda, num_copies=20, num_block=2):
    adv = tf.Variable(tf.identity(x), trainable=True, dtype=tf.float32)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            copies = [_bsr_shuffle_rotate(adv, num_block) for _ in range(num_copies)]
            x_batch = tf.concat(copies, axis=0)
            src_rep = tf.repeat(src_emb_mv, num_copies, axis=0)
            tgt_rep = tf.repeat(tgt_emb_mv, num_copies, axis=0)
            emb = compute_embedding(model, x_batch, multi_view=True)
            cos_t = tf.reduce_sum(emb * tgt_rep, axis=1)
            cos_s = tf.reduce_sum(emb * src_rep, axis=1)
            loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv.assign(adv + alpha * tf.sign(g))
        adv.assign(tf.clip_by_value(adv, x - EPSILON, x + EPSILON))
        adv.assign(tf.clip_by_value(adv, -1.0, 1.0))
    return tf.identity(adv)


def _decowa_grid_points_2d(width, height):
    a = tf.linspace(-1.0, 1.0, height)
    b = tf.linspace(-1.0, 1.0, width)
    xx, yy = tf.meshgrid(a, b, indexing="ij")
    pts = tf.stack([yy, xx], axis=-1)
    return tf.reshape(pts, [-1, 2])


def _decowa_noisy_grid(width, height, noise_map):
    grid = _decowa_grid_points_2d(width, height)
    mod = tf.pad(noise_map, [[1, 1], [1, 1], [0, 0]])
    return grid + tf.reshape(mod, [-1, 2])


def _decowa_K(x_val, y_val):
    eps = 1e-9
    d2 = tf.reduce_sum(tf.square(x_val[:, :, None, :] - y_val[:, None, :, :]), axis=-1)
    return d2 * tf.math.log(d2 + eps)


def _decowa_P(x_val):
    n_val = tf.shape(x_val)[0]
    k_val = tf.shape(x_val)[1]
    return tf.concat([tf.ones([n_val, k_val, 1]), x_val], axis=-1)


def _decowa_tps_coeffs(x_val, y_val):
    k_val = tf.shape(x_val)[1]
    n_val = tf.shape(x_val)[0]
    k_mat = _decowa_K(x_val, x_val)
    p_mat = _decowa_P(x_val)
    top = tf.concat([k_mat, p_mat], axis=-1)
    bottom = tf.concat([tf.transpose(p_mat, [0, 2, 1]), tf.zeros([n_val, 3, 3])], axis=-1)
    l_mat = tf.concat([top, bottom], axis=1)
    z_mat = tf.concat([y_val, tf.zeros([n_val, 3, 2])], axis=1)
    q_val = tf.linalg.solve(l_mat, z_mat)
    return q_val[:, :k_val], q_val[:, k_val:]


def _decowa_dense_grid(height, width):
    gx = tf.linspace(-1.0, 1.0, width)
    gy = tf.linspace(-1.0, 1.0, height)
    x0 = tf.tile(gx[None, None, :], [1, height, 1])
    y0 = tf.tile(gy[None, :, None], [1, 1, width])
    grid = tf.stack([x0, y0], axis=-1)
    return tf.reshape(grid, [1, height * width, 2])


def _decowa_tps_grid(x_val, y_val, height, width):
    w_coef, a_coef = _decowa_tps_coeffs(x_val, y_val)
    base = _decowa_dense_grid(height, width)
    u_mat = _decowa_K(base, x_val)
    p_mat = _decowa_P(base)
    grid = tf.matmul(p_mat, a_coef) + tf.matmul(u_mat, w_coef)
    return tf.reshape(grid, [1, height, width, 2])


def _decowa_grid_sample(img, grid):
    n_val = tf.shape(img)[0]
    height = tf.shape(img)[1]
    width = tf.shape(img)[2]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    x_val = grid[..., 0]
    y_val = grid[..., 1]
    ix = ((x_val + 1.0) * width_f - 1.0) / 2.0
    iy = ((y_val + 1.0) * height_f - 1.0) / 2.0
    ix0 = tf.floor(ix)
    iy0 = tf.floor(iy)
    wx1 = ix - ix0
    wy1 = iy - iy0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    def sample(ixc, iyc):
        in_x = tf.logical_and(ixc >= 0.0, ixc <= width_f - 1.0)
        in_y = tf.logical_and(iyc >= 0.0, iyc <= height_f - 1.0)
        mask = tf.cast(tf.logical_and(in_x, in_y), tf.float32)[..., None]
        xc = tf.clip_by_value(tf.cast(ixc, tf.int32), 0, width - 1)
        yc = tf.clip_by_value(tf.cast(iyc, tf.int32), 0, height - 1)
        bidx = tf.broadcast_to(tf.reshape(tf.range(n_val), [n_val, 1, 1]), tf.shape(xc))
        idx = tf.stack([bidx, yc, xc], axis=-1)
        return tf.gather_nd(img, idx) * mask

    v00 = sample(ix0, iy0)
    v01 = sample(ix0, iy0 + 1.0)
    v10 = sample(ix0 + 1.0, iy0)
    v11 = sample(ix0 + 1.0, iy0 + 1.0)
    return (
        v00 * (wx0 * wy0)[..., None]
        + v10 * (wx1 * wy0)[..., None]
        + v01 * (wx0 * wy1)[..., None]
        + v11 * (wx1 * wy1)[..., None]
    )


def _decowa_warp(adv, noise_map, height, width):
    x_val = _decowa_grid_points_2d(DECOWA_MESH, DECOWA_MESH)[None, ...]
    y_val = _decowa_noisy_grid(DECOWA_MESH, DECOWA_MESH, noise_map)[None, ...]
    grid = _decowa_tps_grid(x_val, y_val, height, width)
    grid = tf.tile(grid, [tf.shape(adv)[0], 1, 1, 1])
    return _decowa_grid_sample(adv, grid)


def _decowa_update_noise_map(model, adv, tgt_emb, attack_type, height, width):
    noise_map = (tf.random.uniform([DECOWA_MESH - 2, DECOWA_MESH - 2, 2]) - 0.5) * DECOWA_NOISE_SCALE
    with tf.GradientTape() as tape:
        tape.watch(noise_map)
        warped = _decowa_warp(adv, noise_map, height, width)
        emb = compute_embedding(model, warped)
        cos = tf.reduce_sum(emb * tgt_emb, axis=1)
        loss = attack_loss(cos, attack_type)
    grad = tape.gradient(loss, noise_map)
    if grad is None:
        return noise_map
    grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
    return noise_map - DECOWA_RHO * grad


def _decowa_update_noise_map_sm(model, adv, src_emb_mv, tgt_emb_mv, attack_type, source_lambda, height, width):
    noise_map = (tf.random.uniform([DECOWA_MESH - 2, DECOWA_MESH - 2, 2]) - 0.5) * DECOWA_NOISE_SCALE
    with tf.GradientTape() as tape:
        tape.watch(noise_map)
        warped = _decowa_warp(adv, noise_map, height, width)
        emb = compute_embedding(model, warped, multi_view=True)
        cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
        cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
        loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
    grad = tape.gradient(loss, noise_map)
    if grad is None:
        return noise_map
    grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
    return noise_map - DECOWA_RHO * grad


def decowa(model, x, tgt_emb, attack_type, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    height, width = int(input_size[1]), int(input_size[0])

    for _ in range(NUM_ITER):
        grads = tf.zeros_like(x)
        for _ in range(DECOWA_NUM_WARPING):
            noise_map = _decowa_update_noise_map(model, tf.stop_gradient(adv), tgt_emb, attack_type, height, width)
            with tf.GradientTape() as tape:
                tape.watch(adv)
                warped = _decowa_warp(adv, noise_map, height, width)
                emb = compute_embedding(model, warped)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad = tape.gradient(loss, adv)
            grads += tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grads = grads / DECOWA_NUM_WARPING
        grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)
        g = DECAY * g + grads
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def decowa_sm(model, x, src_emb_mv, tgt_emb_mv, attack_type, source_lambda, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    src_emb_mv = tf.nn.l2_normalize(src_emb_mv, axis=1)
    tgt_emb_mv = tf.nn.l2_normalize(tgt_emb_mv, axis=1)
    height, width = int(input_size[1]), int(input_size[0])

    for _ in range(NUM_ITER):
        grads = tf.zeros_like(x)
        for _ in range(DECOWA_NUM_WARPING):
            noise_map = _decowa_update_noise_map_sm(
                model,
                tf.stop_gradient(adv),
                src_emb_mv,
                tgt_emb_mv,
                attack_type,
                source_lambda,
                height,
                width,
            )
            with tf.GradientTape() as tape:
                tape.watch(adv)
                warped = _decowa_warp(adv, noise_map, height, width)
                emb = compute_embedding(model, warped, multi_view=True)
                cos_t = tf.reduce_sum(emb * tgt_emb_mv, axis=1)
                cos_s = tf.reduce_sum(emb * src_emb_mv, axis=1)
                loss = attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
            grad = tape.gradient(loss, adv)
            grads += tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grads = grads / DECOWA_NUM_WARPING
        grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)
        g = DECAY * g + grads
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def init_worker(model_name, input_size, base_path, adv_root, attacks, tf_threads, source_lambda, pgd_random_start):
    global WORKER_MODEL, WORKER_MODEL_NAME, WORKER_INPUT_SIZE, WORKER_BASE_PATH, WORKER_ADV_ROOT, WORKER_ATTACKS, WORKER_SOURCE_LAMBDA, WORKER_PGD_RANDOM_START
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    configure_cpu_runtime(tf_threads=tf_threads)
    WORKER_MODEL_NAME = model_name
    WORKER_INPUT_SIZE = input_size
    WORKER_BASE_PATH = base_path
    WORKER_ADV_ROOT = adv_root
    WORKER_ATTACKS = attacks
    WORKER_SOURCE_LAMBDA = float(source_lambda)
    WORKER_PGD_RANDOM_START = bool(pgd_random_start)
    WORKER_MODEL = DeepFace.build_model(model_name).model


def empty_output(row_id, row, existing=None):
    out = {
        "row_id": int(row_id),
        "attacker_model": WORKER_MODEL_NAME,
        "img1": str(row.get("img1", "")),
        "img2": str(row.get("img2", "")),
        "dataset": str(row.get("dataset", "")),
        "attack_type": str(row.get("attack_type", "")),
    }
    for col in ATTACK_COLS.values():
        out[col] = str((existing or {}).get(col, ""))
    return out


def run_attack(attack_name, src, tgt, tgt_emb, src_emb_mv, tgt_emb_mv, attack_type):
    if attack_name == "PGD":
        return pgd_attack(WORKER_MODEL, src, tgt_emb, attack_type)
    if attack_name == "PGD_SM":
        return pgd_sm_attack(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA)
    if attack_name == "MI_FGSM":
        return mi_fgsm(WORKER_MODEL, src, tgt_emb, attack_type)
    if attack_name == "MI_FGSM_SM":
        return mi_fgsm_sm(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA)
    if attack_name == "TI_FGSM":
        return ti_fgsm(WORKER_MODEL, src, tgt_emb, attack_type)
    if attack_name == "TI_FGSM_SM":
        return ti_fgsm_sm(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA)
    if attack_name == "SI_NI_FGSM":
        return si_ni_fgsm(WORKER_MODEL, src, tgt_emb, attack_type)
    if attack_name == "MI_ADMIX_DI_TI":
        pool_imgs = tf.concat([src, tgt, src], axis=0)
        return mi_admix_di_ti(WORKER_MODEL, src, tgt_emb, attack_type, pool_imgs, WORKER_INPUT_SIZE)
    if attack_name == "MI_ADMIX_DI_TI_SM":
        pool_imgs = tf.concat([src, tgt, src], axis=0)
        return mi_admix_di_ti_sm(
            WORKER_MODEL,
            src,
            src_emb_mv,
            tgt_emb_mv,
            attack_type,
            WORKER_SOURCE_LAMBDA,
            pool_imgs,
            WORKER_INPUT_SIZE,
        )
    if attack_name == "RAP":
        return rap_attack(WORKER_MODEL, src, tgt_emb, attack_type)
    if attack_name == "SI_NI_FGSM_SM":
        return si_ni_fgsm_sm(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA)
    if attack_name == "RAP_SM":
        return rap_sm_attack(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA)
    if attack_name == "BPA_CNN":
        return bpa_cnn(WORKER_MODEL, src, tgt_emb, attack_type)
    if attack_name == "BPA_CNN_SM":
        return bpa_cnn_sm(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA)
    if attack_name == "BSR":
        return bsr(WORKER_MODEL, src, tgt_emb, attack_type)
    if attack_name == "BSR_SM":
        return bsr_sm(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA)
    if attack_name == "DECOWA":
        return decowa(WORKER_MODEL, src, tgt_emb, attack_type, WORKER_INPUT_SIZE)
    if attack_name == "DECOWA_SM":
        return decowa_sm(WORKER_MODEL, src, src_emb_mv, tgt_emb_mv, attack_type, WORKER_SOURCE_LAMBDA, WORKER_INPUT_SIZE)
    raise ValueError(f"Unsupported attack: {attack_name}")


def process_row(payload):
    row_id, row, attacks_to_run, existing = payload
    out = empty_output(row_id, row, existing=existing)
    try:
        src_path = resolve_image_path(row.get("img1", ""), WORKER_BASE_PATH)
        tgt_path = resolve_image_path(row.get("img2", ""), WORKER_BASE_PATH)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source image not found: {src_path}")
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Target image not found: {tgt_path}")

        src = tf.expand_dims(load_and_preprocess(src_path, WORKER_INPUT_SIZE), 0)
        tgt = tf.expand_dims(load_and_preprocess(tgt_path, WORKER_INPUT_SIZE), 0)
        tgt_emb = compute_embedding(WORKER_MODEL, tgt, multi_view=False)
        need_sm = any(attack.endswith("_SM") for attack in attacks_to_run)
        src_emb_mv = None
        tgt_emb_mv = None
        if need_sm:
            src_emb_mv = compute_embedding(WORKER_MODEL, src, multi_view=True)
            tgt_emb_mv = compute_embedding(WORKER_MODEL, tgt, multi_view=True)
        attack_type = str(row.get("attack_type", ""))

        for attack in attacks_to_run:
            col = ATTACK_COLS[attack]
            try:
                adv = run_attack(attack, src, tgt, tgt_emb, src_emb_mv, tgt_emb_mv, attack_type)
                adv_img = denormalize(adv.numpy()[0])
                out[col] = save_adv(
                    adv_img,
                    attack,
                    src_path,
                    tgt_path,
                    attack_type.lower(),
                    WORKER_MODEL_NAME,
                    row_id,
                )
            except Exception:
                traceback.print_exc()
                out[col] = ""
    except Exception:
        traceback.print_exc()
    return out


def process_batch(batch_rows):
    return [process_row(item) for item in batch_rows]


def evaluate_batch_performance(
    records,
    attacker_model_name,
    thresholds,
    selected_attacks,
    perf_latest_map,
    victim_models,
    selected_victims,
    base_path,
):
    perf_rows = []
    for rec in records:
        row_id = int(rec["row_id"])
        dataset_name = str(rec.get("dataset", "")).strip()
        attack_type = str(rec.get("attack_type", "")).strip()

        src_path = resolve_image_path(rec.get("img1", ""), base_path)
        tgt_path = resolve_image_path(rec.get("img2", ""), base_path)
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            continue

        for victim_name in selected_victims:
            if equivalent_models(attacker_model_name, victim_name):
                continue
            threshold = threshold_for(thresholds, victim_name, dataset_name)
            if threshold is None:
                continue
            vic_size = ATTACKER_MODELS.get(victim_name)
            if vic_size is None:
                continue
            perf_key = (row_id, str(attacker_model_name), str(victim_name))
            existing_perf = perf_latest_map.get(perf_key)
            pending_attacks = []
            for attack_name in selected_attacks:
                col = ATTACK_COLS[attack_name]
                adv_path = str(rec.get(col, "")).strip()
                if not adv_path or not os.path.exists(adv_path):
                    continue
                if existing_perf and perf_attack_complete(existing_perf, attack_name) and perf_attack_matches_path(existing_perf, attack_name, adv_path):
                    continue
                pending_attacks.append((attack_name, adv_path))
            if not pending_attacks:
                continue
            if victim_name not in victim_models:
                try:
                    victim_models[victim_name] = DeepFace.build_model(victim_name).model
                except Exception:
                    victim_models[victim_name] = None
            vic_model = victim_models.get(victim_name)
            if vic_model is None:
                continue

            src = tf.convert_to_tensor(load_and_preprocess(src_path, vic_size)[None, ...], dtype=tf.float32)
            tgt = tf.convert_to_tensor(load_and_preprocess(tgt_path, vic_size)[None, ...], dtype=tf.float32)
            src_emb = compute_embedding(vic_model, src, multi_view=False)
            tgt_emb = compute_embedding(vic_model, tgt, multi_view=False)
            clean_sim = float(tf.reduce_sum(src_emb * tgt_emb, axis=1).numpy()[0])
            perf_row = empty_perf_row(
                row_id=row_id,
                attacker_model=str(attacker_model_name),
                victim_model=str(victim_name),
                existing=existing_perf,
            )
            perf_row["dataset"] = dataset_name
            perf_row["attack_type"] = attack_type
            perf_row["threshold"] = float(threshold)
            perf_row["clean_similarity"] = float(clean_sim)

            for attack_name, adv_path in pending_attacks:
                adv_img = np.asarray(Image.open(adv_path).convert("RGB"))
                adv = tf.convert_to_tensor(preprocess_uint8_for_model(adv_img, vic_size), dtype=tf.float32)
                adv_emb = compute_embedding(vic_model, adv, multi_view=False)
                adv_sim = float(tf.reduce_sum(adv_emb * tgt_emb, axis=1).numpy()[0])
                breach = int(success_from_threshold(adv_sim, threshold, attack_type))
                impact = float(impact_value(clean_sim, adv_sim, attack_type))
                metrics = perf_metric_columns(attack_name)
                perf_row[metrics["adv_similarity"]] = float(adv_sim)
                perf_row[metrics["breach"]] = int(breach)
                perf_row[metrics["impact"]] = float(impact)
                perf_row[metrics["adv_image_path"]] = adv_path
            perf_rows.append(perf_row)
    return perf_rows


def sync_perf_from_outputs(
    output_map,
    perf_csv,
    perf_latest_map,
    thresholds,
    selected_models,
    selected_attacks,
    selected_victims,
    victim_models,
    base_path,
    chunk_size,
):
    updated_rows = 0
    for model_name in selected_models:
        model_records = []
        for (row_id, attacker_model), rec in sorted(output_map.items(), key=lambda item: (str(item[0][1]), int(item[0][0]))):
            if str(attacker_model) != str(model_name):
                continue
            model_records.append(rec)
        if not model_records:
            continue
        for batch_records in chunks(model_records, max(1, int(chunk_size))):
            perf_rows = evaluate_batch_performance(
                records=batch_records,
                attacker_model_name=model_name,
                thresholds=thresholds,
                selected_attacks=selected_attacks,
                perf_latest_map=perf_latest_map,
                victim_models=victim_models,
                selected_victims=selected_victims,
                base_path=base_path,
            )
            if not perf_rows:
                continue
            updated_rows += merge_perf_rows(perf_latest_map, perf_rows)
    if updated_rows:
        write_perf_csv(perf_csv, perf_latest_map)
    return updated_rows


def chunks(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def parse_list_arg(raw_value, allowed):
    if not raw_value:
        return list(allowed)
    picked = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [item for item in picked if item not in allowed]
    if invalid:
        raise ValueError(f"Invalid values {invalid}. Allowed: {sorted(allowed)}")
    return picked


def canonicalize_model_token(token):
    raw = str(token).strip()
    key = normalize_model_name(raw)
    alias = {
        "arface": "ArcFace",
        "arcface": "ArcFace",
        "ghostfacenet": "GhostFaceNet",
        "facenet512": "Facenet512",
        "facenet": "Facenet",
        "vggface": "VGG-Face",
        "vgg-face": "VGG-Face",
    }
    if key in alias:
        return alias[key]
    return raw


def parse_model_list_arg(raw_value, allowed):
    if not raw_value:
        return list(allowed)
    allowed_set = set(allowed)
    picked = []
    invalid = []
    for item in [x.strip() for x in str(raw_value).split(",") if x.strip()]:
        canon = canonicalize_model_token(item)
        if canon in allowed_set:
            picked.append(canon)
        else:
            invalid.append(item)
    if invalid:
        raise ValueError(f"Invalid model values {invalid}. Allowed: {sorted(allowed)}")
    return picked


def detect_workers(requested_workers):
    total_cores = cpu_count() or 1
    if requested_workers <= 0:
        return max(1, total_cores - 1), total_cores
    return max(1, min(requested_workers, total_cores)), total_cores


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial face samples (vanilla + SM variants) in CPU-only batched multicore mode.")
    parser.add_argument("--input-csv", default="input2400.csv")
    parser.add_argument("--output-csv", default="transfer_adv_paths.csv")
    parser.add_argument(
        "--perf-csv",
        default="transfer_attack_performance.csv",
        help="Reserved for external performance sync tooling. attack_plus.py does not write it.",
    )
    parser.add_argument("--adv-root", default="adv_images")
    parser.add_argument("--base-path", default="dataset_extractedfaces")
    parser.add_argument("--thresholds-json", default="thresholds.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=28, help="Worker processes to use (default: 28)")
    parser.add_argument("--workers", type=int, default=0, help="Legacy alias. Used only when --threads <= 0")
    parser.add_argument("--tf-threads", type=int, default=1, help="TensorFlow threads per process")
    parser.add_argument("--models", default="", help="Legacy alias for --attackers")
    parser.add_argument("--attackers", default="Facenet512,ArcFace,GhostFaceNet")
    parser.add_argument("--victims", default="Facenet,Facenet512,GhostFaceNet,ArcFace,VGG-Face")
    parser.add_argument(
        "--attacks",
        default=",".join(ALL_ATTACKS),
        help=f"Comma-separated from: {','.join(ALL_ATTACKS)}",
    )
    parser.add_argument("--num-iter", type=int, default=5, help="Attack iterations for vanilla and SM attacks")
    parser.add_argument("--source-lambda", type=float, default=0.20, help="Lambda for source-separation term in SM attacks")
    parser.add_argument(
        "--no-pgd-random-start",
        action="store_true",
        help="Disable random-start initialization for PGD and PGD_SM.",
    )
    parser.add_argument("--seed", type=int, default=2027, help="Random seed (default: 2027)")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of rows from input CSV for quick runs")
    parser.add_argument(
        "--cleanup-before-run",
        action="store_true",
        help="Remove orphan adversarial image files that are not referenced by the current output CSV before starting.",
    )
    parser.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        help="Report orphan files that would be removed by --cleanup-before-run without deleting them.",
    )
    parser.add_argument(
        "--print-every-batches",
        type=int,
        default=1,
        help="Print generation progress every N completed batches (default: 1).",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output CSV and process all rows again")
    return parser.parse_args()


def main():
    global NUM_ITER
    args = parse_args()
    NUM_ITER = max(1, int(args.num_iter))
    print_every_batches = max(1, int(args.print_every_batches))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))
    configure_cpu_runtime(tf_threads=max(1, args.tf_threads))

    attackers_raw = args.attackers if str(args.attackers).strip() else args.models
    selected_models = parse_model_list_arg(attackers_raw, ATTACKER_MODELS.keys())
    selected_attacks = parse_list_arg(args.attacks, ALL_ATTACKS)

    os.makedirs(args.adv_root, exist_ok=True)
    input_csv = os.path.abspath(args.input_csv)
    output_csv = os.path.abspath(args.output_csv)
    perf_csv = os.path.abspath(args.perf_csv)
    base_path = os.path.abspath(args.base_path)
    thresholds_json = os.path.abspath(args.thresholds_json)

    requested_workers = int(args.threads) if int(args.threads) > 0 else int(args.workers)
    workers, total_cores = detect_workers(requested_workers)
    pgd_random_start = not bool(args.no_pgd_random_start)
    print(
        f"CPU-only mode | total_cores={total_cores} | workers={workers} | "
        f"batch_size={args.batch_size} | tf_threads_per_worker={max(1, args.tf_threads)} | "
        f"num_iter={NUM_ITER} | pgd_random_start={pgd_random_start}"
    )
    print(f"Input CSV: {input_csv}")
    print(f"Base dataset path: {base_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Performance CSV (external sync): {perf_csv}")
    print(f"Thresholds JSON: {thresholds_json}")

    if args.cleanup_before_run:
        cleanup_stats = cleanup_orphan_adv_images(
            output_csv=output_csv,
            adv_root=args.adv_root,
            attack_cols=ATTACK_COLS,
            selected_models=selected_models,
            selected_attacks=selected_attacks,
            dry_run=bool(args.cleanup_dry_run),
        )
        mode = "dry-run" if cleanup_stats["dry_run"] else "applied"
        removed_mb = cleanup_stats["removed_bytes"] / (1024.0 * 1024.0)
        print(
            f"[cleanup:{mode}] scanned={cleanup_stats['scanned_files']} "
            f"removed_files={cleanup_stats['removed_files']} removed_dirs={cleanup_stats['removed_dirs']} "
            f"freed_mb={removed_mb:.2f}"
        )

    df = pd.read_csv(input_csv)
    if args.limit > 0:
        df = df.head(args.limit)

    rows = [(int(i), row.to_dict()) for i, row in df.iterrows()]

    if args.no_resume:
        for p in [output_csv]:
            if os.path.exists(p):
                os.remove(p)

    ensure_output_csv(output_csv)
    latest_output_map = {} if args.no_resume else load_latest_output_map(output_csv)

    for model_name in selected_models:
        input_size = ATTACKER_MODELS[model_name]

        pending = []
        for row_id, row in rows:
            existing = latest_output_map.get((row_id, model_name), {})
            attacks_to_run = []
            for attack_name in selected_attacks:
                col = ATTACK_COLS[attack_name]
                if str(existing.get(col, "")).strip() == "":
                    attacks_to_run.append(attack_name)
            if attacks_to_run:
                pending.append((row_id, row, attacks_to_run, existing))

        if not pending:
            print(f"[{model_name}] all selected attacks already present, skipping generation.")
            continue

        row_batches = list(chunks(pending, max(1, args.batch_size)))
        total_batches = len(row_batches)
        print(f"[{model_name}] pending_rows={len(pending)} | batches={total_batches}")

        with Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(
                model_name,
                input_size,
                base_path,
                args.adv_root,
                selected_attacks,
                max(1, args.tf_threads),
                args.source_lambda,
                pgd_random_start,
            ),
        ) as pool:
            for idx, records in enumerate(pool.imap_unordered(process_batch, row_batches), start=1):
                if records:
                    pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(
                        output_csv,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    for record in records:
                        latest_output_map[(int(record["row_id"]), model_name)] = record

                if idx % print_every_batches == 0 or idx == total_batches:
                    print(f"[{model_name}] completed_batches={idx}/{total_batches}")

    print("Done.")


if __name__ == "__main__":
    main()
