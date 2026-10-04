#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
import traceback
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from deepface import DeepFace
from PIL import Image

import facesm_attack_core as attack_core
from facesm_attack_core import (
    ATTACKER_MODELS,
    DECAY,
    EPSILON,
    equivalent_models,
    impact_value,
    parse_model_list_arg,
    resolve_image_path,
    threshold_for,
)
from ir152_model import IR_152
from evaluate_attack_performance import (
    compute_embedding,
    get_ir152_embedding,
    load_and_preprocess,
    load_ir152,
)

MODEL_INPUT_SIZES = {
    "Facenet": (160, 160),
    "Facenet512": (160, 160),
    "GhostFaceNet": (112, 112),
    "ArcFace": (112, 112),
    "VGG-Face": (224, 224),
}
BASE_ATTACKS = [
    "PGD",
    "MI_FGSM",
    "TI_FGSM",
    "SI_NI_FGSM",
    "MI_ADMIX_DI_TI",
    "RAP",
]
CONFIGS = [
    ("vanilla", "Vanilla", False, False),
    ("mf_only", "MF Only", True, False),
    ("ss_only", "SS Only", False, True),
    ("facesm", "FaceSM", True, True),
]
DEFAULT_ATTACK = "SI_NI_FGSM"
DEFAULT_ATTACKERS = ["Facenet512", "GhostFaceNet"]
DEFAULT_VICTIMS = ["IR152", "ArcFace", "VGG-Face"]
DEFAULT_SAMPLE_SIZE = 180
DEFAULT_OUT_ROOT = "results/ablation_sm_paper"

WORKER_CONFIG_KEY = "vanilla"
WORKER_CONFIG_LABEL = "Vanilla"
WORKER_USE_MIRROR = False
WORKER_USE_SOURCE = False


def atomic_write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{csv_path.stem}_", suffix=".tmp", dir=str(csv_path.parent))
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    else:
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").astype(int)
    return df


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    groups = [g.copy() for _, g in df.groupby(["dataset", "attack_type"], sort=True)]
    if not groups:
        raise ValueError("No groups available for sampling.")
    sample_size = min(int(sample_size), len(df))
    per_group = sample_size // len(groups)
    remainder = sample_size % len(groups)
    picked = []
    for idx, group in enumerate(groups):
        n = min(len(group), per_group + (1 if idx < remainder else 0))
        picked.append(group.sample(n=n, random_state=seed + idx, replace=False))
    out = pd.concat(picked, ignore_index=False).sort_values("row_id", kind="stable").reset_index(drop=True)
    return out


def load_or_create_sample(input_csv: Path, sample_csv: Path, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_csv.exists():
        return ensure_row_id(pd.read_csv(sample_csv))
    df = ensure_row_id(pd.read_csv(input_csv))
    sampled = stratified_sample(df, sample_size=sample_size, seed=seed)
    atomic_write_csv(sampled, sample_csv)
    return sampled


def load_thresholds(thresholds_json: Path) -> Dict[str, object]:
    with open(thresholds_json, "r", encoding="utf-8") as f:
        return json.load(f)


def build_victim_models(victims: List[str], ir152_weights: Path):
    cache = {}
    for victim in victims:
        if victim == "IR152":
            cache[victim] = load_ir152(str(ir152_weights))
        else:
            cache[victim] = DeepFace.build_model(victim).model
    return cache


def empty_adv_record(row: pd.Series, attacker_model: str, config_key: str, config_label: str, attack_name: str) -> Dict[str, object]:
    return {
        "row_id": int(row["row_id"]),
        "attacker_model": str(attacker_model),
        "config_key": str(config_key),
        "config_label": str(config_label),
        "attack_name": str(attack_name),
        "img1": str(row.get("img1", "")),
        "img2": str(row.get("img2", "")),
        "dataset": str(row.get("dataset", "")),
        "attack_type": str(row.get("attack_type", "")),
        "adv_path": "",
    }


def load_adv_map(csv_path: Path) -> Dict[Tuple[int, str, str], Dict[str, object]]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, rec in df.iterrows():
        key = (int(rec["row_id"]), str(rec["attacker_model"]), str(rec["config_key"]))
        out[key] = rec.to_dict()
    return out


def write_adv_map(csv_path: Path, adv_map: Dict[Tuple[int, str, str], Dict[str, object]]) -> None:
    cols = [
        "row_id",
        "attacker_model",
        "config_key",
        "config_label",
        "attack_name",
        "img1",
        "img2",
        "dataset",
        "attack_type",
        "adv_path",
    ]
    rows = sorted(
        adv_map.values(),
        key=lambda r: (str(r.get("attacker_model", "")), str(r.get("config_key", "")), int(r.get("row_id", -1))),
    )
    atomic_write_csv(pd.DataFrame(rows, columns=cols), csv_path)


def load_similarity_map(csv_path: Path) -> Dict[Tuple[int, str, str, str], Dict[str, object]]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, rec in df.iterrows():
        key = (
            int(rec["row_id"]),
            str(rec["attacker_model"]),
            str(rec["config_key"]),
            str(rec["victim_model"]),
        )
        out[key] = rec.to_dict()
    return out


def write_similarity_map(csv_path: Path, sim_map: Dict[Tuple[int, str, str, str], Dict[str, object]]) -> None:
    cols = [
        "row_id",
        "attacker_model",
        "config_key",
        "config_label",
        "victim_model",
        "attack_name",
        "img1",
        "img2",
        "dataset",
        "attack_type",
        "threshold",
        "clean_similarity",
        "adv_similarity",
        "breach",
        "impact",
        "adv_path",
    ]
    rows = sorted(
        sim_map.values(),
        key=lambda r: (
            str(r.get("attacker_model", "")),
            str(r.get("config_key", "")),
            str(r.get("victim_model", "")),
            int(r.get("row_id", -1)),
        ),
    )
    atomic_write_csv(pd.DataFrame(rows, columns=cols), csv_path)


def get_clean_context(
    row: pd.Series,
    victim_name: str,
    base_path: str,
    victim_models: Dict[str, object],
    clean_cache: Dict[Tuple[int, str], Dict[str, object]],
) -> Dict[str, object]:
    key = (int(row["row_id"]), str(victim_name))
    if key in clean_cache:
        return clean_cache[key]

    src_path = resolve_image_path(row.get("img1", ""), base_path)
    tgt_path = resolve_image_path(row.get("img2", ""), base_path)
    if victim_name == "IR152":
        tgt_emb = get_ir152_embedding(victim_models[victim_name], tgt_path)
        src_emb = get_ir152_embedding(victim_models[victim_name], src_path)
        clean_sim = float(np.dot(src_emb, tgt_emb))
    else:
        input_size = MODEL_INPUT_SIZES[victim_name]
        tgt_emb = compute_embedding(victim_models[victim_name], load_and_preprocess(tgt_path, input_size))
        src_emb = compute_embedding(victim_models[victim_name], load_and_preprocess(src_path, input_size))
        clean_sim = float(tf.reduce_sum(src_emb * tgt_emb).numpy())
    clean_cache[key] = {"target_emb": tgt_emb, "clean_similarity": clean_sim}
    return clean_cache[key]


def evaluate_adv_records(
    adv_records: List[Dict[str, object]],
    victims: List[str],
    thresholds: Dict[str, object],
    base_path: str,
    victim_models: Dict[str, object],
    clean_cache: Dict[Tuple[int, str], Dict[str, object]],
) -> List[Dict[str, object]]:
    sim_rows = []
    for rec in adv_records:
        adv_path = str(rec.get("adv_path", "")).strip()
        if not adv_path or not os.path.exists(adv_path):
            continue
        attacker = str(rec["attacker_model"])
        dataset_name = str(rec["dataset"])
        attack_type = str(rec["attack_type"])
        for victim_name in victims:
            if equivalent_models(attacker, victim_name):
                continue
            threshold = threshold_for(thresholds, victim_name, dataset_name)
            if threshold is None:
                continue
            clean_ctx = get_clean_context(pd.Series(rec), victim_name, base_path, victim_models, clean_cache)
            clean_sim = float(clean_ctx["clean_similarity"])
            if victim_name == "IR152":
                adv_emb = get_ir152_embedding(victim_models[victim_name], adv_path)
                adv_sim = float(np.dot(adv_emb, clean_ctx["target_emb"]))
            else:
                input_size = MODEL_INPUT_SIZES[victim_name]
                adv_emb = compute_embedding(victim_models[victim_name], load_and_preprocess(adv_path, input_size))
                adv_sim = float(tf.reduce_sum(adv_emb * clean_ctx["target_emb"]).numpy())
            breach = int(attack_core.success_from_threshold(adv_sim, threshold, attack_type))
            impact = float(impact_value(clean_sim, adv_sim, attack_type))
            sim_rows.append(
                {
                    "row_id": int(rec["row_id"]),
                    "attacker_model": attacker,
                    "config_key": str(rec["config_key"]),
                    "config_label": str(rec["config_label"]),
                    "victim_model": victim_name,
                    "attack_name": str(rec["attack_name"]),
                    "img1": str(rec["img1"]),
                    "img2": str(rec["img2"]),
                    "dataset": dataset_name,
                    "attack_type": attack_type,
                    "threshold": float(threshold),
                    "clean_similarity": clean_sim,
                    "adv_similarity": adv_sim,
                    "breach": breach,
                    "impact": impact,
                    "adv_path": adv_path,
                }
            )
    return sim_rows


def summarize_similarity_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["attacker_model", "config_key", "config_label", "n", "breach_rate", "impact_mean"])
    out = (
        df.groupby(["attacker_model", "config_key", "config_label"], as_index=False)
        .agg(n=("breach", "size"), breach_rate=("breach", "mean"), impact_mean=("impact", "mean"))
        .sort_values(["attacker_model", "config_key"], kind="stable")
        .reset_index(drop=True)
    )
    return out


def write_cumulative_summary(sim_map, summary_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame(sim_map.values())
    summary = summarize_similarity_df(df)
    atomic_write_csv(summary, summary_csv)
    return summary


def write_overall_summary(summary_df: pd.DataFrame, overall_csv: Path) -> pd.DataFrame:
    if summary_df.empty:
        overall = pd.DataFrame(columns=["config_key", "config_label", "breach_rate", "impact_mean"])
    else:
        overall = (
            summary_df.groupby(["config_key", "config_label"], as_index=False)
            .agg(breach_rate=("breach_rate", "mean"), impact_mean=("impact_mean", "mean"))
            .sort_values("config_key", kind="stable")
            .reset_index(drop=True)
        )
    atomic_write_csv(overall, overall_csv)
    return overall


def append_batch_summary(batch_summary_csv: Path, batch_df: pd.DataFrame) -> None:
    existing = pd.read_csv(batch_summary_csv) if batch_summary_csv.exists() else pd.DataFrame()
    out = pd.concat([existing, batch_df], ignore_index=True)
    atomic_write_csv(out, batch_summary_csv)


def plot_ablation(summary_df: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    overall = (
        summary_df.groupby(["config_key", "config_label"], as_index=False)
        .agg(breach_rate=("breach_rate", "mean"), impact_mean=("impact_mean", "mean"))
        .set_index("config_key")
        .loc[[cfg[0] for cfg in CONFIGS if cfg[0] in set(summary_df["config_key"])]]
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(overall["config_label"], overall["breach_rate"], color=["#777777", "#4C78A8", "#F58518", "#54A24B"][: len(overall)])
    for i, row in overall.iterrows():
        plt.text(i, row["breach_rate"] + 0.003, f"{row['breach_rate']:.3f}", ha="center", fontsize=9)
    plt.ylabel("Breach Rate")
    plt.title("Ablation: Breach Rate by Configuration")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(charts_dir / "ablation_breach_overall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(overall["config_label"], overall["impact_mean"], color=["#777777", "#4C78A8", "#F58518", "#54A24B"][: len(overall)])
    for i, row in overall.iterrows():
        plt.text(i, row["impact_mean"] + 0.0015, f"{row['impact_mean']:.4f}", ha="center", fontsize=9)
    plt.ylabel("Impact")
    plt.title("Ablation: Impact by Configuration")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(charts_dir / "ablation_impact_overall.png", dpi=180)
    plt.close()

    attackers = sorted(summary_df["attacker_model"].unique())
    if attackers:
        x = np.arange(len(CONFIGS))
        width = 0.8 / max(1, len(attackers))
        plt.figure(figsize=(9, 5.5))
        for idx, attacker in enumerate(attackers):
            grp = summary_df[summary_df["attacker_model"] == attacker].set_index("config_key")
            vals = [grp.loc[cfg[0], "breach_rate"] if cfg[0] in grp.index else np.nan for cfg in CONFIGS]
            labels = [cfg[1] for cfg in CONFIGS]
            pos = x - 0.4 + width / 2 + idx * width
            plt.bar(pos, vals, width=width, label=attacker)
        plt.xticks(x, [cfg[1] for cfg in CONFIGS], rotation=0)
        plt.ylabel("Breach Rate")
        plt.title("Ablation: Breach Rate by Configuration and Attacker")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts_dir / "ablation_breach_by_attacker.png", dpi=180)
        plt.close()


def config_by_key(config_key: str):
    for key, label, use_mirror, use_source in CONFIGS:
        if key == config_key:
            return key, label, use_mirror, use_source
    raise KeyError(config_key)


def save_adv_image(img, attack_name: str, config_key: str, src: str, tgt: str, attack_type: str, model_name: str, row_id: int) -> str:
    out_dir = os.path.join(attack_core.WORKER_ADV_ROOT, model_name, attack_name, config_key)
    os.makedirs(out_dir, exist_ok=True)
    s = os.path.splitext(os.path.basename(src))[0].replace(" ", "_")
    t = os.path.splitext(os.path.basename(tgt))[0].replace(" ", "_")
    rand = os.urandom(4).hex()
    name = f"adv_r{row_id}_{s}_to_{t}_{attack_type}_{config_key}_{rand}.png"
    path = os.path.join(out_dir, name)
    Image.fromarray(img).save(path)
    return os.path.abspath(path)


def embedding_for_cfg(model, x, use_mirror: bool):
    return attack_core.compute_embedding(model, x, multi_view=use_mirror)


def loss_for_cfg(model, x, src_emb, tgt_emb, attack_type: str, use_mirror: bool, use_source: bool, source_lambda: float):
    emb = embedding_for_cfg(model, x, use_mirror)
    cos_t = tf.reduce_sum(emb * tgt_emb, axis=1)
    if use_source:
        cos_s = tf.reduce_sum(emb * src_emb, axis=1)
        return attack_core.attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
    return attack_core.attack_loss(cos_t, attack_type)


def pgd_cfg(model, x, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda):
    if attack_core.WORKER_PGD_RANDOM_START:
        noise = tf.random.uniform(tf.shape(x), minval=-EPSILON, maxval=EPSILON, dtype=x.dtype)
        adv = tf.clip_by_value(x + noise, -1.0, 1.0)
    else:
        adv = tf.identity(x)
    alpha = EPSILON / attack_core.NUM_ITER
    for _ in range(attack_core.NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            loss = loss_for_cfg(model, adv, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_fgsm_cfg(model, x, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / attack_core.NUM_ITER
    for _ in range(attack_core.NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            loss = loss_for_cfg(model, adv, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def ti_fgsm_cfg(model, x, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda):
    adv = tf.identity(x)
    alpha = EPSILON / attack_core.NUM_ITER
    kernel = attack_core.gaussian_kernel()
    for _ in range(attack_core.NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            loss = loss_for_cfg(model, adv, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def si_ni_fgsm_cfg(model, x, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / attack_core.NUM_ITER
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    for _ in range(attack_core.NUM_ITER):
        nes = adv + DECAY * alpha * g
        grad_sum = tf.zeros_like(x)
        for s in scales:
            with tf.GradientTape() as tape:
                tape.watch(nes)
                loss = loss_for_cfg(model, nes * s, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
            grad_sum += tape.gradient(loss, nes)
        grad = grad_sum / len(scales)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_admix_di_ti_cfg(model, x, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda, pool_imgs, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / attack_core.NUM_ITER
    kernel = attack_core.gaussian_kernel()
    n_pool = tf.shape(pool_imgs)[0]
    for _ in range(attack_core.NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            idx = tf.random.uniform([3], 0, n_pool, dtype=tf.int32)
            others = tf.gather(pool_imgs, idx)
            adv_rep = tf.repeat(adv, 3, axis=0)
            mixed = adv_rep + 0.2 * (others - adv_rep)
            batch = attack_core.input_diversity(mixed, input_size)
            emb = embedding_for_cfg(model, batch, use_mirror)
            tgt_rep = tf.repeat(tgt_emb, 3, axis=0)
            cos_t = tf.reduce_sum(emb * tgt_rep, axis=1)
            if use_source:
                src_rep = tf.repeat(src_emb, 3, axis=0)
                cos_s = tf.reduce_sum(emb * src_rep, axis=1)
                loss = attack_core.attack_loss_sm(cos_t, cos_s, attack_type, source_lambda)
            else:
                loss = attack_core.attack_loss(cos_t, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def rap_cfg(model, x, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / attack_core.NUM_ITER
    kernel = attack_core.gaussian_kernel()
    for _ in range(attack_core.NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            delta = tf.zeros_like(adv)
            inner_alpha = (EPSILON / 2) / 3
            for _ in range(3):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(delta)
                    loss = loss_for_cfg(model, adv + delta, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
                grad_inner = inner_tape.gradient(loss, delta)
                delta += inner_alpha * tf.sign(grad_inner)
                delta = tf.clip_by_value(delta, -EPSILON / 2, EPSILON / 2)
            mixed = adv + 0.2 * tf.stop_gradient(delta)
            loss = loss_for_cfg(model, mixed, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], "SAME")
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def run_attack_cfg(attack_name, src, tgt, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda):
    model = attack_core.WORKER_MODEL
    if attack_name == "PGD":
        return pgd_cfg(model, src, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
    if attack_name == "MI_FGSM":
        return mi_fgsm_cfg(model, src, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
    if attack_name == "TI_FGSM":
        return ti_fgsm_cfg(model, src, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
    if attack_name == "SI_NI_FGSM":
        return si_ni_fgsm_cfg(model, src, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
    if attack_name == "MI_ADMIX_DI_TI":
        pool_imgs = tf.concat([src, tgt, src], axis=0)
        return mi_admix_di_ti_cfg(model, src, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda, pool_imgs, attack_core.WORKER_INPUT_SIZE)
    if attack_name == "RAP":
        return rap_cfg(model, src, src_emb, tgt_emb, attack_type, use_mirror, use_source, source_lambda)
    raise ValueError(f"Unsupported attack: {attack_name}")


def init_ablation_worker(model_name, input_size, base_path, adv_root, tf_threads, source_lambda, pgd_random_start, config_key):
    global WORKER_CONFIG_KEY, WORKER_CONFIG_LABEL, WORKER_USE_MIRROR, WORKER_USE_SOURCE
    attack_core.init_worker(model_name, input_size, base_path, adv_root, [], tf_threads, source_lambda, pgd_random_start)
    WORKER_CONFIG_KEY, WORKER_CONFIG_LABEL, WORKER_USE_MIRROR, WORKER_USE_SOURCE = config_by_key(config_key)


def process_ablation_row(payload):
    row_id, row, attack_name, config_key = payload
    key, label, use_mirror, use_source = config_by_key(config_key)
    out = {
        "row_id": int(row_id),
        "attacker_model": attack_core.WORKER_MODEL_NAME,
        "config_key": key,
        "config_label": label,
        "attack_name": attack_name,
        "img1": str(row.get("img1", "")),
        "img2": str(row.get("img2", "")),
        "dataset": str(row.get("dataset", "")),
        "attack_type": str(row.get("attack_type", "")),
        "adv_path": "",
    }
    try:
        src_path = resolve_image_path(row.get("img1", ""), attack_core.WORKER_BASE_PATH)
        tgt_path = resolve_image_path(row.get("img2", ""), attack_core.WORKER_BASE_PATH)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source image not found: {src_path}")
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Target image not found: {tgt_path}")

        src = tf.expand_dims(attack_core.load_and_preprocess(src_path, attack_core.WORKER_INPUT_SIZE), 0)
        tgt = tf.expand_dims(attack_core.load_and_preprocess(tgt_path, attack_core.WORKER_INPUT_SIZE), 0)
        attack_type = str(row.get("attack_type", ""))
        tgt_emb = embedding_for_cfg(attack_core.WORKER_MODEL, tgt, use_mirror)
        src_emb = embedding_for_cfg(attack_core.WORKER_MODEL, src, use_mirror if use_source or use_mirror else False)
        adv = run_attack_cfg(attack_name, src, tgt, src_emb, tgt_emb, attack_type, use_mirror, use_source, attack_core.WORKER_SOURCE_LAMBDA)
        adv_img = attack_core.denormalize(adv.numpy()[0])
        out["adv_path"] = save_adv_image(adv_img, attack_name, key, src_path, tgt_path, attack_type.lower(), attack_core.WORKER_MODEL_NAME, row_id)
    except Exception:
        traceback.print_exc()
    return out


def process_ablation_batch(batch_rows):
    return [process_ablation_row(item) for item in batch_rows]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumable ablation for source separation and mirror fusion without changing main attack code.")
    p.add_argument("--input-csv", default="input2400.csv")
    p.add_argument("--base-path", default="dataset_extractedfaces")
    p.add_argument("--thresholds-json", default="verification_thresholds.json")
    p.add_argument("--ir152-weights", default="ir152.pth")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--attack", default=DEFAULT_ATTACK, choices=BASE_ATTACKS)
    p.add_argument("--attackers", default=",".join(DEFAULT_ATTACKERS))
    p.add_argument("--victims", default=",".join(DEFAULT_VICTIMS))
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-lambda", type=float, default=0.20)
    p.add_argument("--num-iter", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--threads", type=int, default=min(24, cpu_count()))
    p.add_argument("--tf-threads", type=int, default=1)
    p.add_argument("--print-every-batches", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    attack_core.NUM_ITER = max(1, int(args.num_iter))
    input_csv = Path(args.input_csv).resolve()
    base_path = Path(args.base_path).resolve()
    thresholds_json = Path(args.thresholds_json).resolve()
    ir152_weights = Path(args.ir152_weights).resolve()
    out_root = Path(args.out_root).resolve()
    charts_dir = out_root / "charts"
    sample_csv = out_root / "sample_rows.csv"
    adv_csv = out_root / "ablation_adv_paths.csv"
    sim_csv = out_root / "ablation_similarity_scores.csv"
    batch_summary_csv = out_root / "ablation_batch_summary.csv"
    cumulative_summary_csv = out_root / "ablation_by_attacker_summary.csv"
    overall_summary_csv = out_root / "ablation_overall_summary.csv"

    attackers = parse_model_list_arg(args.attackers, ATTACKER_MODELS.keys())
    victims = [v.strip() for v in str(args.victims).split(",") if v.strip()]
    victims = [v for v in victims if v in list(MODEL_INPUT_SIZES.keys()) + ["IR152"]]
    if not victims:
        raise SystemExit("No valid victims selected.")

    sample_df = load_or_create_sample(input_csv, sample_csv, args.sample_size, args.seed)
    thresholds = load_thresholds(thresholds_json)
    victim_models = build_victim_models(victims, ir152_weights)
    clean_cache: Dict[Tuple[int, str], Dict[str, object]] = {}
    adv_map = load_adv_map(adv_csv)
    sim_map = load_similarity_map(sim_csv)

    print(f"[config] attack={args.attack} attackers={attackers} victims={victims}")
    print(f"[config] sample_rows={len(sample_df)} source_lambda={args.source_lambda} num_iter={attack_core.NUM_ITER} batch_size={args.batch_size} threads={args.threads} tf_threads={args.tf_threads}")
    print(f"[config] out_root={out_root}")

    for attacker in attackers:
        input_size = ATTACKER_MODELS[attacker]
        for config_key, config_label, _, _ in CONFIGS:
            pending_payloads = []
            for _, row in sample_df.iterrows():
                key = (int(row["row_id"]), attacker, config_key)
                existing = adv_map.get(key)
                if existing and str(existing.get("adv_path", "")).strip() and os.path.exists(str(existing.get("adv_path", "")).strip()):
                    continue
                pending_payloads.append((int(row["row_id"]), row, args.attack, config_key))

            if not pending_payloads:
                print(f"[resume] attacker={attacker} config={config_label} pending=0")
                continue

            row_batches = [pending_payloads[i : i + max(1, int(args.batch_size))] for i in range(0, len(pending_payloads), max(1, int(args.batch_size)))]
            config_adv_root = out_root / "adv_images_ablation"
            print(f"[run] attacker={attacker} config={config_label} pending_rows={len(pending_payloads)} batches={len(row_batches)}")
            ctx = get_context("spawn")
            with ctx.Pool(
                processes=max(1, int(args.threads)),
                initializer=init_ablation_worker,
                initargs=(
                    attacker,
                    input_size,
                    str(base_path),
                    str(config_adv_root),
                    max(1, int(args.tf_threads)),
                    float(args.source_lambda),
                    True,
                    config_key,
                ),
            ) as pool:
                for batch_idx, records in enumerate(pool.imap_unordered(process_ablation_batch, row_batches), start=1):
                    batch_adv_rows = []
                    for rec in records:
                        adv_path = str(rec.get("adv_path", "")).strip()
                        if not adv_path:
                            continue
                        row_id = int(rec["row_id"])
                        key = (row_id, attacker, config_key)
                        adv_row = {
                            "row_id": row_id,
                            "attacker_model": attacker,
                            "config_key": config_key,
                            "config_label": config_label,
                            "attack_name": args.attack,
                            "img1": str(rec.get("img1", "")),
                            "img2": str(rec.get("img2", "")),
                            "dataset": str(rec.get("dataset", "")),
                            "attack_type": str(rec.get("attack_type", "")),
                            "adv_path": adv_path,
                        }
                        adv_map[key] = adv_row
                        batch_adv_rows.append(adv_row)
                    write_adv_map(adv_csv, adv_map)

                    batch_sim_rows = evaluate_adv_records(
                        batch_adv_rows,
                        victims=victims,
                        thresholds=thresholds,
                        base_path=str(base_path),
                        victim_models=victim_models,
                        clean_cache=clean_cache,
                    )
                    for row in batch_sim_rows:
                        sim_key = (
                            int(row["row_id"]),
                            str(row["attacker_model"]),
                            str(row["config_key"]),
                            str(row["victim_model"]),
                        )
                        sim_map[sim_key] = row
                    write_similarity_map(sim_csv, sim_map)

                    batch_summary = summarize_similarity_df(pd.DataFrame(batch_sim_rows))
                    if not batch_summary.empty:
                        batch_summary.insert(0, "batch_index", batch_idx)
                        append_batch_summary(batch_summary_csv, batch_summary)
                    cumulative_summary = write_cumulative_summary(sim_map, cumulative_summary_csv)
                    overall_summary = write_overall_summary(cumulative_summary, overall_summary_csv)
                    plot_ablation(cumulative_summary, charts_dir)

                    if batch_idx % max(1, int(args.print_every_batches)) == 0:
                        print(f"\n[batch] attacker={attacker} config={config_label} batch={batch_idx}/{len(row_batches)} generated={len(batch_adv_rows)} similarity_rows={len(batch_sim_rows)}")
                        if not batch_summary.empty:
                            print("[batch summary]")
                            print(batch_summary.to_string(index=False))
                        print("[overall summary]")
                        print(overall_summary.to_string(index=False))

    final_by_attacker = write_cumulative_summary(sim_map, cumulative_summary_csv)
    final_overall = write_overall_summary(final_by_attacker, overall_summary_csv)
    plot_ablation(final_by_attacker, charts_dir)
    print("\n[done] final overall summary")
    print(final_overall.to_string(index=False))
    print(f"[done] adv_csv={adv_csv}")
    print(f"[done] similarity_csv={sim_csv}")
    print(f"[done] by_attacker_summary_csv={cumulative_summary_csv}")
    print(f"[done] overall_summary_csv={overall_summary_csv}")
    print(f"[done] charts_dir={charts_dir}")


if __name__ == "__main__":
    main()
