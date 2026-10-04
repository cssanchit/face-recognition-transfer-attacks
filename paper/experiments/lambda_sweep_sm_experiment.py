#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from deepface import DeepFace

import facesm_attack_core as attack_core
from facesm_attack_core import (
    ATTACKER_MODELS,
    ATTACK_COLS,
    equivalent_models,
    impact_value,
    init_worker,
    parse_model_list_arg,
    process_batch,
    threshold_for,
)
from ir152_model import IR_152
from evaluate_attack_performance import (
    IR152_INPUT_SIZE,
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
SM_ATTACKS = [
    "PGD_SM",
    "MI_FGSM_SM",
    "TI_FGSM_SM",
    "SI_NI_FGSM_SM",
    "MI_ADMIX_DI_TI_SM",
    "RAP_SM",
]
DEFAULT_LAMBDAS = [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.55, 0.75]
DEFAULT_ATTACKERS = ["Facenet512", "GhostFaceNet"]
DEFAULT_VICTIMS = ["IR152", "ArcFace", "VGG-Face"]
DEFAULT_SAMPLE_SIZE = 180
DEFAULT_OUT_ROOT = "results/lambda_sweep_sm"


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


def normalize_lambda(value: float) -> float:
    return round(float(value), 6)


def lambda_label(value: float) -> str:
    text = f"{normalize_lambda(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def parse_float_list(raw: str) -> List[float]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(normalize_lambda(float(part)))
    return values


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
        df = pd.read_csv(sample_csv)
        return ensure_row_id(df)
    df = ensure_row_id(pd.read_csv(input_csv))
    sampled = stratified_sample(df, sample_size=sample_size, seed=seed)
    atomic_write_csv(sampled, sample_csv)
    return sampled


def empty_adv_record(row: pd.Series, attacker_model: str, lambda_value: float, attack_name: str) -> Dict[str, object]:
    return {
        "row_id": int(row["row_id"]),
        "attacker_model": str(attacker_model),
        "lambda_value": float(normalize_lambda(lambda_value)),
        "lambda_label": lambda_label(lambda_value),
        "attack_name": str(attack_name),
        "img1": str(row.get("img1", "")),
        "img2": str(row.get("img2", "")),
        "dataset": str(row.get("dataset", "")),
        "attack_type": str(row.get("attack_type", "")),
        "adv_path": "",
    }


def load_adv_map(csv_path: Path) -> Dict[Tuple[int, str, float], Dict[str, object]]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, rec in df.iterrows():
        lam = normalize_lambda(rec.get("lambda_value", 0.0))
        key = (int(rec["row_id"]), str(rec["attacker_model"]), lam)
        out[key] = rec.to_dict()
    return out


def write_adv_map(csv_path: Path, adv_map: Dict[Tuple[int, str, float], Dict[str, object]]) -> None:
    cols = [
        "row_id",
        "attacker_model",
        "lambda_value",
        "lambda_label",
        "attack_name",
        "img1",
        "img2",
        "dataset",
        "attack_type",
        "adv_path",
    ]
    rows = sorted(
        adv_map.values(),
        key=lambda r: (str(r.get("attacker_model", "")), float(r.get("lambda_value", 0.0)), int(r.get("row_id", -1))),
    )
    atomic_write_csv(pd.DataFrame(rows, columns=cols), csv_path)


def load_similarity_map(csv_path: Path) -> Dict[Tuple[int, str, float, str], Dict[str, object]]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, rec in df.iterrows():
        lam = normalize_lambda(rec.get("lambda_value", 0.0))
        key = (
            int(rec["row_id"]),
            str(rec["attacker_model"]),
            lam,
            str(rec["victim_model"]),
        )
        out[key] = rec.to_dict()
    return out


def write_similarity_map(csv_path: Path, sim_map: Dict[Tuple[int, str, float, str], Dict[str, object]]) -> None:
    cols = [
        "row_id",
        "attacker_model",
        "lambda_value",
        "lambda_label",
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
            float(r.get("lambda_value", 0.0)),
            str(r.get("victim_model", "")),
            int(r.get("row_id", -1)),
        ),
    )
    atomic_write_csv(pd.DataFrame(rows, columns=cols), csv_path)


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

    src_path = attack_core.resolve_image_path(row.get("img1", ""), base_path)
    tgt_path = attack_core.resolve_image_path(row.get("img2", ""), base_path)
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
                    "lambda_value": float(rec["lambda_value"]),
                    "lambda_label": str(rec["lambda_label"]),
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
        return pd.DataFrame(columns=["attacker_model", "lambda_value", "lambda_label", "n", "breach_rate", "impact_mean"])
    out = (
        df.groupby(["attacker_model", "lambda_value", "lambda_label"], as_index=False)
        .agg(
            n=("breach", "size"),
            breach_rate=("breach", "mean"),
            impact_mean=("impact", "mean"),
        )
        .sort_values(["attacker_model", "lambda_value"], kind="stable")
        .reset_index(drop=True)
    )
    return out


def write_cumulative_summary(sim_map, summary_csv: Path) -> pd.DataFrame:
    df = pd.DataFrame(sim_map.values())
    summary = summarize_similarity_df(df)
    atomic_write_csv(summary, summary_csv)
    return summary


def append_batch_summary(batch_summary_csv: Path, batch_df: pd.DataFrame) -> None:
    existing = pd.read_csv(batch_summary_csv) if batch_summary_csv.exists() else pd.DataFrame()
    out = pd.concat([existing, batch_df], ignore_index=True)
    atomic_write_csv(out, batch_summary_csv)


def plot_breach_vs_lambda(summary_df: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    overall = (
        summary_df.groupby(["lambda_value", "lambda_label"], as_index=False)
        .agg(breach_rate=("breach_rate", "mean"))
        .sort_values("lambda_value", kind="stable")
    )
    plt.figure(figsize=(8, 5))
    plt.plot(overall["lambda_value"], overall["breach_rate"], marker="o", linewidth=2.0, color="#1f77b4")
    for _, row in overall.iterrows():
        plt.text(row["lambda_value"], row["breach_rate"] + 0.003, f"{row['breach_rate']:.3f}", ha="center", fontsize=8)
    plt.xlabel("Lambda")
    plt.ylabel("Breach Rate")
    plt.title("Overall Breach Rate vs Source-Separation Lambda")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(charts_dir / "breach_vs_lambda_overall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for attacker, grp in summary_df.groupby("attacker_model", sort=True):
        grp = grp.sort_values("lambda_value", kind="stable")
        plt.plot(grp["lambda_value"], grp["breach_rate"], marker="o", linewidth=2.0, label=attacker)
        for _, row in grp.iterrows():
            plt.text(row["lambda_value"], row["breach_rate"] + 0.003, f"{row['breach_rate']:.3f}", ha="center", fontsize=7)
    plt.xlabel("Lambda")
    plt.ylabel("Breach Rate")
    plt.title("Breach Rate vs Source-Separation Lambda by Attacker")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "breach_vs_lambda_by_attacker.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resumable lambda sweep for SM attacks using existing attack logic.")
    p.add_argument("--input-csv", default="input2400.csv")
    p.add_argument("--base-path", default="dataset_extractedfaces")
    p.add_argument("--thresholds-json", default="verification_thresholds.json")
    p.add_argument("--ir152-weights", default="ir152.pth")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--attack", default="SI_NI_FGSM_SM", choices=SM_ATTACKS)
    p.add_argument("--attackers", default=",".join(DEFAULT_ATTACKERS))
    p.add_argument("--victims", default=",".join(DEFAULT_VICTIMS))
    p.add_argument("--lambdas", default=",".join(str(v) for v in DEFAULT_LAMBDAS))
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=42)
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
    adv_csv = out_root / "lambda_sweep_adv_paths.csv"
    sim_csv = out_root / "lambda_sweep_similarity_scores.csv"
    batch_summary_csv = out_root / "lambda_sweep_batch_summary.csv"
    cumulative_summary_csv = out_root / "lambda_sweep_cumulative_summary.csv"

    lambdas = parse_float_list(args.lambdas)
    attackers = parse_model_list_arg(args.attackers, ATTACKER_MODELS.keys())
    victims = [v.strip() for v in str(args.victims).split(",") if v.strip()]
    victims = [v for v in victims if v in list(MODEL_INPUT_SIZES.keys()) + ["IR152"]]
    if not victims:
        raise SystemExit("No valid victims selected.")

    sample_df = load_or_create_sample(input_csv, sample_csv, args.sample_size, args.seed)
    thresholds = load_thresholds(thresholds_json)
    victim_models = build_victim_models(victims, ir152_weights)
    clean_cache: Dict[Tuple[int, str], Dict[str, object]] = {}
    adv_map: Dict[Tuple[int, str, float], Dict[str, object]] = load_adv_map(adv_csv)
    sim_map: Dict[Tuple[int, str, float, str], Dict[str, object]] = load_similarity_map(sim_csv)

    print(f"[config] attack={args.attack} attackers={attackers} victims={victims} lambdas={lambdas}")
    print(f"[config] sample_rows={len(sample_df)} num_iter={attack_core.NUM_ITER} batch_size={args.batch_size} threads={args.threads} tf_threads={args.tf_threads}")
    print(f"[config] out_root={out_root}")

    for attacker in attackers:
        input_size = ATTACKER_MODELS[attacker]
        for lambda_value in lambdas:
            lambda_key = normalize_lambda(lambda_value)
            lam_label = lambda_label(lambda_value)
            attack_col = ATTACK_COLS[args.attack]
            lambda_adv_root = out_root / "adv_images_lambda_sweep" / f"lambda_{lam_label}"
            pending_payloads = []
            for _, row in sample_df.iterrows():
                key = (int(row["row_id"]), attacker, lambda_key)
                existing = adv_map.get(key)
                if existing and str(existing.get("adv_path", "")).strip() and os.path.exists(str(existing.get("adv_path", "")).strip()):
                    continue
                existing_payload = {}
                if existing:
                    existing_payload[attack_col] = str(existing.get("adv_path", "")).strip()
                pending_payloads.append((int(row["row_id"]), row, [args.attack], existing_payload))

            if not pending_payloads:
                print(f"[resume] attacker={attacker} lambda={lam_label} pending=0")
                continue

            row_batches = [pending_payloads[i : i + max(1, int(args.batch_size))] for i in range(0, len(pending_payloads), max(1, int(args.batch_size)))]
            print(f"[run] attacker={attacker} lambda={lam_label} pending_rows={len(pending_payloads)} batches={len(row_batches)}")
            ctx = get_context("spawn")
            with ctx.Pool(
                processes=max(1, int(args.threads)),
                initializer=init_worker,
                initargs=(
                    attacker,
                    input_size,
                    str(base_path),
                    str(lambda_adv_root),
                    [args.attack],
                    max(1, int(args.tf_threads)),
                    float(lambda_value),
                    True,
                ),
            ) as pool:
                for batch_idx, records in enumerate(pool.imap_unordered(process_batch, row_batches), start=1):
                    batch_adv_rows: List[Dict[str, object]] = []
                    for rec in records:
                        adv_path = str(rec.get(attack_col, "")).strip()
                        if not adv_path:
                            continue
                        row_id = int(rec["row_id"])
                        key = (row_id, attacker, lambda_key)
                        adv_row = {
                            "row_id": row_id,
                            "attacker_model": attacker,
                            "lambda_value": float(lambda_value),
                            "lambda_label": lam_label,
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
                            normalize_lambda(row["lambda_value"]),
                            str(row["victim_model"]),
                        )
                        sim_map[sim_key] = row
                    write_similarity_map(sim_csv, sim_map)

                    batch_summary = summarize_similarity_df(pd.DataFrame(batch_sim_rows))
                    if not batch_summary.empty:
                        batch_summary.insert(0, "batch_index", batch_idx)
                        append_batch_summary(batch_summary_csv, batch_summary)
                    cumulative_summary = write_cumulative_summary(sim_map, cumulative_summary_csv)
                    plot_breach_vs_lambda(cumulative_summary, charts_dir)

                    if batch_idx % max(1, int(args.print_every_batches)) == 0:
                        print(f"\n[batch] attacker={attacker} lambda={lam_label} batch={batch_idx}/{len(row_batches)} generated={len(batch_adv_rows)} similarity_rows={len(batch_sim_rows)}")
                        if not batch_summary.empty:
                            print("[batch summary]")
                            print(batch_summary.to_string(index=False))
                        print("[cumulative summary]")
                        print(cumulative_summary.to_string(index=False))

    final_summary = write_cumulative_summary(sim_map, cumulative_summary_csv)
    plot_breach_vs_lambda(final_summary, charts_dir)
    print("\n[done] final cumulative summary")
    print(final_summary.to_string(index=False))
    print(f"[done] adv_csv={adv_csv}")
    print(f"[done] similarity_csv={sim_csv}")
    print(f"[done] batch_summary_csv={batch_summary_csv}")
    print(f"[done] cumulative_summary_csv={cumulative_summary_csv}")
    print(f"[done] charts_dir={charts_dir}")


if __name__ == "__main__":
    main()
