#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile
import time
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
from deepface import DeepFace
from torchvision import transforms

from facesm_attack_core import (
    ALL_ATTACKS,
    ATTACKER_MODELS,
    ATTACK_COLS,
    PERF_COLUMNS,
    add_perf_stat,
    build_perf_columns,
    empty_perf_row,
    equivalent_models,
    impact_value,
    parse_list_arg,
    parse_model_list_arg,
    perf_metric_columns,
    print_pairwise_perf,
    threshold_for,
    write_perf_csv,
)
from ir152_model import IR_152

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MODEL_INPUT_SIZES: Dict[str, Tuple[int, int]] = {
    "Facenet": (160, 160),
    "Facenet512": (160, 160),
    "GhostFaceNet": (112, 112),
    "ArcFace": (112, 112),
    "VGG-Face": (224, 224),
}
ALL_SUPPORTED_VICTIMS = list(MODEL_INPUT_SIZES.keys()) + ["FaceAPI", "IR152"]

ATTACK_TO_PATH_COL = {attack: ATTACK_COLS[attack] for attack in ALL_ATTACKS}

FACEAPI_URL = ""
FACEAPI_INPUT_SIZE = (160, 160)
FACEAPI_JPEG_QUALITY = 100
IR152_INPUT_SIZE = (112, 112)
IR152_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IR152_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

KEY_COLS = ["row_id", "attacker_model", "img1", "img2", "dataset", "attack_type"]
SIMILARITY_BASE_COLUMNS = KEY_COLS + list(ATTACK_TO_PATH_COL.values())

WORKER_VICTIM_NAME = ""
WORKER_BASE_PATH = ""
WORKER_ADV_BASE_PATH = ""
WORKER_ATTACKS: List[str] = []
WORKER_MODEL = None


def configure_cpu_runtime(tf_threads: int = 1) -> None:
    tf_threads = max(1, int(tf_threads))
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    except Exception:
        pass
    try:
        torch.set_num_threads(tf_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def norm_text(value) -> str:
    if is_blank(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def snapshot_input_csv(input_csv: str, max_retries: int = 3, sleep_seconds: float = 1.0) -> pd.DataFrame:
    last_error = None
    for _ in range(max(1, max_retries)):
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="similarity_sync_", suffix=".csv")
        os.close(tmp_fd)
        try:
            shutil.copy2(input_csv, tmp_path)
            return pd.read_csv(tmp_path, dtype={"row_id": str})
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    raise RuntimeError(f"Could not read a stable snapshot of {input_csv}: {last_error}")


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


def load_and_preprocess(image_path: str, input_size: Tuple[int, int]) -> tf.Tensor:
    img = Image.open(image_path).convert("RGB").resize(input_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - 0.5) * 2.0
    return tf.convert_to_tensor(arr)


def compute_embedding(model, image_tensor: tf.Tensor) -> tf.Tensor:
    if len(image_tensor.shape) == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)
    emb = model(image_tensor, training=False)
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    return tf.nn.l2_normalize(emb, axis=1)


def cosine_similarity(emb1: tf.Tensor, emb2: tf.Tensor) -> float:
    return float(tf.reduce_sum(emb1 * emb2).numpy())


def load_models(model_input_sizes: Dict[str, Tuple[int, int]]) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for model_name in model_input_sizes:
        print(f"[models] loading {model_name}")
        try:
            models[model_name] = DeepFace.build_model(model_name).model
        except Exception as exc:
            print(f"[models] error loading {model_name}: {exc}")
            models[model_name] = None
    return models


def load_ir152(weights_path: str):
    model = IR_152(IR152_INPUT_SIZE)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def get_ir152_embedding(model, image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = IR152_TRANSFORM(img).unsqueeze(0)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = F.normalize(out, p=2, dim=1)
    return out.cpu().numpy().squeeze()


def resolve_clean_path(csv_path: str, base_path: str) -> str:
    path = norm_text(csv_path)
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    remapped = remap_to_base_dir(path, base_path)
    if remapped and os.path.exists(remapped):
        return remapped

    wrong_prefixes = [
        "/content/face_module/dataset_extractedfaces/",
        "/content/face_module/dataset_extractedfaces",
    ]
    for prefix in wrong_prefixes:
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break

    remapped = remap_to_base_dir(path, base_path)
    if remapped and os.path.exists(remapped):
        return remapped

    return os.path.join(base_path, path.lstrip("/"))


def resolve_adv_path(csv_path: str, adv_base_path: str) -> str:
    path = norm_text(csv_path)
    if not path:
        return ""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    remapped = remap_with_markers(path, adv_base_path, ["adv_images_all12", "adv_images"])
    if remapped:
        return remapped
    return os.path.join(adv_base_path, path.lstrip("/"))


def unique_paths(paths: List[Path]) -> List[Path]:
    ordered: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)
    return ordered


def discover_project_root(input_csv: Path) -> Path:
    candidates = unique_paths([Path.cwd(), input_csv.parent, *list(input_csv.parents)])

    def score(root: Path) -> int:
        value = 0
        if (root / "dataset_extractedfaces").is_dir():
            value += 4
        if (root / "verification_thresholds.json").is_file():
            value += 3
        if (root / "ir152_model.py").is_file():
            value += 1
        if (root / "ir152.pth").is_file():
            value += 1
        if any(root.glob("adv_images*")):
            value += 1
        return value

    best = max(candidates, key=score)
    return best if score(best) > 0 else input_csv.parent.resolve()


def first_existing_dir(candidates: List[Path]) -> Optional[Path]:
    for candidate in unique_paths(candidates):
        if candidate.is_dir():
            return candidate
    return None


def first_existing_file(candidates: List[Path]) -> Optional[Path]:
    for candidate in unique_paths(candidates):
        if candidate.is_file():
            return candidate
    return None


def derive_similarity_csv(output_csv: Path) -> Path:
    stem = output_csv.stem
    if "performance" in stem:
        new_stem = stem.replace("performance", "similarity_scores")
    else:
        new_stem = f"{stem}_similarity_scores"
    return output_csv.with_name(f"{new_stem}{output_csv.suffix or '.csv'}")


def remap_to_base_dir(path: str, base_dir: str) -> str:
    norm = norm_text(path).replace("\\", "/")
    if not norm:
        return ""
    base_name = Path(base_dir).name
    token = f"/{base_name}/"
    if token in norm:
        tail = norm.split(token, 1)[1]
        return str(Path(base_dir) / tail)
    if norm.startswith(base_name + "/"):
        tail = norm[len(base_name) + 1 :]
        return str(Path(base_dir) / tail)
    return ""


def remap_with_markers(path: str, root_dir: str, markers: List[str]) -> str:
    norm = norm_text(path).replace("\\", "/")
    if not norm:
        return ""
    root = Path(root_dir)
    for marker in markers:
        token = f"/{marker}/"
        if token in norm:
            tail = norm.split(token, 1)[1]
            return str(root / marker / tail)
        if norm.startswith(marker + "/"):
            return str(root / norm)
    return ""


def save_tmp_for_faceapi(image_path: str) -> Optional[str]:
    try:
        img = Image.open(image_path).convert("RGB").resize(FACEAPI_INPUT_SIZE)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, format="JPEG", quality=FACEAPI_JPEG_QUALITY)
        return tmp.name
    except Exception as exc:
        print(f"[FaceAPI] tmp save error: {exc}")
        return None


def faceapi_similarity(img1_path: str, img2_path: str, max_retries: int = 3) -> Optional[float]:
    tmp1 = None
    tmp2 = None
    try:
        tmp1 = save_tmp_for_faceapi(img1_path)
        tmp2 = save_tmp_for_faceapi(img2_path)
        if not tmp1 or not tmp2:
            return None

        for attempt in range(max_retries):
            try:
                with open(tmp1, "rb") as f1, open(tmp2, "rb") as f2:
                    response = requests.post(
                        FACEAPI_URL,
                        files={"image1": f1, "image2": f2},
                        timeout=30,
                    )
                if response.status_code == 200:
                    payload = response.json()
                    return payload.get("similarity")
            except requests.exceptions.RequestException:
                pass
            if attempt < max_retries - 1:
                time.sleep(1)
        return None
    finally:
        for tmp in [tmp1, tmp2]:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


def ensure_similarity_columns(
    df: pd.DataFrame,
    model_names: List[str],
    attacks: List[str],
    enable_faceapi: bool,
    enable_ir152: bool,
) -> pd.DataFrame:
    for col in SIMILARITY_BASE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for model in model_names:
        clean_col = f"{model}_clean"
        if clean_col not in df.columns:
            df[clean_col] = np.nan
        for attack in attacks:
            adv_col = f"{model}_{attack}_adv"
            if adv_col not in df.columns:
                df[adv_col] = np.nan

    if enable_faceapi:
        if "FaceAPI_clean" not in df.columns:
            df["FaceAPI_clean"] = np.nan
        for attack in attacks:
            col = f"FaceAPI_{attack}_adv"
            if col not in df.columns:
                df[col] = np.nan

    if enable_ir152:
        if "IR152_clean" not in df.columns:
            df["IR152_clean"] = np.nan
        for attack in attacks:
            col = f"IR152_{attack}_adv"
            if col not in df.columns:
                df[col] = np.nan

    return df


def row_key(row: pd.Series) -> Tuple[str, str, str, str, str, str]:
    return tuple(norm_text(row.get(c, "")) for c in KEY_COLS)  # type: ignore[return-value]


def consolidate_duplicate_rows(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    for c in key_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()

    dedup_rows = []
    for _, grp in df.groupby(key_cols, dropna=False, sort=False):
        base = grp.iloc[0].copy()
        for i in range(1, len(grp)):
            base = base.combine_first(grp.iloc[i])
            for col in grp.columns:
                if pd.isna(base[col]) or str(base[col]).strip() == "":
                    val = grp.iloc[i][col]
                    if not (pd.isna(val) or str(val).strip() == ""):
                        base[col] = val
        dedup_rows.append(base)
    return pd.DataFrame(dedup_rows)


def compute_row_updates(
    row: pd.Series,
    out_row: pd.Series,
    models: Dict[str, object],
    model_names: List[str],
    attacks: List[str],
    base_path: str,
    adv_base_path: str,
    enable_faceapi: bool,
    ir152_model,
    enable_ir152: bool,
) -> Dict[str, object]:
    updates: Dict[str, object] = {}

    img1_path = resolve_clean_path(str(row["img1"]), base_path)
    img2_path = resolve_clean_path(str(row["img2"]), base_path)
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return updates

    needs_clean = any(pd.isna(out_row.get(f"{m}_clean", np.nan)) for m in model_names)
    needs_adv = False
    for attack in attacks:
        path_col = ATTACK_TO_PATH_COL[attack]
        adv_csv_path = norm_text(row.get(path_col, ""))
        if not adv_csv_path:
            continue
        for model_name in model_names:
            if pd.isna(out_row.get(f"{model_name}_{attack}_adv", np.nan)):
                needs_adv = True
                break
        if needs_adv:
            break

    if enable_faceapi:
        if pd.isna(out_row.get("FaceAPI_clean", np.nan)):
            needs_clean = True
        for attack in attacks:
            if pd.isna(out_row.get(f"FaceAPI_{attack}_adv", np.nan)):
                needs_adv = True
                break

    if enable_ir152:
        if pd.isna(out_row.get("IR152_clean", np.nan)):
            needs_clean = True
        for attack in attacks:
            if pd.isna(out_row.get(f"IR152_{attack}_adv", np.nan)):
                needs_adv = True
                break

    if not (needs_clean or needs_adv):
        return updates

    emb_img2: Dict[str, Optional[tf.Tensor]] = {}
    if needs_clean or needs_adv:
        for model_name in model_names:
            model = models.get(model_name)
            if model is None:
                continue
            try:
                emb_img2[model_name] = compute_embedding(
                    model,
                    load_and_preprocess(img2_path, MODEL_INPUT_SIZES[model_name]),
                )
            except Exception:
                emb_img2[model_name] = None

    for model_name in model_names:
        model = models.get(model_name)
        if model is None:
            continue
        clean_col = f"{model_name}_clean"
        if pd.isna(out_row.get(clean_col, np.nan)):
            try:
                emb1 = compute_embedding(model, load_and_preprocess(img1_path, MODEL_INPUT_SIZES[model_name]))
                emb2 = emb_img2.get(model_name)
                if emb2 is not None:
                    updates[clean_col] = cosine_similarity(emb1, emb2)
            except Exception:
                pass

    if enable_faceapi and pd.isna(out_row.get("FaceAPI_clean", np.nan)):
        sim = faceapi_similarity(img1_path, img2_path)
        updates["FaceAPI_clean"] = np.nan if sim is None else sim

    emb_ir152_tgt = None
    if enable_ir152 and ir152_model is not None and pd.isna(out_row.get("IR152_clean", np.nan)):
        try:
            emb_ir152_tgt = get_ir152_embedding(ir152_model, img2_path)
            emb_ir152_src = get_ir152_embedding(ir152_model, img1_path)
            updates["IR152_clean"] = float(np.dot(emb_ir152_src, emb_ir152_tgt))
        except Exception:
            pass

    for attack in attacks:
        path_col = ATTACK_TO_PATH_COL[attack]
        adv_csv_path = norm_text(row.get(path_col, ""))
        if not adv_csv_path:
            continue
        adv_img_path = resolve_adv_path(adv_csv_path, adv_base_path)
        if not os.path.exists(adv_img_path):
            continue

        for model_name in model_names:
            col = f"{model_name}_{attack}_adv"
            if not pd.isna(out_row.get(col, np.nan)):
                continue
            model = models.get(model_name)
            emb2 = emb_img2.get(model_name)
            if model is None or emb2 is None:
                continue
            try:
                emb_adv = compute_embedding(model, load_and_preprocess(adv_img_path, MODEL_INPUT_SIZES[model_name]))
                updates[col] = cosine_similarity(emb_adv, emb2)
            except Exception:
                pass

        if enable_faceapi:
            col = f"FaceAPI_{attack}_adv"
            if pd.isna(out_row.get(col, np.nan)):
                sim = faceapi_similarity(adv_img_path, img2_path)
                updates[col] = np.nan if sim is None else sim

        if enable_ir152 and ir152_model is not None:
            col = f"IR152_{attack}_adv"
            if pd.isna(out_row.get(col, np.nan)):
                try:
                    if emb_ir152_tgt is None:
                        emb_ir152_tgt = get_ir152_embedding(ir152_model, img2_path)
                    emb_ir152_adv = get_ir152_embedding(ir152_model, adv_img_path)
                    updates[col] = float(np.dot(emb_ir152_adv, emb_ir152_tgt))
                except Exception:
                    pass

    return updates


def pending_for_victim(out_row: pd.Series, row_data: Dict[str, str], victim_name: str, attacks: List[str]) -> Tuple[bool, List[str]]:
    clean_col = f"{victim_name}_clean"
    need_clean = clean_col in out_row.index and pd.isna(out_row.get(clean_col, np.nan))
    pending_attacks: List[str] = []
    for attack in attacks:
        path_col = ATTACK_TO_PATH_COL[attack]
        if not norm_text(row_data.get(path_col, "")):
            continue
        adv_col = f"{victim_name}_{attack}_adv"
        if adv_col in out_row.index and pd.isna(out_row.get(adv_col, np.nan)):
            pending_attacks.append(attack)
    return need_clean, pending_attacks


def init_similarity_worker(
    victim_name: str,
    base_path: str,
    adv_base_path: str,
    attacks: List[str],
    tf_threads: int,
    ir152_weights: str,
) -> None:
    global WORKER_VICTIM_NAME, WORKER_BASE_PATH, WORKER_ADV_BASE_PATH, WORKER_ATTACKS, WORKER_MODEL
    configure_cpu_runtime(tf_threads=max(1, int(tf_threads)))
    WORKER_VICTIM_NAME = victim_name
    WORKER_BASE_PATH = base_path
    WORKER_ADV_BASE_PATH = adv_base_path
    WORKER_ATTACKS = list(attacks)
    WORKER_MODEL = None

    if victim_name in MODEL_INPUT_SIZES:
        WORKER_MODEL = DeepFace.build_model(victim_name).model
    elif victim_name == "IR152":
        WORKER_MODEL = load_ir152(ir152_weights)


def compute_victim_row_task(task: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
    out_i = int(task["out_i"])
    row_data = task["row_data"]
    need_clean = bool(task["need_clean"])
    pending_attacks = list(task["pending_attacks"])
    updates: Dict[str, object] = {}

    img1_path = resolve_clean_path(str(row_data["img1"]), WORKER_BASE_PATH)
    img2_path = resolve_clean_path(str(row_data["img2"]), WORKER_BASE_PATH)
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return out_i, updates

    victim_name = WORKER_VICTIM_NAME
    clean_col = f"{victim_name}_clean"

    if victim_name in MODEL_INPUT_SIZES:
        try:
            emb_tgt = compute_embedding(
                WORKER_MODEL,
                load_and_preprocess(img2_path, MODEL_INPUT_SIZES[victim_name]),
            )
        except Exception:
            return out_i, updates

        if need_clean:
            try:
                emb_src = compute_embedding(
                    WORKER_MODEL,
                    load_and_preprocess(img1_path, MODEL_INPUT_SIZES[victim_name]),
                )
                updates[clean_col] = cosine_similarity(emb_src, emb_tgt)
            except Exception:
                pass

        for attack in pending_attacks:
            adv_path = resolve_adv_path(str(row_data.get(ATTACK_TO_PATH_COL[attack], "")), WORKER_ADV_BASE_PATH)
            if not adv_path or not os.path.exists(adv_path):
                continue
            try:
                emb_adv = compute_embedding(
                    WORKER_MODEL,
                    load_and_preprocess(adv_path, MODEL_INPUT_SIZES[victim_name]),
                )
                updates[f"{victim_name}_{attack}_adv"] = cosine_similarity(emb_adv, emb_tgt)
            except Exception:
                pass

    elif victim_name == "IR152":
        emb_tgt = None
        try:
            emb_tgt = get_ir152_embedding(WORKER_MODEL, img2_path)
        except Exception:
            return out_i, updates

        if need_clean:
            try:
                emb_src = get_ir152_embedding(WORKER_MODEL, img1_path)
                updates[clean_col] = float(np.dot(emb_src, emb_tgt))
            except Exception:
                pass

        for attack in pending_attacks:
            adv_path = resolve_adv_path(str(row_data.get(ATTACK_TO_PATH_COL[attack], "")), WORKER_ADV_BASE_PATH)
            if not adv_path or not os.path.exists(adv_path):
                continue
            try:
                emb_adv = get_ir152_embedding(WORKER_MODEL, adv_path)
                updates[f"{victim_name}_{attack}_adv"] = float(np.dot(emb_adv, emb_tgt))
            except Exception:
                pass

    elif victim_name == "FaceAPI":
        if need_clean:
            sim = faceapi_similarity(img1_path, img2_path)
            if sim is not None:
                updates[clean_col] = sim
        for attack in pending_attacks:
            adv_path = resolve_adv_path(str(row_data.get(ATTACK_TO_PATH_COL[attack], "")), WORKER_ADV_BASE_PATH)
            if not adv_path or not os.path.exists(adv_path):
                continue
            sim = faceapi_similarity(adv_path, img2_path)
            if sim is not None:
                updates[f"{victim_name}_{attack}_adv"] = sim

    return out_i, updates


def init_perf_stats(attacks: List[str]) -> Dict[str, Dict[str, float]]:
    return {attack: {"success": 0, "total": 0, "impact_sum": 0.0} for attack in attacks}


def build_perf_from_similarity(
    similarity_df: pd.DataFrame,
    perf_csv: str,
    thresholds: Dict[str, object],
    attacks: List[str],
    victim_models: List[str],
    include_same_family: bool,
) -> Dict[str, Dict[str, float]]:
    perf_latest_map = {}
    perf_stats = init_perf_stats(attacks)

    for _, rec in similarity_df.iterrows():
        row_id = int(pd.to_numeric(rec.get("row_id", -1), errors="coerce"))
        attacker_model = str(rec.get("attacker_model", "")).strip()
        dataset_name = str(rec.get("dataset", "")).strip()
        attack_type = str(rec.get("attack_type", "")).strip()

        for victim_name in victim_models:
            if not include_same_family and equivalent_models(attacker_model, victim_name):
                continue
            threshold = threshold_for(thresholds, victim_name, dataset_name)
            if threshold is None:
                continue

            clean_col = f"{victim_name}_clean"
            clean_sim = rec.get(clean_col, np.nan)
            if pd.isna(clean_sim):
                continue

            perf_row = empty_perf_row(row_id=row_id, attacker_model=attacker_model, victim_model=victim_name)
            perf_row["dataset"] = dataset_name
            perf_row["attack_type"] = attack_type
            perf_row["threshold"] = float(threshold)
            perf_row["clean_similarity"] = float(clean_sim)

            for attack_name in attacks:
                adv_col = f"{victim_name}_{attack_name}_adv"
                adv_sim = rec.get(adv_col, np.nan)
                adv_path = norm_text(rec.get(ATTACK_TO_PATH_COL[attack_name], ""))
                if pd.isna(adv_sim) or not adv_path:
                    continue
                breach = int(
                    float(adv_sim) >= float(threshold)
                    if str(attack_type).strip().lower() == "impersonation_attack"
                    else float(adv_sim) < float(threshold)
                )
                impact = float(impact_value(float(clean_sim), float(adv_sim), attack_type))
                metrics = perf_metric_columns(attack_name)
                perf_row[metrics["adv_similarity"]] = float(adv_sim)
                perf_row[metrics["breach"]] = breach
                perf_row[metrics["impact"]] = impact
                perf_row[metrics["adv_image_path"]] = adv_path
                add_perf_stat(perf_stats, attack_name, breach, impact)

            perf_latest_map[(row_id, attacker_model, victim_name)] = perf_row

    write_perf_csv(perf_csv, perf_latest_map)
    return perf_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Read transfer adversarial paths, calculate victim similarities, and build a performance CSV.")
    parser.add_argument("--input-csv", default="transfer_adv_paths_all12.csv")
    parser.add_argument(
        "--output-csv",
        default="transfer_attack_performance_all12.csv",
        help="Performance output CSV. The raw similarity CSV is auto-derived unless --similarity-csv is provided.",
    )
    parser.add_argument("--perf-csv", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--similarity-csv",
        default="",
        help="Optional raw similarity CSV path. Defaults to a file derived from --output-csv.",
    )
    parser.add_argument("--base-path", default="", help="Optional dataset_extractedfaces directory. Auto-detected if omitted.")
    parser.add_argument("--adv-base-path", default="", help="Optional project root for adversarial image paths. Auto-detected if omitted.")
    parser.add_argument("--thresholds-json", default="", help="Optional verification_thresholds.json path. Auto-detected if omitted.")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=50, help="Print per-victim progress every N completed tasks.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(24, cpu_count()), help="Parallel workers for similarity sync.")
    parser.add_argument("--tf-threads", type=int, default=1)
    parser.add_argument("--snapshot-retries", type=int, default=3)
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--include-same-family-perf", action="store_true")
    parser.add_argument(
        "--enable-faceapi",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-faceapi",
        action="store_true",
        help="Disable FaceAPI similarity columns and performance rows.",
    )
    parser.add_argument("--enable-ir152", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-ir152", action="store_true", help="Disable IR152 similarity columns and performance rows.")
    parser.add_argument("--ir152-weights", default="", help="Optional IR152 weights path. Auto-detected if omitted.")
    parser.add_argument(
        "--attacks",
        default=",".join(ALL_ATTACKS),
        help="Comma-separated attack names to process.",
    )
    parser.add_argument(
        "--victim-models",
        default="",
        help="Optional comma-separated local DeepFace victim models to compute. Default: all supported local models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu_runtime(tf_threads=max(1, int(args.tf_threads)))

    input_csv = Path(args.input_csv).expanduser().resolve()
    project_root = discover_project_root(input_csv)

    output_raw = args.perf_csv or args.output_csv or "transfer_attack_performance_all12.csv"
    perf_csv = Path(output_raw).expanduser()
    if not perf_csv.is_absolute():
        perf_csv = (Path.cwd() / perf_csv).resolve()
    else:
        perf_csv = perf_csv.resolve()

    if args.similarity_csv:
        similarity_csv = Path(args.similarity_csv).expanduser()
        if not similarity_csv.is_absolute():
            similarity_csv = (Path.cwd() / similarity_csv).resolve()
        else:
            similarity_csv = similarity_csv.resolve()
    else:
        similarity_csv = derive_similarity_csv(perf_csv)

    base_path = Path(args.base_path).expanduser().resolve() if args.base_path else (
        first_existing_dir(
            [
                project_root / "dataset_extractedfaces",
                input_csv.parent / "dataset_extractedfaces",
                Path.cwd() / "dataset_extractedfaces",
            ]
        )
        or (project_root / "dataset_extractedfaces")
    )
    adv_base_path = Path(args.adv_base_path).expanduser().resolve() if args.adv_base_path else project_root
    thresholds_json = Path(args.thresholds_json).expanduser().resolve() if args.thresholds_json else (
        first_existing_file(
            [
                project_root / "verification_thresholds.json",
                input_csv.parent / "verification_thresholds.json",
                Path.cwd() / "verification_thresholds.json",
                project_root / "thresholds.json",
                input_csv.parent / "thresholds.json",
                Path.cwd() / "thresholds.json",
            ]
        )
        or (project_root / "verification_thresholds.json")
    )
    ir152_weights = Path(args.ir152_weights).expanduser().resolve() if args.ir152_weights else (
        first_existing_file(
            [
                project_root / "ir152.pth",
                input_csv.parent / "ir152.pth",
                Path.cwd() / "ir152.pth",
            ]
        )
        or (project_root / "ir152.pth")
    )

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not thresholds_json.is_file():
        raise FileNotFoundError(f"verification_thresholds.json not found. Checked near {project_root} and {input_csv.parent}")
    if not base_path.is_dir():
        raise FileNotFoundError(f"dataset_extractedfaces directory not found: {base_path}")
    perf_csv.parent.mkdir(parents=True, exist_ok=True)
    similarity_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(thresholds_json, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    enable_faceapi = not args.disable_faceapi and "FaceAPI" in thresholds
    if args.enable_faceapi:
        enable_faceapi = True

    enable_ir152 = not args.disable_ir152 and ir152_weights.is_file() and "IR152" in thresholds
    if args.enable_ir152:
        enable_ir152 = True

    attacks = parse_list_arg(args.attacks, ALL_ATTACKS)
    model_names = parse_model_list_arg(args.victim_models, MODEL_INPUT_SIZES.keys())
    requested_workers = max(1, int(args.workers))
    progress_every = max(1, int(args.progress_every))
    print(f"[config] input_csv={input_csv}")
    print(f"[config] output_csv={perf_csv}")
    print(f"[config] similarity_csv={similarity_csv}")
    print(f"[config] project_root={project_root}")
    print(f"[config] base_path={base_path}")
    print(f"[config] adv_base_path={adv_base_path}")
    print(f"[config] thresholds_json={thresholds_json}")
    print(f"[config] ir152_weights={ir152_weights if enable_ir152 else 'disabled'}")
    print(f"[config] workers={requested_workers} tf_threads_per_worker={int(args.tf_threads)}")
    print(f"[config] victim_models={model_names + ([ 'FaceAPI'] if enable_faceapi else []) + ([ 'IR152'] if enable_ir152 else [])}")

    df_in = snapshot_input_csv(input_csv=str(input_csv), max_retries=max(1, int(args.snapshot_retries)))
    for col in KEY_COLS:
        if col not in df_in.columns:
            df_in[col] = ""
        df_in[col] = df_in[col].astype(str)
    df_in = consolidate_duplicate_rows(df_in, KEY_COLS)
    if args.limit > 0:
        df_in = df_in.head(int(args.limit))

    if similarity_csv.exists():
        df_out = pd.read_csv(similarity_csv, dtype={"row_id": str})
    else:
        df_out = pd.DataFrame(columns=SIMILARITY_BASE_COLUMNS)

    drop_cols = [c for c in df_out.columns if c.startswith("Unnamed:")]
    if drop_cols:
        df_out = df_out.drop(columns=drop_cols)

    for col in SIMILARITY_BASE_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = ""
        df_out[col] = (
            df_out[col]
            .astype(str)
            .replace({"nan": "", "None": "", "none": ""})
            .fillna("")
        )

    df_out = ensure_similarity_columns(df_out, model_names, attacks, enable_faceapi, enable_ir152)
    df_out = consolidate_duplicate_rows(df_out, KEY_COLS)

    out_index: Dict[Tuple[str, str, str, str, str, str], int] = {}
    for i, row in df_out.iterrows():
        out_index[row_key(row)] = i

    new_rows = []
    for _, in_row in df_in.iterrows():
        key = row_key(in_row)
        if key in out_index:
            continue
        data = {c: norm_text(in_row.get(c, "")) for c in SIMILARITY_BASE_COLUMNS}
        new_rows.append(data)
    if new_rows:
        df_out = pd.concat([df_out, pd.DataFrame(new_rows)], ignore_index=True)
        out_index = {}
        for i, row in df_out.iterrows():
            out_index[row_key(row)] = i

    updated = 0
    considered = 0
    entries = []
    for _, in_row in df_in.iterrows():
        has_any_adv = any(norm_text(in_row.get(ATTACK_TO_PATH_COL[attack], "")) for attack in attacks)
        if not has_any_adv:
            continue

        considered += 1
        key = row_key(in_row)
        out_i = out_index[key]
        row_data = {c: norm_text(in_row.get(c, "")) for c in SIMILARITY_BASE_COLUMNS}
        for attack_name in attacks:
            path_col = ATTACK_TO_PATH_COL[attack_name]
            incoming_path = row_data.get(path_col, "")
            if incoming_path and norm_text(df_out.at[out_i, path_col]) == "":
                df_out.at[out_i, path_col] = incoming_path
        entries.append((out_i, row_data))

    victim_order = list(model_names)
    if enable_ir152:
        victim_order.append("IR152")
    if enable_faceapi:
        victim_order.append("FaceAPI")

    ctx = get_context("spawn")
    checkpoint_every = max(1, int(args.checkpoint_every))

    for victim_name in victim_order:
        tasks = []
        for out_i, row_data in entries:
            out_row = df_out.loc[out_i]
            need_clean, pending_attacks = pending_for_victim(out_row, row_data, victim_name, attacks)
            if not need_clean and not pending_attacks:
                continue
            tasks.append(
                {
                    "out_i": out_i,
                    "row_data": row_data,
                    "need_clean": need_clean,
                    "pending_attacks": pending_attacks,
                }
            )

        if not tasks:
            print(f"[victim:{victim_name}] no pending rows")
            continue

        worker_count = min(requested_workers, len(tasks))
        chunksize = max(1, len(tasks) // max(1, worker_count * 4))
        print(f"[victim:{victim_name}] pending_rows={len(tasks)} workers={worker_count} chunksize={chunksize}")
        victim_done = 0

        with ctx.Pool(
            processes=worker_count,
            initializer=init_similarity_worker,
            initargs=(
                victim_name,
                str(base_path),
                str(adv_base_path),
                attacks,
                int(args.tf_threads),
                str(ir152_weights),
            ),
        ) as pool:
            for out_i, updates_dict in pool.imap_unordered(compute_victim_row_task, tasks, chunksize=chunksize):
                victim_done += 1
                if not updates_dict:
                    if victim_done % progress_every == 0 or victim_done == len(tasks):
                        print(f"[victim:{victim_name}] progress={victim_done}/{len(tasks)}")
                    continue
                for col, value in updates_dict.items():
                    df_out.at[out_i, col] = value
                updated += 1
                if victim_done % progress_every == 0 or victim_done == len(tasks):
                    print(f"[victim:{victim_name}] progress={victim_done}/{len(tasks)}")
                if updated % checkpoint_every == 0:
                    atomic_write_csv(df_out, similarity_csv)
                    print(f"[checkpoint] similarity_tasks_updated={updated}")

        atomic_write_csv(df_out, similarity_csv)
        print(f"[victim:{victim_name}] completed")

    atomic_write_csv(df_out, similarity_csv)
    print(f"[similarity] rows_considered={considered} tasks_updated={updated} output={similarity_csv}")

    perf_victims = list(model_names)
    if enable_faceapi and "FaceAPI" in thresholds:
        perf_victims.append("FaceAPI")
    if enable_ir152 and "IR152" in thresholds:
        perf_victims.append("IR152")

    perf_stats = build_perf_from_similarity(
        similarity_df=df_out,
        perf_csv=str(perf_csv),
        thresholds=thresholds,
        attacks=attacks,
        victim_models=perf_victims,
        include_same_family=bool(args.include_same_family_perf),
    )
    print(f"[perf] output={perf_csv}")

    if args.print_summary:
        print_pairwise_perf(perf_stats, attacks, "[perf] vanilla vs SM")



if __name__ == "__main__":
    main()
