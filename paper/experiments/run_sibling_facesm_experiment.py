#!/usr/bin/env python3
"""Run the Sibling-Attack and FaceSM variant on local face-pair data.

The upstream Sibling scripts are Colab-oriented and expect hard-coded paths.
This wrapper prepares a local CSV, selects a comparable evaluation slice, and
sets the device/weight paths explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys

import pandas as pd
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIBLING_DIR = ROOT / "Sibling-Attack"
DEFAULT_DATASET_ROOT = Path(os.environ.get("FACESM_DATASET_ROOT", "dataset_extractedfaces"))
CONTENT_DATASET_ROOT = "/content/face_module/dataset_extractedfaces"


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def map_image_path(value: str, dataset: str, dataset_root: Path) -> str:
    value = str(value)
    if value.startswith(CONTENT_DATASET_ROOT):
        rel = value[len(CONTENT_DATASET_ROOT) :].lstrip("/")
        return str(dataset_root / rel)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(dataset_root / dataset / path.name)


def prepare_pairs(src_csv: Path, out_csv: Path, dataset_root: Path, dataset: str, limit: int | None) -> int:
    df = pd.read_csv(src_csv)
    if "attack_type" in df.columns:
        mask = df["attack_type"].astype(str).str.strip().str.lower().eq("impersonation_attack")
        df = df.loc[mask].copy()
    if dataset:
        df = df.loc[df["dataset"].astype(str).eq(dataset)].copy()
    if limit is not None:
        df = df.head(limit).copy()

    if df.empty:
        raise ValueError("No pairs selected for the requested dataset/limit.")

    df["img1"] = [map_image_path(v, d, dataset_root) for v, d in zip(df["img1"], df["dataset"])]
    df["img2"] = [map_image_path(v, d, dataset_root) for v, d in zip(df["img2"], df["dataset"])]

    missing = sorted(
        {p for p in df["img1"].tolist() + df["img2"].tolist() if not Path(p).exists()}
    )
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} selected image paths are missing, for example:\n{preview}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return len(df)


def ensure_weights(sibling_dir: Path, weights_dir: Path | None) -> None:
    model_dir = sibling_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    required = ["ir152.pth", "ir152_ar.pth"]
    for name in required:
        target = model_dir / name
        if target.exists():
            continue
        if weights_dir is None:
            raise FileNotFoundError(
                f"Missing {target}. Pass --weights-dir containing {', '.join(required)}."
            )
        src = weights_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required weight: {src}")
        try:
            target.symlink_to(src)
        except OSError:
            shutil.copy2(src, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sibling-dir", type=Path, default=DEFAULT_SIBLING_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--source-csv", type=Path, default=None)
    parser.add_argument("--work-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results_sibling")
    parser.add_argument("--dataset", default="lfw_pairs")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--outer-loops", type=int, default=None)
    parser.add_argument("--inner-loops", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-facesm", action="store_true")
    args = parser.parse_args()

    sibling_dir = args.sibling_dir.resolve()
    source_csv = args.source_csv or sibling_dir / "datasets" / "pairs.csv"
    work_csv = args.work_csv or args.output_root / f"sibling_{args.dataset}_pairs.csv"

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    ensure_weights(sibling_dir, args.weights_dir)
    if args.resume and work_csv.exists():
        selected = len(pd.read_csv(work_csv))
    else:
        selected = prepare_pairs(source_csv, work_csv, args.dataset_root, args.dataset, args.limit)

    config = OmegaConf.load(sibling_dir / "configs" / "config.yaml")
    if args.outer_loops is not None:
        config.attack["outer_loops"] = args.outer_loops
    if args.inner_loops is not None:
        config.attack["inner_loops"] = args.inner_loops
    config.attack["gpu"] = 0

    sys.path.insert(0, str(sibling_dir))
    old_cwd = Path.cwd()
    os.chdir(sibling_dir)
    try:
        attack_mod = import_module(sibling_dir / "attack.py", "sibling_attack_vanilla")
        facesm_mod = import_module(sibling_dir / "attack_facesm.py", "sibling_attack_facesm")
        attack_mod.device = device
        facesm_mod.device = device
        attack_mod.VERBOSE = not args.quiet
        facesm_mod.VERBOSE = not args.quiet

        fr_model, ar_model = attack_mod.load_surrogate_model()
        print(f"Selected {selected} pairs from {args.dataset}; device={device}; csv={work_csv}")

        if not args.skip_vanilla:
            attack_mod.run_and_log(
                str(work_csv),
                "/",
                attack_mod.sibling_attack,
                fr_model,
                ar_model,
                config,
                out_col="sibling_attack_path",
                save_dir=str(args.output_root / "results_adv_images_vanilla"),
            )

        if not args.skip_facesm:
            facesm_mod.run_and_log(
                str(work_csv),
                "/",
                facesm_mod.sibling_attack_facesm,
                fr_model,
                ar_model,
                config,
                out_col="sibling_attack_facesm",
                save_dir=str(args.output_root / "results_adv_images_facesm"),
            )
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
