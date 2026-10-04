#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
ROBFR_DIR = ROOT / 'robfr'
DATASET_ROOT = ROOT / 'dataset_extractedfaces' / 'lfw_pairs'
INPUT_CSV = ROOT / 'input2400.csv'
RESULTS_ROOT = ROOT / 'results' / 'robfr_facesm_pilot'

MODEL_IMG_SHAPES = {
    'ArcFace': (112, 112),
    'MobileFace': (112, 112),
    'Mobilenet': (112, 112),
    'Mobilenet-stride1': (112, 112),
    'MobilenetV2': (112, 112),
    'MobilenetV2-stride1': (112, 112),
    'ResNet50': (112, 112),
    'ResNet50-casia': (112, 112),
    'ShuffleNet_V1_GDConv': (112, 112),
    'ShuffleNet_V2_GDConv-stride1': (112, 112),
    'IR50-Softmax': (112, 112),
    'IR50-ArcFace': (112, 112),
    'FaceNet-VGGFace2': (160, 160),
    'FaceNet-casia': (160, 160),
    'CosFace': (112, 96),
    'SphereFace': (112, 96),
}

NAME_RE = re.compile(r'^(.*)_(\d+)\.jpg$')


def parse_lfw_filename(name: str):
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f'Unexpected LFW filename: {name}')
    return m.group(1), int(m.group(2))


def ensure_lfw_subset(num_pairs: int, seed: int, goal: str, img_shape):
    df = pd.read_csv(INPUT_CSV)
    attack_type = 'dodging_attack' if goal == 'dodging' else 'impersonation_attack'
    lfw = df[(df['dataset'] == 'lfw_pairs') & (df['attack_type'] == attack_type)].copy()
    sample = lfw.sample(n=min(num_pairs, len(lfw)), random_state=seed).reset_index(drop=True)

    h, w = img_shape
    data_dir = ROBFR_DIR / 'data' / f'lfw-{h}x{w}'
    pairs_file = RESULTS_ROOT / f'pairs_lfw_{goal}_{len(sample)}_{h}x{w}.txt'
    sample_csv = RESULTS_ROOT / f'lfw_{goal}_sample_{len(sample)}_{h}x{w}.csv'
    data_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    seen = set()
    pair_lines = []
    for _, row in sample.iterrows():
        src_name = Path(row['img1']).name
        dst_name = Path(row['img2']).name
        src_person, src_idx = parse_lfw_filename(src_name)
        dst_person, dst_idx = parse_lfw_filename(dst_name)
        for person, idx, fname in [(src_person, src_idx, src_name), (dst_person, dst_idx, dst_name)]:
            key = (person, idx)
            if key in seen:
                continue
            seen.add(key)
            src_path = DATASET_ROOT / fname
            out_dir = data_dir / person
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'{person}_{idx:04d}.jpg'
            if not out_path.exists():
                img = Image.open(src_path).convert('RGB').resize((w, h), Image.BILINEAR)
                img.save(out_path, quality=95)
        if goal == 'dodging':
            pair_lines.append(f'{src_person}\t{src_idx}\t{dst_idx}')
        else:
            pair_lines.append(f'{src_person}\t{src_idx}\t{dst_person}\t{dst_idx}')

    sample.to_csv(sample_csv, index=False)
    pairs_file.write_text('\n'.join(pair_lines) + '\n', encoding='utf-8')
    return data_dir, pairs_file, sample_csv, len(sample)


def run_cmd(cmd, cwd, env, log_path):
    with open(log_path, 'w', encoding='utf-8') as logf:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}\nSee log: {log_path}')


def summarize_log(log_csv: Path):
    rows = []
    with open(log_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {'n': 0, 'success_rate': 0.0, 'score_mean': 0.0, 'dist_mean': 0.0}
    scores = [float(r['score']) for r in rows]
    dists = [float(r['dist']) for r in rows]
    succ = [int(r['success']) for r in rows]
    return {
        'n': len(rows),
        'success_rate': sum(succ) / len(succ),
        'score_mean': sum(scores) / len(scores),
        'dist_mean': sum(dists) / len(dists),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-pairs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--attacks', type=str, default='FGSM')
    parser.add_argument('--goal', type=str, default='dodging', choices=['dodging', 'impersonate'])
    parser.add_argument('--surrogate', type=str, default='ArcFace')
    parser.add_argument('--victims', type=str, default='FaceNet-VGGFace2')
    parser.add_argument('--eps', type=float, default=4.0)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    if args.surrogate not in MODEL_IMG_SHAPES:
        raise ValueError(f'Unknown surrogate image shape for {args.surrogate}')
    surrogate_shape = MODEL_IMG_SHAPES[args.surrogate]
    data_dir, pairs_file, sample_csv, actual_n = ensure_lfw_subset(
        args.num_pairs,
        args.seed,
        args.goal,
        surrogate_shape,
    )
    attacks = [a.strip() for a in args.attacks.split(',') if a.strip()]
    victims = [v.strip() for v in args.victims.split(',') if v.strip()]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROBFR_DIR)
    env['ROBFR_LFW_PAIRS_FILE'] = str(pairs_file)

    variants = [
        ('vanilla', []),
        ('facesm', ['--facesm', '--source-lambda', '0.20']),
    ]

    summary_rows = []
    for attack_name in attacks:
        for variant_name, extra_flags in variants:
            output_dir = RESULTS_ROOT / f'lfw_{attack_name.lower()}_{variant_name}_{args.goal}_{args.surrogate.lower()}'
            if output_dir.exists():
                shutil.rmtree(output_dir)
            cmd = [
                'python3', '-m', f'RobFR.benchmark.{attack_name}_black',
                '--device', 'cpu',
                '--dataset', 'lfw',
                '--model', args.surrogate,
                '--goal', args.goal,
                '--distance', 'l2',
                '--eps', str(args.eps),
                '--batch_size', str(args.batch_size),
                '--output', str(output_dir),
            ] + extra_flags
            run_cmd(
                cmd,
                cwd=ROBFR_DIR,
                env=env,
                log_path=RESULTS_ROOT / f'run_{attack_name.lower()}_{variant_name}_{args.goal}_{args.surrogate.lower()}.log',
            )

            anno_path = output_dir / 'annotation.txt'
            for victim in victims:
                log_name = f'log_lfw_{victim}_{attack_name}_{variant_name}_{args.goal}_{args.surrogate}.csv'
                log_path = ROBFR_DIR / 'log' / log_name
                if log_path.exists():
                    log_path.unlink()
                test_cmd = [
                    'python3', '-m', 'RobFR.benchmark.run_test',
                    '--device', 'cpu',
                    '--dataset', 'lfw',
                    '--model', victim,
                    '--distance', 'l2',
                    '--anno', str(anno_path),
                    '--log', log_name,
                    '--goal', args.goal,
                ]
                run_cmd(
                    test_cmd,
                    cwd=ROBFR_DIR,
                    env=env,
                    log_path=RESULTS_ROOT / f'test_{attack_name.lower()}_{variant_name}_{args.goal}_{args.surrogate}_{victim}.log',
                )
                stats = summarize_log(log_path)
                stats.update({
                    'attack': attack_name,
                    'goal': args.goal,
                    'variant': variant_name,
                    'surrogate': args.surrogate,
                    'victim': victim,
                    'num_pairs': actual_n,
                    'sample_csv': str(sample_csv),
                    'pairs_file': str(pairs_file),
                    'annotation': str(anno_path),
                    'score_log': str(log_path),
                })
                summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_ROOT / 'pilot_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(summary_df.to_string(index=False))
    print(f'\nWrote summary to {summary_csv}')


if __name__ == '__main__':
    main()
