#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Support both direct script execution and python -m from the repository root.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core.transfer_attack_core import (
    ATTACKER_MODELS,
    ALL_ATTACKS,
    FACESM_ATTACKS,
    build_attacker,
    configure_cpu_runtime,
    denormalize,
    load_and_preprocess,
    resolve_image_path,
    run_attack,
    save_adv,
)

import tensorflow as tf


def main():
    ap = argparse.ArgumentParser(description='Generate adversarial face images with selected transfer attacks.')
    ap.add_argument('--input-csv', required=True, help='CSV containing row_id, img1, img2, dataset, attack_type columns.')
    ap.add_argument('--dataset-root', required=True, help='Root directory containing aligned face images.')
    ap.add_argument('--output-root', required=True, help='Directory where generated adversarial images and path CSV will be written.')
    ap.add_argument('--attacker-model', required=True, choices=list(ATTACKER_MODELS.keys()))
    ap.add_argument('--attacks', default=None, help='Comma-separated attack names from ALL_ATTACKS.')
    ap.add_argument('--objective', choices=['vanilla', 'facesm', 'both'], default='vanilla')
    ap.add_argument('--source-lambda', type=float, default=0.20,
                    help='Source repulsion weight; zero gives mirror fusion only.')
    args = ap.parse_args()
    if not math.isfinite(args.source_lambda) or args.source_lambda < 0:
        ap.error('--source-lambda must be finite and nonnegative')

    configure_cpu_runtime(1)
    supported = ALL_ATTACKS if args.objective == 'vanilla' else FACESM_ATTACKS
    attacks = supported if args.attacks is None else [a.strip() for a in args.attacks.split(',') if a.strip()]
    if not attacks or any(a not in supported for a in attacks):
        ap.error('Choose attacks from: ' + ','.join(supported))
    objectives = ['vanilla', 'facesm'] if args.objective == 'both' else [args.objective]
    input_size = ATTACKER_MODELS[args.attacker_model]
    model = build_attacker(args.attacker_model)
    df = pd.read_csv(args.input_csv)
    rows = []

    for _, rec in df.iterrows():
        row_id = int(rec['row_id'])
        src_path = resolve_image_path(rec['img1'], args.dataset_root)
        tgt_path = resolve_image_path(rec['img2'], args.dataset_root)
        src = tf.expand_dims(load_and_preprocess(src_path, input_size), 0)
        tgt = tf.expand_dims(load_and_preprocess(tgt_path, input_size), 0)
        out = {
            'row_id': row_id,
            'attacker_model': args.attacker_model,
            'img1': rec['img1'],
            'img2': rec['img2'],
            'dataset': rec['dataset'],
            'attack_type': rec['attack_type'],
        }
        out['objective'] = args.objective
        out['source_lambda'] = args.source_lambda if args.objective != 'vanilla' else 0.0
        for attack in attacks:
            for objective in objectives:
                label = attack if objective == 'vanilla' else attack + '_SM'
                adv = run_attack(attack, model, src, tgt, rec['attack_type'], input_size,
                                 objective=objective, source_lambda=args.source_lambda)
                out[f'{label.lower()}_path'] = save_adv(
                    denormalize(adv.numpy()[0]), label, src_path, tgt_path,
                    rec['attack_type'], args.attacker_model, row_id, args.output_root
                )
        rows.append(out)

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = '' if args.objective == 'vanilla' else '_' + args.objective
    pd.DataFrame(rows).to_csv(out_dir / f'{args.attacker_model}{suffix}_adv_paths.csv', index=False)


if __name__ == '__main__':
    main()
