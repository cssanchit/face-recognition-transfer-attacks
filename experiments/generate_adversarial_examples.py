#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from core.transfer_attack_core import (
    ATTACKER_MODELS,
    ALL_ATTACKS,
    build_attacker,
    configure_cpu_runtime,
    denormalize,
    load_and_preprocess,
    resolve_image_path,
    run_attack,
    save_adv,
)


def main():
    ap = argparse.ArgumentParser(description='Generate adversarial face images with selected transfer attacks.')
    ap.add_argument('--input-csv', required=True, help='CSV containing row_id, img1, img2, dataset, attack_type columns.')
    ap.add_argument('--dataset-root', required=True, help='Root directory containing aligned face images.')
    ap.add_argument('--output-root', required=True, help='Directory where generated adversarial images and path CSV will be written.')
    ap.add_argument('--attacker-model', required=True, choices=list(ATTACKER_MODELS.keys()))
    ap.add_argument('--attacks', default=','.join(ALL_ATTACKS), help='Comma-separated attack names from ALL_ATTACKS.')
    args = ap.parse_args()

    configure_cpu_runtime(1)
    attacks = [a.strip() for a in args.attacks.split(',') if a.strip()]
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
        for attack in attacks:
            adv = run_attack(attack, model, src, tgt, rec['attack_type'], input_size)
            out[f'{attack.lower()}_path'] = save_adv(
                denormalize(adv.numpy()[0]), attack, src_path, tgt_path,
                rec['attack_type'], args.attacker_model, row_id, args.output_root
            )
        rows.append(out)

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f'{args.attacker_model}_adv_paths.csv', index=False)


if __name__ == '__main__':
    main()
