#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ROBFR_DIR = ROOT / 'robfr'
BASE_RESULTS = ROOT / 'results' / 'robfr_facesm_pilot'
RESULTS_ROOT = ROOT / 'results' / 'robfr_facesm_lgc_pilot'

DEFAULT_SURROGATE = 'ArcFace'
DEFAULT_VICTIM = 'FaceNet-VGGFace2'
DEFAULT_ATTACK = 'LGC'

SAMPLE_FILES = {
    'dodging': (
        'lfw_dodging_sample_300_112x112.csv',
        'pairs_lfw_dodging_300_112x112.txt',
    ),
    'impersonate': (
        'lfw_impersonate_sample_300_112x112.csv',
        'pairs_lfw_impersonate_300_112x112.txt',
    ),
}


def ensure_reference_files(goal: str):
    sample_name, pair_name = SAMPLE_FILES[goal]
    src_sample = BASE_RESULTS / sample_name
    src_pairs = BASE_RESULTS / pair_name
    if not src_sample.exists() or not src_pairs.exists():
        raise FileNotFoundError(f'Missing base RobFR sample files for {goal}: {src_sample}, {src_pairs}')
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    dst_sample = RESULTS_ROOT / sample_name
    dst_pairs = RESULTS_ROOT / pair_name
    shutil.copy2(src_sample, dst_sample)
    shutil.copy2(src_pairs, dst_pairs)
    return dst_sample, dst_pairs


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


def build_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    idx = ['goal', 'attack', 'surrogate', 'victim', 'num_pairs']
    wide = summary_df.pivot_table(index=idx, columns='variant', values=['dist_mean', 'score_mean', 'success_rate'])
    wide.columns = [f'{metric}_{variant}' for metric, variant in wide.columns]
    wide = wide.reset_index()
    wide['delta_success_pp'] = 100.0 * (wide['success_rate_facesm'] - wide['success_rate_vanilla'])
    wide['delta_score'] = wide['score_mean_facesm'] - wide['score_mean_vanilla']
    wide['delta_dist'] = wide['dist_mean_facesm'] - wide['dist_mean_vanilla']
    return wide.sort_values(['goal', 'attack']).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', default=DEFAULT_SURROGATE)
    parser.add_argument('--victim', default=DEFAULT_VICTIM)
    parser.add_argument('--attack', default=DEFAULT_ATTACK)
    parser.add_argument('--eps', type=float, default=4.0)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--sigma', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--source-lambda', type=float, default=0.20)
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROBFR_DIR)

    summary_rows = []
    for goal in ['dodging', 'impersonate']:
        sample_csv, pairs_file = ensure_reference_files(goal)
        env['ROBFR_LFW_PAIRS_FILE'] = str(pairs_file)
        for variant_name, extra_flags in [
            ('vanilla', []),
            ('facesm', ['--facesm', '--source-lambda', f'{args.source_lambda:.2f}']),
        ]:
            output_dir = RESULTS_ROOT / f'lfw_{args.attack.lower()}_{variant_name}_{goal}_{args.surrogate.lower()}'
            if output_dir.exists():
                shutil.rmtree(output_dir)
            run_cmd(
                [
                    'python3', '-m', f'RobFR.benchmark.{args.attack}_black',
                    '--device', args.device,
                    '--dataset', 'lfw',
                    '--model', args.surrogate,
                    '--goal', goal,
                    '--distance', 'l2',
                    '--eps', str(args.eps),
                    '--iters', str(args.iters),
                    '--mu', str(args.mu),
                    '--num_samples', str(args.num_samples),
                    '--sigma', str(args.sigma),
                    '--batch_size', str(args.batch_size),
                    '--output', str(output_dir),
                ] + extra_flags,
                cwd=ROBFR_DIR,
                env=env,
                log_path=RESULTS_ROOT / f'run_{args.attack.lower()}_{variant_name}_{goal}_{args.surrogate.lower()}.log',
            )

            anno_path = output_dir / 'annotation.txt'
            log_name = f'log_lfw_{args.victim}_{args.attack}_{variant_name}_{goal}_{args.surrogate}.csv'
            log_path = ROBFR_DIR / 'log' / log_name
            if log_path.exists():
                log_path.unlink()
            run_cmd(
                [
                    'python3', '-m', 'RobFR.benchmark.run_test',
                    '--device', args.device,
                    '--dataset', 'lfw',
                    '--model', args.victim,
                    '--distance', 'l2',
                    '--anno', str(anno_path),
                    '--log', log_name,
                    '--goal', goal,
                ],
                cwd=ROBFR_DIR,
                env=env,
                log_path=RESULTS_ROOT / f'test_{args.attack.lower()}_{variant_name}_{goal}_{args.surrogate}_{args.victim}.log',
            )
            stats = summarize_log(log_path)
            stats.update({
                'attack': args.attack,
                'goal': goal,
                'variant': variant_name,
                'surrogate': args.surrogate,
                'victim': args.victim,
                'num_pairs': 300,
                'sample_csv': str(sample_csv),
                'pairs_file': str(pairs_file),
                'annotation': str(anno_path),
                'score_log': str(log_path),
            })
            summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows).sort_values(['goal', 'attack', 'variant']).reset_index(drop=True)
    summary_df.to_csv(RESULTS_ROOT / 'pilot_summary_lgc_only.csv', index=False)
    summary_df[summary_df['goal'] == 'dodging'].to_csv(RESULTS_ROOT / 'pilot_summary_dodging_300pairs_lgc.csv', index=False)
    summary_df[summary_df['goal'] == 'impersonate'].to_csv(RESULTS_ROOT / 'pilot_summary_impersonate_300pairs_lgc.csv', index=False)
    summary_df.to_csv(RESULTS_ROOT / 'pilot_summary_combined_300pairs_lgc.csv', index=False)

    comparison_df = build_comparison(summary_df)
    comparison_df.to_csv(RESULTS_ROOT / 'pilot_comparison_combined_300pairs_lgc.csv', index=False)

    base_summary = pd.read_csv(BASE_RESULTS / 'pilot_summary_combined_300pairs_3attacks.csv')
    base_comparison = pd.read_csv(BASE_RESULTS / 'pilot_comparison_combined_300pairs_3attacks.csv')
    summary_4 = pd.concat([base_summary, summary_df], ignore_index=True).sort_values(['goal', 'attack', 'variant']).reset_index(drop=True)
    comparison_4 = pd.concat([base_comparison, comparison_df], ignore_index=True).sort_values(['goal', 'attack']).reset_index(drop=True)
    summary_4.to_csv(RESULTS_ROOT / 'pilot_summary_combined_300pairs_4attacks.csv', index=False)
    comparison_4.to_csv(RESULTS_ROOT / 'pilot_comparison_combined_300pairs_4attacks.csv', index=False)

    readme = RESULTS_ROOT / 'README.md'
    readme.write_text(
        '\n'.join([
            '# RobFR LGC FaceSM Extension',
            '',
            'This folder contains a fresh RobFR experiment for the missing LGC attack.',
            'It reuses the same 300-pair LFW dodging and impersonation samples from the existing RobFR pilot so that results remain directly comparable.',
            '',
            'Files:',
            '- `pilot_summary_lgc_only.csv`: raw summary for vanilla and FaceSM LGC.',
            '- `pilot_comparison_combined_300pairs_lgc.csv`: vanilla vs FaceSM comparison for LGC.',
            '- `pilot_summary_combined_300pairs_4attacks.csv`: existing BIM/MIM/CIM rows plus the new LGC rows.',
            '- `pilot_comparison_combined_300pairs_4attacks.csv`: existing BIM/MIM/CIM comparisons plus the new LGC comparisons.',
        ]) + '\n',
        encoding='utf-8',
    )

    print(summary_df.to_string(index=False))
    print('\nWrote LGC results to', RESULTS_ROOT)


if __name__ == '__main__':
    main()
