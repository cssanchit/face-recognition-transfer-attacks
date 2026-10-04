#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_ROOT = Path('results')
OUT_DIR = RESULTS_ROOT / 'paper_ready_lambda020_limit1000'
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'charts').mkdir(parents=True, exist_ok=True)

PERF_FILES = [
    RESULTS_ROOT / 'ARCFACE' / 'transfer_attack_performance_all12_lambda020_limit1000.csv',
    RESULTS_ROOT / 'Facenet512' / 'transfer_attack_performance_all12_lambda020_limit1000.csv',
    RESULTS_ROOT / 'Ghostfacenet' / 'transfer_attack_performance_all12_lambda020_limit1000.csv',
    RESULTS_ROOT / 'VGG-Face' / 'transfer_attack_performance_all12_lambda020_limit1000.csv',
]

ATTACK_PAIRS: List[Tuple[str, str, str]] = [
    ('pgd', 'pgd_sm', 'PGD'),
    ('mi_fgsm', 'mi_fgsm_sm', 'MI-FGSM'),
    ('ti_fgsm', 'ti_fgsm_sm', 'TI-FGSM'),
    ('si_ni_fgsm', 'si_ni_fgsm_sm', 'SI-NI-FGSM'),
    ('mi_admix_di_ti', 'mi_admix_di_ti_sm', 'MI-ADMIX-DI-TI'),
]
ATTACK_ORDER = [x[2] for x in ATTACK_PAIRS]
DATASET_LABELS = {
    'celeba_pairs': 'CelebA',
    'lfw_pairs': 'LFW',
    'vggface2_pairs': 'VGGFace2',
}
ATTACKER_LABELS = ['ArcFace', 'Facenet512', 'GhostFaceNet', 'VGG-Face']
VICTIM_LABELS = ['ArcFace', 'GhostFaceNet', 'IR152', 'VGG-Face']


def validate_perf_file(df: pd.DataFrame, path: Path) -> None:
    required = {'row_id', 'attacker_model', 'victim_model', 'dataset', 'attack_type'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing required columns: {sorted(missing)}')
    for vanilla, plus, label in ATTACK_PAIRS:
        for prefix in (vanilla, plus):
            for suffix in ('breach', 'impact'):
                col = f'{prefix}_{suffix}'
                if col not in df.columns:
                    raise ValueError(f'{path} missing required attack column: {col}')
        # Ensure chosen attacks actually have data.
        if not df[f'{vanilla}_breach'].notna().any() or not df[f'{plus}_breach'].notna().any():
            raise ValueError(f'{path} lacks usable data for attack pair {label}')


def load_long_results() -> pd.DataFrame:
    rows = []
    for path in PERF_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        validate_perf_file(df, path)
        # Restrict the paper summaries to the stated common victim pool.
        df = df[df['victim_model'].astype(str).isin(VICTIM_LABELS)].copy()
        for vanilla, plus, label in ATTACK_PAIRS:
            for variant, prefix, is_plus in [('vanilla', vanilla, False), ('plus', plus, True)]:
                bcol = f'{prefix}_breach'
                icol = f'{prefix}_impact'
                sub = df[['row_id', 'attacker_model', 'victim_model', 'dataset', 'attack_type', bcol, icol]].copy()
                sub = sub[sub[bcol].notna() & sub[icol].notna()].copy()
                sub = sub.rename(columns={bcol: 'breach', icol: 'impact'})
                sub['attack'] = label
                sub['variant'] = variant
                sub['is_plus'] = is_plus
                rows.append(sub)
    long_df = pd.concat(rows, ignore_index=True)
    long_df['setting'] = long_df['attack_type'].map({
        'impersonation_attack': 'Impersonation',
        'dodging_attack': 'Dodging',
    })
    if long_df['setting'].isna().any():
        bad = sorted(long_df.loc[long_df['setting'].isna(), 'attack_type'].astype(str).unique())
        raise ValueError(f'Unexpected attack_type values: {bad}')
    return long_df


def summarize_pairwise(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in ['Impersonation', 'Dodging']:
        for attack in ATTACK_ORDER:
            g = long_df[(long_df['setting'] == setting) & (long_df['attack'] == attack)]
            van = g[g['variant'] == 'vanilla']
            plu = g[g['variant'] == 'plus']
            rows.append({
                'setting': setting,
                'attack': attack,
                'vanilla_breach_rate': van['breach'].mean(),
                'plus_breach_rate': plu['breach'].mean(),
                'vanilla_impact_mean': van['impact'].mean(),
                'plus_impact_mean': plu['impact'].mean(),
                'delta_breach_pp': (plu['breach'].mean() - van['breach'].mean()) * 100.0,
                'delta_impact': plu['impact'].mean() - van['impact'].mean(),
            })
    out = pd.DataFrame(rows)
    return out


def summarize_modelwise(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attacker in ATTACKER_LABELS:
        g = long_df[long_df['attacker_model'] == attacker]
        van = g[g['variant'] == 'vanilla']
        plu = g[g['variant'] == 'plus']
        rows.append({
            'attacker_model': attacker,
            'vanilla_breach_rate': van['breach'].mean(),
            'plus_breach_rate': plu['breach'].mean(),
            'delta_breach_pp': (plu['breach'].mean() - van['breach'].mean()) * 100.0,
            'vanilla_impact_mean': van['impact'].mean(),
            'plus_impact_mean': plu['impact'].mean(),
        })
    return pd.DataFrame(rows)


def summarize_datasetwise(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ['celeba_pairs', 'lfw_pairs', 'vggface2_pairs']:
        g = long_df[long_df['dataset'] == dataset]
        van = g[g['variant'] == 'vanilla']
        plu = g[g['variant'] == 'plus']
        rows.append({
            'dataset': dataset,
            'dataset_label': DATASET_LABELS[dataset],
            'vanilla_breach_rate': van['breach'].mean(),
            'plus_breach_rate': plu['breach'].mean(),
            'delta_breach_pp': (plu['breach'].mean() - van['breach'].mean()) * 100.0,
            'vanilla_impact_mean': van['impact'].mean(),
            'plus_impact_mean': plu['impact'].mean(),
        })
    return pd.DataFrame(rows)


def summarize_cross_model(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (attacker, victim), g in long_df.groupby(['attacker_model', 'victim_model'], sort=True):
        van = g[g['variant'] == 'vanilla']
        plu = g[g['variant'] == 'plus']
        rows.append({
            'attacker_model': attacker,
            'victim_model': victim,
            'vanilla_breach_rate': van['breach'].mean(),
            'plus_breach_rate': plu['breach'].mean(),
            'delta_breach_pp': (plu['breach'].mean() - van['breach'].mean()) * 100.0,
            'vanilla_impact_mean': van['impact'].mean(),
            'plus_impact_mean': plu['impact'].mean(),
        })
    return pd.DataFrame(rows).sort_values(['attacker_model', 'victim_model'], kind='stable').reset_index(drop=True)


def plot_pairwise(pair_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {'Impersonation': '#4C78A8', 'Dodging': '#F58518'}
    x = np.arange(len(ATTACK_ORDER))
    width = 0.38
    for idx, setting in enumerate(['Impersonation', 'Dodging']):
        grp = pair_df[pair_df['setting'] == setting].set_index('attack').loc[ATTACK_ORDER].reset_index()
        axes[0].bar(x + (idx - 0.5) * width, grp['delta_breach_pp'], width=width, label=setting, color=colors[setting])
        axes[1].bar(x + (idx - 0.5) * width, grp['delta_impact'], width=width, label=setting, color=colors[setting])
    axes[0].set_xticks(x, ATTACK_ORDER, rotation=20, ha='right')
    axes[0].set_ylabel('Delta Breach (pp)')
    axes[0].set_title('Pairwise FaceSM Gain by Attack')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend()
    axes[1].set_xticks(x, ATTACK_ORDER, rotation=20, ha='right')
    axes[1].set_ylabel('Delta Impact')
    axes[1].set_title('Impact Gain by Attack')
    axes[1].grid(axis='y', alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_model_dataset(model_df: pd.DataFrame, dataset_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    # model-wise breach
    x = np.arange(len(model_df))
    width = 0.36
    axes[0].bar(x - width / 2, model_df['vanilla_breach_rate'] * 100.0, width=width, label='Vanilla', color='#9E9E9E')
    axes[0].bar(x + width / 2, model_df['plus_breach_rate'] * 100.0, width=width, label='FaceSM', color='#2E8B57')
    axes[0].set_xticks(x, model_df['attacker_model'])
    axes[0].set_ylabel('Breach Rate (%)')
    axes[0].set_title('Surrogate-wise Breach Rate')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend()
    # dataset-wise breach
    x2 = np.arange(len(dataset_df))
    axes[1].bar(x2 - width / 2, dataset_df['vanilla_breach_rate'] * 100.0, width=width, label='Vanilla', color='#9E9E9E')
    axes[1].bar(x2 + width / 2, dataset_df['plus_breach_rate'] * 100.0, width=width, label='FaceSM', color='#2E8B57')
    axes[1].set_xticks(x2, dataset_df['dataset_label'])
    axes[1].set_ylabel('Breach Rate (%)')
    axes[1].set_title('Dataset-wise Breach Rate')
    axes[1].grid(axis='y', alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_modelwise(model_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    x = np.arange(len(model_df))
    width = 0.36
    ax.bar(x - width / 2, model_df['vanilla_breach_rate'] * 100.0, width=width, label='Vanilla', color='#9E9E9E')
    ax.bar(x + width / 2, model_df['plus_breach_rate'] * 100.0, width=width, label='FaceSM', color='#2E8B57')
    ax.set_xticks(x, model_df['attacker_model'])
    ax.set_ylabel('Breach Rate (%)')
    ax.set_title('Surrogate-wise Breach Rate')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_datasetwise(dataset_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    x = np.arange(len(dataset_df))
    width = 0.36
    ax.bar(x - width / 2, dataset_df['vanilla_breach_rate'] * 100.0, width=width, label='Vanilla', color='#9E9E9E')
    ax.bar(x + width / 2, dataset_df['plus_breach_rate'] * 100.0, width=width, label='FaceSM', color='#2E8B57')
    ax.set_xticks(x, dataset_df['dataset_label'])
    ax.set_ylabel('Breach Rate (%)')
    ax.set_title('Dataset-wise Breach Rate')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cross_model(cross_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    victims = VICTIM_LABELS
    attackers = ATTACKER_LABELS
    x = np.arange(len(victims))
    width = 0.18
    colors = ['#4C78A8', '#F58518', '#54A24B', '#B279A2']
    piv = cross_df.pivot(index='victim_model', columns='attacker_model', values='delta_breach_pp').reindex(victims)
    for idx, attacker in enumerate(attackers):
        vals = piv[attacker] if attacker in piv.columns else pd.Series(np.nan, index=victims)
        pos = x - 1.5 * width + idx * width
        ax.bar(pos, vals, width=width, label=attacker, color=colors[idx])
    ax.set_xticks(x, victims, rotation=20, ha='right')
    ax.set_ylabel('Delta Breach (pp)')
    ax.set_title('Cross-Model FaceSM Gain by Victim')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_ablation(ablation_df: pd.DataFrame, out_path: Path) -> None:
    order = ['vanilla', 'mf_only', 'ss_only', 'facesm']
    labels = ['Vanilla', 'MF Only', 'SS Only', 'FaceSM']
    df = ablation_df.set_index('config_key').loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    colors = ['#9E9E9E', '#4C78A8', '#F58518', '#2E8B57']
    axes[0].bar(labels, df['breach_rate'] * 100.0, color=colors)
    axes[0].set_ylabel('Breach Rate (%)')
    axes[0].set_title('Ablation: Breach Rate')
    axes[0].grid(axis='y', alpha=0.25)
    axes[1].bar(labels, df['impact_mean'], color=colors)
    axes[1].set_ylabel('Impact')
    axes[1].set_title('Ablation: Impact')
    axes[1].grid(axis='y', alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_lambda(lambda_df: pd.DataFrame, out_path: Path) -> None:
    overall = (
        lambda_df.groupby('lambda_value', as_index=False)
        .agg(breach_rate=('breach_rate', 'mean'), impact_mean=('impact_mean', 'mean'))
        .sort_values('lambda_value', kind='stable')
    )
    # Keep the summary CSV complete, but omit 0.25 from the plot to make the
    # trend easier to read in the paper figure.
    overall_plot = overall[~np.isclose(overall['lambda_value'], 0.25)].copy()
    chosen_lambda = 0.20
    tick_vals = overall_plot['lambda_value'].tolist()

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    breach_vals = overall_plot['breach_rate'] * 100.0
    impact_vals = overall_plot['impact_mean']

    # Left: breach rate
    axes[0].plot(
        overall_plot['lambda_value'],
        breach_vals,
        marker='o',
        markersize=5.5,
        color='#3A6EA5',
        linewidth=2.2,
    )
    chosen_row = overall_plot[np.isclose(overall_plot['lambda_value'], chosen_lambda)]
    if not chosen_row.empty:
        axes[0].scatter(
            chosen_row['lambda_value'],
            chosen_row['breach_rate'] * 100.0,
            s=46,
            color='#C0392B',
            zorder=5,
            label=r'Selected $\lambda=0.20$',
        )
    axes[0].axvline(chosen_lambda, color='#C0392B', linestyle=':', linewidth=1.3, alpha=0.9)
    axes[0].set_xlabel(r'$\lambda$')
    axes[0].set_ylabel('Breach Rate (%)')
    axes[0].set_xticks(tick_vals)
    axes[0].set_xticklabels([f'{x:.2f}'.rstrip('0').rstrip('.') for x in tick_vals], rotation=0)
    axes[0].grid(alpha=0.22, linewidth=0.8)
    axes[0].legend(loc='lower left', frameon=False)

    # Right: impact
    axes[1].plot(
        overall_plot['lambda_value'],
        impact_vals,
        marker='o',
        markersize=5.5,
        color='#E67E22',
        linewidth=2.2,
    )
    if not chosen_row.empty:
        axes[1].scatter(
            chosen_row['lambda_value'],
            chosen_row['impact_mean'],
            s=46,
            color='#C0392B',
            zorder=5,
        )
    axes[1].axvline(chosen_lambda, color='#C0392B', linestyle=':', linewidth=1.3, alpha=0.9)
    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel('Impact')
    axes[1].set_xticks(tick_vals)
    axes[1].set_xticklabels([f'{x:.2f}'.rstrip('0').rstrip('.') for x in tick_vals], rotation=0)
    axes[1].grid(alpha=0.22, linewidth=0.8)

    for ax in axes:
        ax.spines['top'].set_alpha(0.5)
        ax.spines['right'].set_alpha(0.5)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_lambda_vertical(lambda_df: pd.DataFrame, out_path: Path) -> None:
    overall = (
        lambda_df.groupby('lambda_value', as_index=False)
        .agg(breach_rate=('breach_rate', 'mean'), impact_mean=('impact_mean', 'mean'))
        .sort_values('lambda_value', kind='stable')
    )
    overall_plot = overall[~np.isclose(overall['lambda_value'], 0.25)].copy()
    chosen_lambda = 0.20
    tick_vals = overall_plot['lambda_value'].tolist()

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.2), constrained_layout=True)

    breach_vals = overall_plot['breach_rate'] * 100.0
    impact_vals = overall_plot['impact_mean']
    chosen_row = overall_plot[np.isclose(overall_plot['lambda_value'], chosen_lambda)]
    tick_labels = [f'{x:.2f}'.rstrip('0').rstrip('.') for x in tick_vals]

    axes[0].plot(
        overall_plot['lambda_value'],
        breach_vals,
        marker='o',
        markersize=6,
        color='#3A6EA5',
        linewidth=2.3,
    )
    if not chosen_row.empty:
        axes[0].scatter(
            chosen_row['lambda_value'],
            chosen_row['breach_rate'] * 100.0,
            s=50,
            color='#C0392B',
            zorder=5,
            label=r'Selected $\lambda=0.20$',
        )
    axes[0].axvline(chosen_lambda, color='#C0392B', linestyle=':', linewidth=1.3, alpha=0.9)
    axes[0].set_ylabel('Breach Rate (%)')
    axes[0].set_xticks(tick_vals)
    axes[0].set_xticklabels(tick_labels)
    axes[0].grid(alpha=0.22, linewidth=0.8)
    axes[0].legend(loc='lower left', frameon=False)

    axes[1].plot(
        overall_plot['lambda_value'],
        impact_vals,
        marker='o',
        markersize=6,
        color='#E67E22',
        linewidth=2.3,
    )
    if not chosen_row.empty:
        axes[1].scatter(
            chosen_row['lambda_value'],
            chosen_row['impact_mean'],
            s=50,
            color='#C0392B',
            zorder=5,
        )
    axes[1].axvline(chosen_lambda, color='#C0392B', linestyle=':', linewidth=1.3, alpha=0.9)
    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel('Impact')
    axes[1].set_xticks(tick_vals)
    axes[1].set_xticklabels(tick_labels)
    axes[1].grid(alpha=0.22, linewidth=0.8)

    for ax in axes:
        ax.spines['top'].set_alpha(0.5)
        ax.spines['right'].set_alpha(0.5)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    long_df = load_long_results()
    pair_df = summarize_pairwise(long_df)
    model_df = summarize_modelwise(long_df)
    dataset_df = summarize_datasetwise(long_df)
    cross_df = summarize_cross_model(long_df)

    pair_df.to_csv(OUT_DIR / 'pairwise_split_summary.csv', index=False)
    model_df.to_csv(OUT_DIR / 'modelwise_summary.csv', index=False)
    dataset_df.to_csv(OUT_DIR / 'datasetwise_summary.csv', index=False)
    cross_df.to_csv(OUT_DIR / 'cross_model_summary.csv', index=False)

    ablation_df = pd.read_csv(RESULTS_ROOT / 'ablation_sm_paper' / 'ablation_overall_summary.csv')
    lambda_df = pd.read_csv(RESULTS_ROOT / 'lambda_sweep_sm_paper' / 'lambda_sweep_cumulative_summary.csv')
    ablation_df.to_csv(OUT_DIR / 'ablation_overall_summary.csv', index=False)
    lambda_df.to_csv(OUT_DIR / 'lambda_sweep_summary.csv', index=False)

    plot_pairwise(pair_df, OUT_DIR / 'charts' / 'pairwise_setting_gains.png')
    plot_model_dataset(model_df, dataset_df, OUT_DIR / 'charts' / 'model_dataset_breach.png')
    plot_modelwise(model_df, OUT_DIR / 'charts' / 'surrogate_breach.png')
    plot_datasetwise(dataset_df, OUT_DIR / 'charts' / 'dataset_breach.png')
    plot_cross_model(cross_df, OUT_DIR / 'charts' / 'cross_model_delta_breach.png')
    plot_ablation(ablation_df, OUT_DIR / 'charts' / 'ablation_components.png')
    plot_lambda(lambda_df, OUT_DIR / 'charts' / 'lambda_sweep.png')
    plot_lambda_vertical(lambda_df, OUT_DIR / 'charts' / 'lambda_sweep_vertical.png')

    print('Wrote summaries to', OUT_DIR)
    print('Wrote charts to', OUT_DIR / 'charts')


if __name__ == '__main__':
    main()
