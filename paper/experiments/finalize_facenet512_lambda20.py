#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'Facenet512'

new_paths = RES / 'transfer_adv_paths_all12_lambda020_limit1000_sm.csv'
old_paths = RES / 'transfer_adv_paths_all12_baseline_legacy.csv'
new_sim = RES / 'transfer_attack_similarity_scores_all12_lambda020_limit1000_sm_only.csv'
old_sim = RES / 'transfer_attack_similarity_scores_all12_baseline_legacy.csv'
new_perf = RES / 'transfer_attack_performance_all12_lambda020_limit1000_sm_only.csv'
old_perf = RES / 'transfer_attack_performance_all12_baseline_legacy.csv'

final_paths = RES / 'transfer_adv_paths_all12_lambda020_limit1000.csv'
final_sim = RES / 'transfer_attack_similarity_scores_all12_lambda020_limit1000.csv'
final_perf = RES / 'transfer_attack_performance_all12_lambda020_limit1000.csv'

KEY_PATH = ['row_id']
KEY_PERF = ['row_id','attacker_model','victim_model','dataset','attack_type']
PATH_COLS = ['pgd_path','mi_fgsm_path','ti_fgsm_path','si_ni_fgsm_path','mi_admix_di_ti_path','rap_path',
             'pgd_sm_path','mi_fgsm_sm_path','ti_fgsm_sm_path','mi_admix_di_ti_sm_path','si_ni_fgsm_sm_path','rap_sm_path']
SM_PATH_COLS = ['pgd_sm_path','mi_fgsm_sm_path','ti_fgsm_sm_path','mi_admix_di_ti_sm_path','si_ni_fgsm_sm_path','rap_sm_path']

def overlay(base: pd.DataFrame, fresh: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = base.copy()
    if out.empty:
        return fresh.copy()
    out = out.set_index(keys)
    fresh = fresh.set_index(keys)
    all_cols = list(dict.fromkeys(list(out.columns) + list(fresh.columns)))
    out = out.reindex(columns=all_cols)
    for idx, row in fresh.iterrows():
        if idx not in out.index:
            out.loc[idx, :] = np.nan
        for col, val in row.items():
            if pd.isna(val):
                continue
            if isinstance(val, str) and val.strip() == '':
                continue
            out.at[idx, col] = val
    return out.reset_index()

# Paths
newp = pd.read_csv(new_paths)
target_ids = sorted(newp['row_id'].unique().tolist())
oldp = pd.read_csv(old_paths)
oldp = oldp[oldp['row_id'].isin(target_ids)].copy()
# prefer old rows for vanilla paths, overlay fresh SM paths and metadata from new
if oldp.empty:
    merged_paths = newp.copy()
else:
    merged_paths = overlay(oldp, newp, KEY_PATH)
# keep one row per row_id in target order
merged_paths = merged_paths.sort_values('row_id').drop_duplicates('row_id', keep='last')
# preserve canonical column order when possible
ordered_cols = [c for c in ['row_id','attacker_model','img1','img2','dataset','attack_type'] + PATH_COLS if c in merged_paths.columns]
rest = [c for c in merged_paths.columns if c not in ordered_cols]
merged_paths = merged_paths[ordered_cols + rest]
merged_paths.to_csv(final_paths, index=False)
print('wrote', final_paths, 'rows', len(merged_paths), 'unique_row_id', merged_paths['row_id'].nunique())

# Similarity
news = pd.read_csv(new_sim)
olds = pd.read_csv(old_sim)
olds = olds[olds['row_id'].isin(target_ids)].copy()
merged_sim = overlay(olds, news, KEY_PATH) if not olds.empty else news.copy()
merged_sim = merged_sim.sort_values('row_id').drop_duplicates('row_id', keep='last')
merged_sim.to_csv(final_sim, index=False)
print('wrote', final_sim, 'rows', len(merged_sim), 'unique_row_id', merged_sim['row_id'].nunique())

# Performance
newf = pd.read_csv(new_perf)
oldf = pd.read_csv(old_perf)
# Keep only the victim models actually rescored in the fresh SM-only file
victims = sorted(newf['victim_model'].dropna().unique().tolist())
oldf = oldf[oldf['row_id'].isin(target_ids) & oldf['victim_model'].isin(victims)].copy()
merged_perf = overlay(oldf, newf, KEY_PERF) if not oldf.empty else newf.copy()
merged_perf = merged_perf.sort_values(KEY_PERF).drop_duplicates(KEY_PERF, keep='last')
merged_perf.to_csv(final_perf, index=False)
print('wrote', final_perf, 'rows', len(merged_perf), 'unique_row_id', merged_perf['row_id'].nunique(), 'victims', sorted(merged_perf['victim_model'].dropna().unique().tolist()))
