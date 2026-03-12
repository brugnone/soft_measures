"""
Compare Pearson r(GT vs AI) for node and edge counts:
  reasoning vs no-reasoning, per model × dataset, with Fisher z-tests.
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(r'C:\Users\Nbrug\Desktop\all_fcm_results_combined.csv')

MODEL_PAIRS = [
    ('gpt5mini',     'gpt5mini_nr',     'GPT-5-mini'),
    ('gpt52',        'gpt52_nr',        'GPT-5.2'),
    ('gemini25flash','gemini25flash_nr', 'Gemini 2.5 Flash'),
    ('gemini3flash', 'gemini3flash_nr',  'Gemini 3 Flash'),
    ('qwen',         'qwen_nr',          'Qwen'),
    ('mistral',      'mistral_nr',       'Mistral'),
]

DATASETS  = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DS_LABEL  = {'biodiversity': 'Biodiv.', 'flpp': 'FLPP',
             'osw': 'OSW', 'red_snapper': 'RedSnap'}

def pearson(a, b):
    mask = pd.notna(a) & pd.notna(b)
    n = mask.sum()
    if n < 3:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(a[mask], b[mask])
    return round(r, 3), round(p, 4), n

def fisher_z_test(r1, r2, n1, n2):
    """Two-sample Fisher z-test for equality of two independent correlations."""
    if any(np.isnan(x) for x in [r1, r2]) or n1 < 4 or n2 < 4:
        return np.nan, np.nan
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return round(z_stat, 2), round(p_val, 4)

def bootstrap_delta_r(gt_r, ai_r, gt_nr, ai_nr, B=9999, seed=42):
    """
    Bootstrap p-value for H0: r_R == r_NR (two independent samples).

    Strategy: resample each group independently; shift the null distribution
    of Δr_boot to be centred on 0.  p = fraction of |Δr_boot_shifted| >= |Δr_obs|.
    Also returns 95% percentile CI for Δr.
    """
    rng = np.random.default_rng(seed)

    def _r(a, b):
        if len(a) < 3:
            return np.nan
        try:
            return stats.pearsonr(a, b)[0]
        except Exception:
            return np.nan

    # Drop NAs
    mask_r  = pd.notna(gt_r)  & pd.notna(ai_r)
    mask_nr = pd.notna(gt_nr) & pd.notna(ai_nr)
    g_r,  a_r  = np.asarray(gt_r[mask_r]),  np.asarray(ai_r[mask_r])
    g_nr, a_nr = np.asarray(gt_nr[mask_nr]), np.asarray(ai_nr[mask_nr])

    if len(g_r) < 4 or len(g_nr) < 4:
        return np.nan, (np.nan, np.nan)

    obs_delta = _r(g_r, a_r) - _r(g_nr, a_nr)
    if np.isnan(obs_delta):
        return np.nan, (np.nan, np.nan)

    boot_deltas = np.empty(B)
    for i in range(B):
        idx_r  = rng.integers(0, len(g_r),  size=len(g_r))
        idx_nr = rng.integers(0, len(g_nr), size=len(g_nr))
        boot_deltas[i] = (_r(g_r[idx_r], a_r[idx_r]) -
                          _r(g_nr[idx_nr], a_nr[idx_nr]))

    # Shift to centre at 0 for null hypothesis test
    centred = boot_deltas - np.nanmean(boot_deltas)
    p_val   = np.mean(np.abs(centred) >= np.abs(obs_delta))
    ci_lo, ci_hi = np.nanpercentile(boot_deltas, [2.5, 97.5])
    return round(float(p_val), 4), (round(float(ci_lo), 3), round(float(ci_hi), 3))

# ── Per-dataset r table ───────────────────────────────────────────────────────
for metric_label, col_gt, col_ai in [
    ('NODES (r between GT and AI node count)',  'gt_nodes', 'ai_nodes'),
    ('EDGES (r between GT and AI edge count)',  'gt_edges', 'ai_edges'),
]:
    print('=' * 90)
    print(metric_label)
    print('=' * 90)
    header = f'{"Model":<22} {"Cond":<4}' + \
             ''.join(f'  {DS_LABEL[d]:>8}' for d in DATASETS) + \
             f'  {"ALL":>8}  {"Δr(R-NR)":>9}  {"z":>6}  {"p":>7}'
    print(header)
    print('-' * 90)

    for r_key, nr_key, label in MODEL_PAIRS:
        rows = {}
        for cond, key in [('R', r_key), ('NR', nr_key)]:
            sub = df[df['method'] == key]
            row_vals = []
            for ds in DATASETS:
                s = sub[sub['dataset'] == ds]
                r_val, p, n = pearson(s[col_gt], s[col_ai])
                sig = '*' if (not np.isnan(p) and p < 0.05) else ' '
                row_vals.append((r_val, sig, n))
            all_s = sub.dropna(subset=[col_gt, col_ai])
            r_all, p_all, n_all = pearson(all_s[col_gt], all_s[col_ai])
            sig_all = '*' if (not np.isnan(p_all) and p_all < 0.05) else ' '
            rows[cond] = (row_vals, r_all, sig_all, n_all)

        # Print R row
        rv, r_all, sig_all, n_all = rows['R']
        line = f'{label:<22} {"R":<4}'
        line += ''.join(f'  {v[0]:>7.3f}{v[1]}' if not np.isnan(v[0]) else f'  {"nan":>7} ' for v in rv)
        line += f'  {r_all:>7.3f}{sig_all}'
        print(line)

        # Print NR row + delta + Fisher z
        rv_nr, r_all_nr, sig_all_nr, n_all_nr = rows['NR']
        line = f'{"":22} {"NR":<4}'
        line += ''.join(f'  {v[0]:>7.3f}{v[1]}' if not np.isnan(v[0]) else f'  {"nan":>7} ' for v in rv_nr)
        line += f'  {r_all_nr:>7.3f}{sig_all_nr}'

        z_stat, p_val = fisher_z_test(r_all, r_all_nr, rows['R'][3], rows['NR'][3])
        delta = (r_all - r_all_nr) if not (np.isnan(r_all) or np.isnan(r_all_nr)) else np.nan
        sig_fisher = ' *' if (not np.isnan(p_val) and p_val < 0.05) else \
                     '  .' if (not np.isnan(p_val) and p_val < 0.10) else '   '
        dr_str = f'{delta:>+9.3f}' if not np.isnan(delta) else f'{"nan":>9}'
        z_str  = f'{z_stat:>6.2f}' if not np.isnan(z_stat) else f'{"nan":>6}'
        p_str  = f'{p_val:>7.4f}' if not np.isnan(p_val) else f'{"nan":>7}'
        line  += f'{dr_str}  {z_str}  {p_str}{sig_fisher}'
        print(line)
        print()

# ── Dataset-level bootstrap tests (significant / trending only) ──────────────
print()
print('=' * 100)
print('DATASET-LEVEL bootstrap p-values for Δr (R vs NR), B=9999  [all combinations]')
print('=' * 100)
print(f'{"Model":<22} {"Metric":<7} {"Dataset":<12} {"r_R":>7}  {"r_NR":>7}  {"Δr":>8}  '
      f'{"Fisher-p":>9}  {"Boot-p":>7}  {"95% CI Δr":>14}  sig')
print('-' * 100)

any_printed = False
for metric_label, col_gt, col_ai in [
    ('Nodes', 'gt_nodes', 'ai_nodes'),
    ('Edges', 'gt_edges', 'ai_edges'),
]:
    for r_key, nr_key, label in MODEL_PAIRS:
        sub_r  = df[df['method'] == r_key]
        sub_nr = df[df['method'] == nr_key]
        for ds in DATASETS:
            s_r  = sub_r[sub_r['dataset']  == ds]
            s_nr = sub_nr[sub_nr['dataset'] == ds]
            r_r,  _, n_r  = pearson(s_r[col_gt],  s_r[col_ai])
            r_nr, _, n_nr = pearson(s_nr[col_gt], s_nr[col_ai])
            # Fisher z (parametric)
            _, fz_p = fisher_z_test(r_r, r_nr, n_r, n_nr)
            # Bootstrap
            bp, ci = bootstrap_delta_r(
                s_r[col_gt].reset_index(drop=True),  s_r[col_ai].reset_index(drop=True),
                s_nr[col_gt].reset_index(drop=True), s_nr[col_ai].reset_index(drop=True),
            )
            if np.isnan(bp):
                continue
            delta = r_r - r_nr
            sig = '**' if bp < 0.05 else (' .' if bp < 0.10 else '  ')
            fz_str = f'{fz_p:>9.4f}' if not np.isnan(fz_p) else f'{"nan":>9}'
            ci_str = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'
            # Print all rows (remove filter) so user can see full picture
            print(f'{label:<22} {metric_label:<7} {ds:<12} '
                  f'{r_r:>7.3f}  {r_nr:>7.3f}  {delta:>+8.3f}  '
                  f'{fz_str}  {bp:>7.4f}  {ci_str:>14}  {sig}')
            any_printed = True
        print()  # blank between models within a metric

print()
print('sig: ** = boot p < .05,  . = boot p < .10')
print()

# ── Grand summary: average AI nodes/edges vs GT ───────────────────────────────
print('=' * 70)
print('Mean AI vs GT counts (all datasets pooled)')
print('=' * 70)
print(f'{"Model":<22} {"Cond":<4}  {"Mean GT nodes":>13}  {"Mean AI nodes":>13}  {"Mean GT edges":>13}  {"Mean AI edges":>13}')
print('-' * 70)
for r_key, nr_key, label in MODEL_PAIRS:
    for cond, key in [('R', r_key), ('NR', nr_key)]:
        sub = df[df['method'] == key]
        print(f'{label:<22} {cond:<4}  '
              f'{sub["gt_nodes"].mean():>13.1f}  {sub["ai_nodes"].mean():>13.1f}  '
              f'{sub["gt_edges"].mean():>13.1f}  {sub["ai_edges"].mean():>13.1f}')
    print()
