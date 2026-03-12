"""
Variance analysis of soft F1 scores: reasoning vs no-reasoning per model.
  - Descriptives: mean, SD, IQR, CV per model × condition × dataset
  - Levene's test for equality of variance (R vs NR)
  - Figure: SD heatmap (model × dataset) + overall SD bar chart
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DESKTOP = Path(r'C:\Users\Nbrug\Desktop')
OUT_DIR = DESKTOP / 'fcm_paired_analysis'
OUT_DIR.mkdir(exist_ok=True)

PP_WEIGHT = 0.6

MODEL_PAIRS = [
    ('gpt5mini',      'gpt5mini_nr',      'GPT-5-mini'),
    ('gpt52',         'gpt52_nr',         'GPT-5.2'),
    ('gemini25flash', 'gemini25flash_nr',  'Gemini 2.5 Flash'),
    ('gemini3flash',  'gemini3flash_nr',   'Gemini 3 Flash'),
    ('qwen',          'qwen_nr',           'Qwen'),
    ('mistral',       'mistral_nr',        'Mistral'),
]

DATASETS = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DS_LABELS = {'biodiversity': 'Biodiversity', 'flpp': 'FLPP',
             'osw': 'Gulf OSW', 'red_snapper': 'Red Snapper'}

MODEL_COLORS = {
    'gpt5mini':      '#2E86AB',
    'gpt52':         '#3BB273',
    'gemini25flash': '#E87040',
    'gemini3flash':  '#9B5DE5',
    'qwen':          '#E84855',
    'mistral':       '#FF9F1C',
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DESKTOP / 'all_fcm_results_combined.csv')
num = 2 * df['TP'] + PP_WEIGHT * df['PP']
df['soft_F1'] = (num / (num + df['FP'] + df['FN'])).replace([np.inf, -np.inf], np.nan)

# ── 1. Full descriptives table ────────────────────────────────────────────────
print('='*100)
print('SOFT F1 DESCRIPTIVES  (mean ± SD, IQR, CV)  per model × condition × dataset')
print('='*100)
hdr = f'{"Model":<22} {"Cond":<4} {"Dataset":<14}  {"n":>4}  {"Mean":>6}  {"SD":>6}  {"IQR":>6}  {"CV%":>6}'
print(hdr)
print('-'*100)

desc_rows = []
for r_key, nr_key, label in MODEL_PAIRS:
    for cond, key in [('R', r_key), ('NR', nr_key)]:
        sub_all = df[df['method'] == key]['soft_F1'].dropna()
        for ds in DATASETS + ['ALL']:
            if ds == 'ALL':
                vals = sub_all
                ds_lbl = 'ALL'
            else:
                vals = df[(df['method'] == key) & (df['dataset'] == ds)]['soft_F1'].dropna()
                ds_lbl = DS_LABELS[ds]
            if len(vals) < 2:
                continue
            mean_ = vals.mean()
            sd_   = vals.std()
            iqr_  = vals.quantile(0.75) - vals.quantile(0.25)
            cv_   = (sd_ / mean_ * 100) if mean_ > 0 else np.nan
            print(f'{label:<22} {cond:<4} {ds_lbl:<14}  {len(vals):>4}  '
                  f'{mean_:>6.3f}  {sd_:>6.3f}  {iqr_:>6.3f}  {cv_:>6.1f}')
            desc_rows.append({'model': label, 'condition': cond, 'dataset': ds_lbl,
                               'n': len(vals), 'mean': mean_, 'sd': sd_,
                               'iqr': iqr_, 'cv': cv_, 'key': key})
    print()

pd.DataFrame(desc_rows).to_csv(OUT_DIR / 'soft_f1_variance_descriptives.csv', index=False)

# ── 2. Levene's test: R vs NR variance per model × dataset ───────────────────
print()
print('='*90)
print("LEVENE'S TEST for equality of variance: R vs NR per model × dataset")
print('='*90)
print(f'{"Model":<22} {"Dataset":<14}  {"SD_R":>6}  {"SD_NR":>6}  {"ΔSD":>7}  '
      f'{"Lev-F":>7}  {"p":>7}  sig')
print('-'*90)

lev_rows = []
for r_key, nr_key, label in MODEL_PAIRS:
    for ds in DATASETS + ['ALL']:
        if ds == 'ALL':
            r_vals  = df[df['method'] == r_key ]['soft_F1'].dropna().values
            nr_vals = df[df['method'] == nr_key]['soft_F1'].dropna().values
            ds_lbl  = 'ALL'
        else:
            r_vals  = df[(df['method'] == r_key)  & (df['dataset'] == ds)]['soft_F1'].dropna().values
            nr_vals = df[(df['method'] == nr_key) & (df['dataset'] == ds)]['soft_F1'].dropna().values
            ds_lbl  = DS_LABELS[ds]
        if len(r_vals) < 3 or len(nr_vals) < 3:
            continue
        lev_f, lev_p = stats.levene(r_vals, nr_vals, center='median')
        sd_r  = r_vals.std()
        sd_nr = nr_vals.std()
        delta = sd_r - sd_nr
        sig   = '**' if lev_p < 0.05 else (' .' if lev_p < 0.10 else '  ')
        print(f'{label:<22} {ds_lbl:<14}  {sd_r:>6.3f}  {sd_nr:>6.3f}  '
              f'{delta:>+7.3f}  {lev_f:>7.3f}  {lev_p:>7.4f}  {sig}')
        lev_rows.append({'model': label, 'dataset': ds_lbl,
                          'sd_r': sd_r, 'sd_nr': sd_nr, 'delta_sd': delta,
                          'levene_F': lev_f, 'levene_p': lev_p})
    print()

lev_df = pd.DataFrame(lev_rows)
lev_df.to_csv(OUT_DIR / 'soft_f1_levene_tests.csv', index=False)

# ── 3. Figure A: overall SD bar chart (R vs NR side-by-side) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
labels = [lbl for _, _, lbl in MODEL_PAIRS]

for ax, (metric, metric_lbl) in zip(axes, [('sd', 'SD (soft F1)'), ('iqr', 'IQR (soft F1)')]):
    r_vals_plot  = []
    nr_vals_plot = []
    for _, _, lbl in MODEL_PAIRS:
        r_row  = next((r for r in desc_rows if r['model']==lbl and r['condition']=='R'  and r['dataset']=='ALL'), None)
        nr_row = next((r for r in desc_rows if r['model']==lbl and r['condition']=='NR' and r['dataset']=='ALL'), None)
        r_vals_plot.append(r_row[metric]  if r_row  else np.nan)
        nr_vals_plot.append(nr_row[metric] if nr_row else np.nan)

    x = np.arange(len(labels))
    w = 0.35
    bars_r  = ax.bar(x - w/2, r_vals_plot,  w, label='Reasoning',    color=[MODEL_COLORS[k] for k,_,_ in MODEL_PAIRS], alpha=0.9)
    bars_nr = ax.bar(x + w/2, nr_vals_plot, w, label='No Reasoning', color=[MODEL_COLORS[k] for k,_,_ in MODEL_PAIRS], alpha=0.45,
                     edgecolor=[MODEL_COLORS[k] for k,_,_ in MODEL_PAIRS], linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(metric_lbl, fontsize=10)
    ax.set_title(f'Overall {metric_lbl}: R (solid) vs NR (light)', fontsize=10)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_ylim(0, max(max(r_vals_plot), max(nr_vals_plot)) * 1.3)
    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars_nr:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

fig.suptitle('Soft F1 Variability: Reasoning vs No-Reasoning (all datasets pooled)', fontsize=12)
plt.tight_layout()
out_a = OUT_DIR / 'figD_variance_overall.png'
fig.savefig(out_a, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved → {out_a}')

# ── 4. Figure B: SD heatmap  model × dataset  (R minus NR) ───────────────────
# Build matrix: rows=models, cols=datasets, value = SD_R - SD_NR
mat_delta = np.full((len(MODEL_PAIRS), len(DATASETS)), np.nan)
mat_r     = np.full((len(MODEL_PAIRS), len(DATASETS)), np.nan)
mat_nr    = np.full((len(MODEL_PAIRS), len(DATASETS)), np.nan)

for i, (r_key, nr_key, label) in enumerate(MODEL_PAIRS):
    for j, ds in enumerate(DATASETS):
        r_s  = df[(df['method'] == r_key)  & (df['dataset'] == ds)]['soft_F1'].dropna()
        nr_s = df[(df['method'] == nr_key) & (df['dataset'] == ds)]['soft_F1'].dropna()
        if len(r_s) > 1 and len(nr_s) > 1:
            mat_r[i, j]     = r_s.std()
            mat_nr[i, j]    = nr_s.std()
            mat_delta[i, j] = r_s.std() - nr_s.std()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ds_lbls = [DS_LABELS[d] for d in DATASETS]
mdl_lbls = [lbl for _, _, lbl in MODEL_PAIRS]

for ax, mat, title, cmap, vcenter in [
    (axes[0], mat_r,     'SD (Reasoning)',    'Blues',  None),
    (axes[1], mat_nr,    'SD (No Reasoning)', 'Blues',  None),
    (axes[2], mat_delta, 'ΔSD (R − NR)',      'RdBu_r', 0),
]:
    vmax = np.nanmax(np.abs(mat)) if vcenter == 0 else np.nanmax(mat)
    vmin = -vmax if vcenter == 0 else 0
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(ds_lbls)))
    ax.set_xticklabels(ds_lbls, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(mdl_lbls)))
    ax.set_yticklabels(mdl_lbls, fontsize=9)
    ax.set_title(title, fontsize=10)
    # annotate cells
    for ii in range(mat.shape[0]):
        for jj in range(mat.shape[1]):
            val = mat[ii, jj]
            if not np.isnan(val):
                ax.text(jj, ii, f'{val:.3f}', ha='center', va='center',
                        fontsize=7.5,
                        color='white' if abs(val) > (vmax * 0.6) else 'black')

fig.suptitle('Soft F1 Standard Deviation by Model × Dataset', fontsize=12)
plt.tight_layout()
out_b = OUT_DIR / 'figE_variance_heatmap.png'
fig.savefig(out_b, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved → {out_b}')
print('\nDone. CSVs and figures saved to', OUT_DIR)
