"""
Paired analysis: soft-F1 difference (reasoning – no reasoning) for each model.

Steps:
  1. Load all_fcm_results_combined.csv
  2. Compute soft_F1 = (2*TP + 0.6*PP) / (2*TP + 0.6*PP + FP + FN)
  3. For each model pair, merge on (dataset, file_id) and compute delta
  4. Summarise by model_pair × dataset and overall
  5. Organise individual result CSVs into a tidy directory tree
  6. Save summary tables and a combined paired CSV

Output directory: C:\\Users\\Nbrug\\Desktop\\fcm_paired_analysis\\
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
DESKTOP  = Path(r'C:\Users\Nbrug\Desktop')
OUT_DIR  = DESKTOP / 'fcm_paired_analysis'
ORG_DIR  = OUT_DIR / 'results_by_model'   # organised CSVs
OUT_DIR.mkdir(exist_ok=True)
ORG_DIR.mkdir(exist_ok=True)

MODEL_PAIRS = [
    'gpt5mini',
    'gpt52',
    'gemini25flash',
    'gemini3flash',
    'qwen',
    'mistral',
]

MODEL_DISPLAY = {
    'gpt5mini':      'GPT-5-mini',
    'gpt52':         'GPT-5.2',
    'gemini25flash': 'Gemini 2.5 Flash',
    'gemini3flash':  'Gemini 3 Flash',
    'qwen':          'Qwen',
    'mistral':       'Mistral',
}

MODEL_COLORS = {
    'gpt5mini':      '#2E86AB',
    'gpt52':         '#3BB273',
    'gemini25flash': '#E87040',
    'gemini3flash':  '#9B5DE5',
    'qwen':          '#E84855',
    'mistral':       '#FF9F1C',
}

DATASET_ORDER  = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}

PP_WEIGHT = 0.6

def soft_f1(df):
    num = 2 * df['TP'] + PP_WEIGHT * df['PP']
    return num / (num + df['FP'] + df['FN'])


# ── 1. Load combined CSV ───────────────────────────────────────────────────────
print("Loading combined CSV...")
df = pd.read_csv(DESKTOP / 'all_non-aco_fcm_results_combined.csv')
df['soft_F1'] = soft_f1(df).replace([np.inf, -np.inf], np.nan)
print(f"  {len(df)} rows, {df['method'].nunique()} methods")


# ── 2. Organise results into tidy directory tree (from combined CSV) ───────────
print("\nOrganising result CSVs by model/reasoning...")
REASONING_METHODS = {m for m in df['method'].unique() if not m.endswith('_nr')}
for method in sorted(df['method'].unique()):
    if method.endswith('_nr'):
        model_key    = method[:-3]
        reason_label = 'no_reasoning'
    else:
        model_key    = method
        reason_label = 'reasoning'
    dest_dir = ORG_DIR / model_key / reason_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df['method'] == method].copy()
    out_csv = dest_dir / f'{method}_results.csv'
    sub.to_csv(out_csv, index=False)
    print(f"  {model_key}/{reason_label}: {len(sub)} rows -> {out_csv.name}")


# ── 3. Build paired dataset ───────────────────────────────────────────────────
print("\nBuilding paired dataset...")

all_pairs = []

for model in MODEL_PAIRS:
    r_key  = model
    nr_key = f'{model}_nr'

    r_df  = df[df['method'] == r_key ][['dataset','file_id','soft_F1','F1','TP','PP','FP','FN']].copy()
    nr_df = df[df['method'] == nr_key][['dataset','file_id','soft_F1','F1','TP','PP','FP','FN']].copy()

    r_df  = r_df.rename(columns={c: f'{c}_r'  for c in ['soft_F1','F1','TP','PP','FP','FN']})
    nr_df = nr_df.rename(columns={c: f'{c}_nr' for c in ['soft_F1','F1','TP','PP','FP','FN']})

    merged = pd.merge(r_df, nr_df, on=['dataset','file_id'], how='inner')
    merged['model'] = model
    merged['delta_soft_F1'] = merged['soft_F1_r'] - merged['soft_F1_nr']
    merged['delta_F1']      = merged['F1_r']      - merged['F1_nr']

    print(f"  {MODEL_DISPLAY[model]:20s}: {len(merged):3d} paired observations")
    all_pairs.append(merged)

paired = pd.concat(all_pairs, ignore_index=True)
print(f"\n  Total paired rows: {len(paired)}")

# Save full paired CSV
paired_out = OUT_DIR / 'paired_soft_f1_all.csv'
paired.to_csv(paired_out, index=False)
print(f"  Saved -> {paired_out}")


# ── 4. Summary statistics ─────────────────────────────────────────────────────
print("\n" + "="*72)
print("PAIRED DIFFERENCE SUMMARY  (explanation - no explanation, soft F1)")
print("="*72)

summary_rows = []

# Per model × dataset
for model in MODEL_PAIRS:
    for ds in DATASET_ORDER:
        sub   = paired[(paired['model']==model) & (paired['dataset']==ds)]['delta_soft_F1'].dropna()
        if len(sub) < 2:
            continue
        t, p  = stats.ttest_1samp(sub, 0)
        summary_rows.append({
            'model':         MODEL_DISPLAY[model],
            'dataset':       DATASET_LABELS[ds],
            'n':             len(sub),
            'mean_delta':    sub.mean(),
            'sd_delta':      sub.std(),
            'se_delta':      sub.sem(),
            't_stat':        t,
            'p_value':       p,
            'sig':           '*' if p < 0.05 else ('.' if p < 0.10 else ''),
        })

# Per model (all datasets)
print(f"\n{'Model':<22} {'Dataset':<16} {'n':>4}  {'Mean Δ':>8}  {'SD':>7}  {'p':>7}  sig")
print("-"*72)
for row in summary_rows:
    print(f"{row['model']:<22} {row['dataset']:<16} {row['n']:>4}  "
          f"{row['mean_delta']:>+8.4f}  {row['sd_delta']:>7.4f}  "
          f"{row['p_value']:>7.4f}  {row['sig']}")

# Per model (collapsed across datasets)
print(f"\n{'Model (all datasets)':<22} {'---':>8}  {'n':>4}  {'Mean Δ':>8}  {'SD':>7}  {'p':>7}  sig")
print("-"*72)
model_rows = []
for model in MODEL_PAIRS:
    sub  = paired[paired['model']==model]['delta_soft_F1'].dropna()
    t, p = stats.ttest_1samp(sub, 0)
    model_rows.append({
        'model':      MODEL_DISPLAY[model],
        'dataset':    'ALL',
        'n':          len(sub),
        'mean_delta': sub.mean(),
        'sd_delta':   sub.std(),
        'se_delta':   sub.sem(),
        't_stat':     t,
        'p_value':    p,
        'sig':        '*' if p < 0.05 else ('.' if p < 0.10 else ''),
    })
    print(f"{MODEL_DISPLAY[model]:<22}           {len(sub):>4}  "
          f"{sub.mean():>+8.4f}  {sub.std():>7.4f}  {p:>7.4f}  {'*' if p < 0.05 else ('.' if p < 0.10 else '')}")

# Grand average (all models + all datasets)
grand = paired['delta_soft_F1'].dropna()
t_g, p_g = stats.ttest_1samp(grand, 0)
print(f"\n{'GRAND AVERAGE':<22}           {len(grand):>4}  "
      f"{grand.mean():>+8.4f}  {grand.std():>7.4f}  {p_g:>7.4f}  "
      f"{'*' if p_g < 0.05 else ('.' if p_g < 0.10 else '')}")
print(f"  95% CI: [{grand.mean() - 1.96*grand.sem():.4f}, {grand.mean() + 1.96*grand.sem():.4f}]")

# Save summaries
all_summary = pd.DataFrame(summary_rows + model_rows + [{
    'model': 'GRAND AVERAGE', 'dataset': 'ALL',
    'n': len(grand), 'mean_delta': grand.mean(), 'sd_delta': grand.std(),
    'se_delta': grand.sem(), 't_stat': t_g, 'p_value': p_g,
    'sig': '*' if p_g < 0.05 else '',
}])
summary_out = OUT_DIR / 'paired_summary_table.csv'
all_summary.to_csv(summary_out, index=False)
print(f"\n  Saved -> {summary_out}")


# ── 5. Figure: mean delta soft-F1 per model, faceted by dataset ───────────────
print("\nGenerating figures...")

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.family': 'Arial', 'font.size': 10,
    'axes.linewidth': 1.0, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

# Figure A: boxplot of delta per model, one panel per dataset
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle('Soft F1: Explanation – No Explanation  (paired differences per participant)',
             fontsize=13, fontweight='bold')

for ax, ds in zip(axes, DATASET_ORDER):
    sub = paired[paired['dataset'] == ds]
    data_by_model = [sub[sub['model']==m]['delta_soft_F1'].dropna().values for m in MODEL_PAIRS]
    colors = [MODEL_COLORS[m] for m in MODEL_PAIRS]

    bp = ax.boxplot(data_by_model, patch_artist=True, showfliers=True,
                    showmeans=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5, linestyle='none'),
                    meanprops=dict(marker='D', markersize=5,
                                   markerfacecolor='black', markeredgecolor='black'),
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85)

    ax.axhline(0, color='black', linewidth=1.0, linestyle='-', alpha=0.5)
    ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight='bold')
    ax.set_xticks(range(1, len(MODEL_PAIRS)+1))
    ax.set_xticklabels([MODEL_DISPLAY[m].replace(' ', '\n') for m in MODEL_PAIRS],
                       fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('Δ Soft F1 (explanation − no explanation)' if ax == axes[0] else '')

handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_DISPLAY[m]) for m in MODEL_PAIRS]
fig.legend(handles=handles, loc='lower center', ncol=6, frameon=True,
           fontsize=9, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(OUT_DIR / 'figA_delta_soft_f1_per_dataset.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  [OK] figA_delta_soft_f1_per_dataset.png")

# Figure B: mean ± SE delta per model (all datasets combined), with zero line
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODEL_PAIRS))
means = [paired[paired['model']==m]['delta_soft_F1'].mean() for m in MODEL_PAIRS]
ses   = [paired[paired['model']==m]['delta_soft_F1'].sem()  for m in MODEL_PAIRS]
colors = [MODEL_COLORS[m] for m in MODEL_PAIRS]

bars = ax.bar(x, means, yerr=ses, capsize=5, color=colors, alpha=0.85,
              error_kw=dict(linewidth=1.5, ecolor='#333333'))
ax.axhline(0, color='black', linewidth=1.0, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_PAIRS], fontsize=10)
ax.set_ylabel('Mean Δ Soft F1 (explanation − no explanation)', fontsize=10)
ax.set_title('Average Soft F1 Gain from Explanation (all datasets combined)',
             fontsize=12, fontweight='bold')

# Annotate bars with mean value and significance
for i, (m, mu, se) in enumerate(zip(MODEL_PAIRS, means, ses)):
    sub = paired[paired['model']==m]['delta_soft_F1'].dropna()
    _, p = stats.ttest_1samp(sub, 0)
    sig = '**' if p < 0.01 else ('*' if p < 0.05 else ('†' if p < 0.10 else 'ns'))
    offset = se + 0.004
    ax.text(i, mu + offset if mu >= 0 else mu - offset,
            f'{mu:+.3f}\n{sig}', ha='center',
            va='bottom' if mu >= 0 else 'top', fontsize=9, fontweight='bold')

# Grand average line
ax.axhline(grand.mean(), color='grey', linewidth=1.5, linestyle='--', alpha=0.8,
           label=f'Grand mean Δ = {grand.mean():+.4f}')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / 'figB_mean_delta_by_model.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  [OK] figB_mean_delta_by_model.png")

# Figure C: heatmap of mean delta per model × dataset
fig, ax = plt.subplots(figsize=(10, 5))
heat_data = np.array([
    [paired[(paired['model']==m) & (paired['dataset']==ds)]['delta_soft_F1'].mean()
     for ds in DATASET_ORDER]
    for m in MODEL_PAIRS
])
im = ax.imshow(heat_data, cmap='RdYlGn', vmin=-0.25, vmax=0.25, aspect='auto')
plt.colorbar(im, ax=ax, label='Mean Δ Soft F1 (explanation − no explanation)')
ax.set_xticks(range(len(DATASET_ORDER)))
ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASET_ORDER], fontsize=11)
ax.set_yticks(range(len(MODEL_PAIRS)))
ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_PAIRS], fontsize=11)
ax.set_title('Mean Δ Soft F1 (explanation − no explanation) by Model × Dataset',
             fontsize=12, fontweight='bold', pad=10)
for i, m in enumerate(MODEL_PAIRS):
    for j, ds in enumerate(DATASET_ORDER):
        val = heat_data[i, j]
        sub = paired[(paired['model']==m) & (paired['dataset']==ds)]['delta_soft_F1'].dropna()
        _, p = stats.ttest_1samp(sub, 0) if len(sub) >= 2 else (np.nan, np.nan)
        sig = '*' if p < 0.05 else ''
        ax.text(j, i, f'{val:+.3f}{sig}', ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='black' if abs(val) < 0.15 else 'white')
fig.tight_layout()
fig.savefig(OUT_DIR / 'figC_heatmap_delta_model_x_dataset.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  [OK] figC_heatmap_delta_model_x_dataset.png")

print(f'\nAll outputs saved to: {OUT_DIR}')
