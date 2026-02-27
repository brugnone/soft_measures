"""
Boxplots comparing both AI batches across all datasets.
  - Batch 1: fcm_ai_20260225  (labelled "AI v1")
  - Batch 2: fcm_ai_20260227  (labelled "AI v2")

Produces two figures:
  1. Performance metrics  – F1, Jaccard
  2. Edge matching metrics – TP, PP, FP, FN
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Load & combine data ────────────────────────────────────────────────────────
df1 = pd.read_csv(r'C:\Users\Nbrug\Desktop\fcm_comparison_results_CORRECT\all_fcm_comparisons_CORRECT.csv')
df2 = pd.read_csv(r'C:\Users\Nbrug\Desktop\fcm_comparison_results_20260227\all_fcm_comparisons_20260227.csv')

df1['batch'] = 'AI v1 (Feb 25)'
df2['batch'] = 'AI v2 (Feb 27)'

df = pd.concat([df1, df2], ignore_index=True)

print(f"Loaded  AI v1: {len(df1)} rows  |  AI v2: {len(df2)} rows  |  Total: {len(df)} rows")

# ── Helpers ────────────────────────────────────────────────────────────────────
DATASETS      = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}

BATCH_COLORS = {
    'AI v1 (Feb 25)': '#2E86AB',   # blue
    'AI v2 (Feb 27)': '#E84855',   # red
}
BATCHES = ['AI v1 (Feb 25)', 'AI v2 (Feb 27)']

def make_boxplot_grid(fig, axes_flat, metrics, metric_labels, df, title):
    """Draw side-by-side boxplots (one pair per dataset) for each metric."""
    n_datasets = len(DATASETS)
    x_positions = np.arange(n_datasets)
    width = 0.35

    for ax, metric, ylabel in zip(axes_flat, metrics, metric_labels):
        for i, (batch, color) in enumerate(BATCH_COLORS.items()):
            offsets = x_positions + (i - 0.5) * width
            data_per_dataset = [
                df[(df['dataset'] == ds) & (df['batch'] == batch)][metric].dropna().values
                for ds in DATASETS
            ]
            bp = ax.boxplot(
                data_per_dataset,
                positions=offsets,
                widths=width * 0.85,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=3, alpha=0.5,
                                markerfacecolor=color, markeredgewidth=0.5),
                medianprops=dict(color='black', linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.75),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(ylabel, fontsize=11, fontweight='bold', pad=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    # Shared legend
    patches = [mpatches.Patch(color=c, alpha=0.75, label=b)
               for b, c in BATCH_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=2,
               frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))


# ── Figure 1: Performance metrics ─────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
make_boxplot_grid(
    fig1, axes1.flat,
    metrics       = ['F1',    'Jaccard'],
    metric_labels = ['F1 Score', 'Jaccard Similarity'],
    df            = df,
    title         = 'FCM Scoring Performance by Dataset and AI Batch',
)
fig1.tight_layout()

# ── Figure 2: Edge matching metrics ───────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
make_boxplot_grid(
    fig2, axes2.flat,
    metrics       = ['TP', 'PP', 'FP', 'FN'],
    metric_labels = ['True Positives (TP)', 'Partial Positives (PP)',
                     'False Positives (FP)', 'False Negatives (FN)'],
    df            = df,
    title         = 'Edge Matching Metrics by Dataset and AI Batch',
)
fig2.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = Path(r'C:\Users\Nbrug\Desktop\fcm_comparison_results_20260227\visualizations')
out_dir.mkdir(exist_ok=True)

for fig, fname in [(fig1, 'performance_metrics_both_batches.png'),
                   (fig2, 'edge_matching_both_batches.png')]:
    path = out_dir / fname
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = path.stat().st_size / 1024
    print(f'[OK] {path.name}  ({kb:.1f} KB)')

# ── Summary table ──────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('MEAN F1 BY DATASET AND BATCH')
print('=' * 70)
summary = (df.groupby(['dataset', 'batch'])['F1']
             .agg(['mean', 'std', 'count'])
             .rename(columns={'mean':'F1_mean','std':'F1_std','count':'n'})
             .round(3))
print(summary.to_string())

print()
print('=' * 70)
print('MEAN JACCARD BY DATASET AND BATCH')
print('=' * 70)
summary_j = (df.groupby(['dataset', 'batch'])['Jaccard']
               .agg(['mean', 'std'])
               .rename(columns={'mean':'Jaccard_mean','std':'Jaccard_std'})
               .round(3))
print(summary_j.to_string())
