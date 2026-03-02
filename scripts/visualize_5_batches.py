"""
Boxplots comparing all 5 AI-generated FCM sets across all datasets.

  1. GPT-5-mini + explanation  (fcm_comparison_results_CORRECT)
  2. GPT-5-mini                (fcm_comparison_results_20260227)
  3. GPT-5.2 + explanation     (fcm_comparison_results_gpt52_20260227)
  4. Gemini 2.5 Flash          (fcm_comparison_results_..._gemini25flash_20260301)
  5. Gemini 3 Flash            (fcm_comparison_results_..._gemini3flash_20260301)

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

DESKTOP = Path(r'C:\Users\Nbrug\Desktop')

# ── Load data ──────────────────────────────────────────────────────────────────
sources = [
    (
        DESKTOP / 'fcm_comparison_results_CORRECT' /
        'all_fcm_comparisons_CORRECT.csv',
        'GPT-5-mini + explanation',
    ),
    (
        DESKTOP / 'fcm_comparison_results_20260227' /
        'all_fcm_comparisons_20260227.csv',
        'GPT-5-mini',
    ),
    (
        DESKTOP / 'fcm_comparison_results_gpt52_20260227' /
        'all_fcm_comparisons_fcm_adjacency_matrices_fcm_interviews_gpt52_20260227_174543.csv',
        'GPT-5.2 + explanation',
    ),
    (
        DESKTOP / 'fcm_comparison_results_fcm_adjacency_matrices_fcm_interviews_gemini25flash_20260301_182916' /
        'all_fcm_comparisons_fcm_adjacency_matrices_fcm_interviews_gemini25flash_20260301_182916.csv',
        'Gemini 2.5 Flash',
    ),
    (
        DESKTOP / 'fcm_comparison_results_fcm_adjacency_matrices_fcm_interviews_gemini3flash_20260301_182933' /
        'all_fcm_comparisons_fcm_adjacency_matrices_fcm_interviews_gemini3flash_20260301_182933.csv',
        'Gemini 3 Flash',
    ),
]

frames = []
for path, label in sources:
    df_tmp = pd.read_csv(path)
    df_tmp['batch'] = label
    frames.append(df_tmp)
    print(f'  {label}: {len(df_tmp)} rows')

df = pd.concat(frames, ignore_index=True)
print(f'\nTotal rows: {len(df)}')

# ── Constants ──────────────────────────────────────────────────────────────────
DATASETS = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}

BATCHES = [label for _, label in sources]

BATCH_COLORS = {
    'GPT-5-mini + explanation': '#2E86AB',   # blue
    'GPT-5-mini':               '#E84855',   # red
    'GPT-5.2 + explanation':    '#3BB273',   # green
    'Gemini 2.5 Flash':         '#F4A261',   # orange
    'Gemini 3 Flash':           '#8B5CF6',   # purple
}

# ── Boxplot helper ─────────────────────────────────────────────────────────────
def make_boxplot_grid(fig, axes_flat, metrics, metric_labels, df, title):
    """Draw side-by-side boxplots (one group per dataset) for each metric."""
    n_datasets = len(DATASETS)
    n_batches  = len(BATCHES)
    width      = 0.14
    offsets    = np.linspace(-(n_batches - 1) * width / 2,
                              (n_batches - 1) * width / 2,
                              n_batches)
    x_positions = np.arange(n_datasets)

    for ax, metric, ylabel in zip(axes_flat, metrics, metric_labels):
        for i, (batch, color) in enumerate(BATCH_COLORS.items()):
            pos = x_positions + offsets[i]
            data_per_dataset = [
                df[(df['dataset'] == ds) & (df['batch'] == batch)][metric].dropna().values
                for ds in DATASETS
            ]
            ax.boxplot(
                data_per_dataset,
                positions=pos,
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

    patches = [mpatches.Patch(color=BATCH_COLORS[b], alpha=0.75, label=b)
               for b in BATCHES]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))


# ── Figure 1: Performance metrics ─────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
make_boxplot_grid(
    fig1, axes1.flat,
    metrics       = ['F1',       'Jaccard'],
    metric_labels = ['F1 Score', 'Jaccard Similarity'],
    df            = df,
    title         = 'FCM Scoring Performance by Dataset and Method (All 5 AI Sets)',
)
fig1.tight_layout()

# ── Figure 2: Edge matching metrics ───────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
make_boxplot_grid(
    fig2, axes2.flat,
    metrics       = ['TP', 'PP', 'FP', 'FN'],
    metric_labels = ['True Positives (TP)', 'Partial Positives (PP)',
                     'False Positives (FP)', 'False Negatives (FN)'],
    df            = df,
    title         = 'Edge Matching Metrics by Dataset and Method (All 5 AI Sets)',
)
fig2.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = DESKTOP / 'fcm_visualizations_all5'
out_dir.mkdir(exist_ok=True)

for fig, fname in [(fig1, 'performance_metrics_all5.png'),
                   (fig2, 'edge_matching_all5.png')]:
    path = out_dir / fname
    fig.savefig(path, dpi=300, bbox_inches='tight')
    kb = path.stat().st_size / 1024
    print(f'[OK] {path}  ({kb:.1f} KB)')

# ── Summary table ──────────────────────────────────────────────────────────────
print()
print('=' * 75)
print('MEAN F1 BY DATASET AND METHOD')
print('=' * 75)
summary = (df.groupby(['dataset', 'batch'])['F1']
             .agg(['mean', 'std', 'count'])
             .rename(columns={'mean': 'F1_mean', 'std': 'F1_std', 'count': 'n'})
             .round(3))
print(summary.to_string())

print()
print('=' * 75)
print('MEAN JACCARD BY DATASET AND METHOD')
print('=' * 75)
summary_j = (df.groupby(['dataset', 'batch'])['Jaccard']
               .agg(['mean', 'std'])
               .rename(columns={'mean': 'Jaccard_mean', 'std': 'Jaccard_std'})
               .round(3))
print(summary_j.to_string())
