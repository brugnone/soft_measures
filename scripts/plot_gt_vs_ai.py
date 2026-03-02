"""
Scatter plots comparing Ground Truth vs AI-generated FCM node and edge counts,
for all 5 AI-generated FCM sets.

  - Color = AI method
  - Alpha = F1 score (more opaque = higher F1)
  - Marker shape = dataset
  - Diagonal dashed line = perfect agreement (GT == AI)

Output: Desktop/fcm_visualizations_all5/gt_vs_ai_scatter_all5.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DESKTOP = Path(r'C:\Users\Nbrug\Desktop')

# ── Data sources ───────────────────────────────────────────────────────────────
SOURCES = [
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

BATCH_COLORS = {
    'GPT-5-mini + explanation': '#2E86AB',
    'GPT-5-mini':               '#E84855',
    'GPT-5.2 + explanation':    '#3BB273',
    'Gemini 2.5 Flash':         '#F4A261',   # orange
    'Gemini 3 Flash':           '#E040FB',   # magenta
}

DATASET_MARKERS = {
    'biodiversity': 'o',
    'flpp':         's',
    'osw':          '^',
    'red_snapper':  'D',
}
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}

# ── Load ───────────────────────────────────────────────────────────────────────
frames = []
for path, label in SOURCES:
    tmp = pd.read_csv(path)
    tmp['batch'] = label
    frames.append(tmp)
    print(f'  {label}: {len(tmp)} rows')
df = pd.concat(frames, ignore_index=True)
print(f'\nTotal: {len(df)} rows\n')


def hex_to_rgb(hex_color):
    """Convert #RRGGBB to (r, g, b) in [0, 1]."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def scatter_with_f1_alpha(ax, x, y, f1, color_hex, marker, size=60):
    """
    Draw a scatter where each point's alpha is its F1 score.
    Uses per-point RGBA colour array (matplotlib scatter supports this).
    """
    r, g, b = hex_to_rgb(color_hex)
    # Clamp F1 to [0.15, 1.0] so even low-F1 points remain faintly visible
    alpha_vals = np.clip(np.asarray(f1, dtype=float), 0.15, 1.0)
    colors_rgba = np.column_stack([
        np.full(len(alpha_vals), r),
        np.full(len(alpha_vals), g),
        np.full(len(alpha_vals), b),
        alpha_vals,
    ])
    ax.scatter(x, y, c=colors_rgba, marker=marker, s=size,
               edgecolors='none', zorder=3)


def compute_r2(x, y):
    """Pearson r^2 (squared correlation coefficient) between arrays x and y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float('nan')
    r = np.corrcoef(x, y)[0, 1]
    return r ** 2


def draw_panel(ax, x_col, y_col, x_label, y_label, title):
    """Populate one scatter panel (nodes or edges)."""
    for _, batch_label in SOURCES:
        color = BATCH_COLORS[batch_label]
        batch_df = df[df['batch'] == batch_label]
        for dataset, marker in DATASET_MARKERS.items():
            sub = batch_df[batch_df['dataset'] == dataset]
            if sub.empty:
                continue
            scatter_with_f1_alpha(
                ax,
                sub[x_col].values,
                sub[y_col].values,
                sub['F1'].values,
                color,
                marker,
            )

    # Diagonal reference
    combined_max = max(df[x_col].max(), df[y_col].max()) * 1.05
    ax.plot([0, combined_max], [0, combined_max],
            'k--', linewidth=1.2, alpha=0.45, zorder=1)

    ax.set_xlim(0, combined_max)
    ax.set_ylim(0, combined_max)
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.25, linestyle='--')

    # ── Per-method R² annotation (all datasets combined) ──
    lines = []
    r2_vals = []
    for _, batch_label in SOURCES:
        batch_df = df[df['batch'] == batch_label]
        r2 = compute_r2(batch_df[x_col], batch_df[y_col])
        r2_vals.append((batch_label, r2))

    # Place text box in upper-left, stacking one line per method
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_pos = x_lim[0] + 0.03 * (x_lim[1] - x_lim[0])
    y_top = y_lim[0] + 0.97 * (y_lim[1] - y_lim[0])
    line_h = 0.07 * (y_lim[1] - y_lim[0])

    for i, (batch_label, r2) in enumerate(r2_vals):
        color = BATCH_COLORS[batch_label]
        short = batch_label.replace(' + explanation', '+exp')
        ax.text(
            x_pos, y_top - i * line_h,
            f'{short}: R²={r2:.3f}',
            color=color, fontsize=7.5, fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6),
            zorder=5,
        )


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

draw_panel(ax1,
           x_col='gt_nodes', y_col='ai_nodes',
           x_label='Ground Truth Node Count',
           y_label='AI Node Count',
           title='Node Count: GT vs AI')

draw_panel(ax2,
           x_col='gt_edges', y_col='ai_edges',
           x_label='Ground Truth Edge Count',
           y_label='AI Edge Count',
           title='Edge Count: GT vs AI')

# ── Legends ────────────────────────────────────────────────────────────────────
method_patches = [
    mpatches.Patch(color=BATCH_COLORS[b], label=b)
    for _, b in SOURCES
]
dataset_handles = [
    mlines.Line2D([], [], color='#555555', marker=m, linestyle='None',
                  markersize=7, label=DATASET_LABELS[d])
    for d, m in DATASET_MARKERS.items()
]

fig.tight_layout()
fig.subplots_adjust(bottom=0.32, wspace=0.25)

# Left compartment: AI Method
leg1 = fig.legend(
    handles=method_patches,
    title='AI Method',
    title_fontsize=9,
    loc='lower left',
    ncol=1,
    frameon=True,
    fontsize=8.5,
    bbox_to_anchor=(0.08, 0.02),
    borderpad=0.7,
)
leg1.get_title().set_fontweight('bold')

# Right compartment: Dataset
leg2 = fig.legend(
    handles=dataset_handles,
    title='Dataset',
    title_fontsize=9,
    loc='lower right',
    ncol=1,
    frameon=True,
    fontsize=8.5,
    bbox_to_anchor=(0.92, 0.02),
    borderpad=0.7,
)
leg2.get_title().set_fontweight('bold')
fig.add_artist(leg1)   # keep leg1 visible after leg2 is added

# Opacity note centred between the two legends
fig.text(0.5, 0.09, 'Opacity ~ F1 score (more opaque = higher F1)',
         ha='center', va='bottom', fontsize=8.5, style='italic', color='#444444')

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = DESKTOP / 'fcm_visualizations_all5'
out_dir.mkdir(exist_ok=True)
out_path = out_dir / 'gt_vs_ai_scatter_all5.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
kb = out_path.stat().st_size / 1024
print(f'[OK] {out_path}  ({kb:.1f} KB)')

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('MEAN NODE / EDGE COUNTS  (GT vs AI)  BY METHOD')
print('=' * 70)
summary = (df.groupby('batch')[['gt_nodes', 'ai_nodes', 'gt_edges', 'ai_edges']]
             .mean().round(1))
print(summary.to_string())

print()
print('DONE!')
