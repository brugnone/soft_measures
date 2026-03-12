"""
graph_metrics_plots.py
Produces two publication-quality figures of FCM graph-structure metrics,
using the same colour palette as visualize_all_methods.py.

  figF_graph_density_degree.png  — edge density (total / +ve / -ve) + max degree × 4 datasets
  figG_graph_node_edge_counts.png — node count + edge count × 4 datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
METRICS_CSV = Path(r'C:\Users\Nbrug\Desktop\fcm_paired_analysis\fcm_graph_metrics_all.csv')
OUT_DIR = Path(r'C:\Users\Nbrug\Desktop\fcm_visualizations_all_methods')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style constants ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':  'sans-serif',
    'font.size':    8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

METHOD_COLORS = {
    'gpt5mini':         '#2E86AB',
    'gpt5mini_nr':      '#7BBFD4',
    'gpt52':            '#3BB273',
    'gpt52_nr':         '#93D4B0',
    'gemini25flash':    '#E87040',
    'gemini25flash_nr': '#F4A27A',
    'gemini3flash':     '#9B5DE5',
    'gemini3flash_nr':  '#C89EEF',
    'qwen':             '#E84855',
    'qwen_nr':          '#F4929A',
    'mistral':          '#FF9F1C',
    'mistral_nr':       '#FFCC7A',
}

GT_COLOR   = '#555555'
GT_ALPHA   = 0.15       # band fill alpha
FLIER_SIZE = 2.0        # outlier marker size
BOX_ALPHA  = 0.85

MODEL_PAIRS = [
    ('gpt5mini',      'gpt5mini_nr',      'GPT-5-mini'),
    ('gpt52',         'gpt52_nr',         'GPT-5.2'),
    ('gemini25flash', 'gemini25flash_nr', 'Gemini 2.5\nFlash'),
    ('gemini3flash',  'gemini3flash_nr',  'Gemini 3\nFlash'),
    ('qwen',          'qwen_nr',          'Qwen'),
    ('mistral',       'mistral_nr',       'Mistral'),
]

DATASET_ORDER = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(METRICS_CSV)
df['dataset'] = pd.Categorical(df['dataset'], categories=DATASET_ORDER, ordered=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def box_positions(n_pairs=6, within_gap=0.30, between_gap=1.10):
    """Return (r_positions, nr_positions, pair_centres) arrays."""
    r_pos, nr_pos, centres = [], [], []
    x = 0.0
    for _ in range(n_pairs):
        r_pos.append(x)
        nr_pos.append(x + within_gap)
        centres.append(x + within_gap / 2)
        x += within_gap + between_gap
    return np.array(r_pos), np.array(nr_pos), np.array(centres)


R_POS, NR_POS, CENTRES = box_positions()


def draw_panel(ax, dataset, metric, ylabel=None, title=None, show_xlabel=True):
    """Draw one subplot: box plots for all model pairs + GT band."""
    sub = df[df['dataset'] == dataset]
    gt_vals = sub.loc[sub['method'] == 'gt', metric].dropna().values

    # GT reference: dashed line at mean + shaded ±1 SD band
    if len(gt_vals):
        gt_mean = gt_vals.mean()
        gt_sd   = gt_vals.std()
        ax.axhline(gt_mean, color=GT_COLOR, linewidth=1.2, linestyle='--', zorder=1,
                   label='GT mean')
        ax.axhspan(gt_mean - gt_sd, gt_mean + gt_sd,
                   color=GT_COLOR, alpha=GT_ALPHA, zorder=0, label='GT ±1 SD')

    # Box plots per model pair
    all_data_r, all_data_nr = [], []
    for (r_key, nr_key, _) in MODEL_PAIRS:
        r_vals  = sub.loc[sub['method'] == r_key,  metric].dropna().values
        nr_vals = sub.loc[sub['method'] == nr_key, metric].dropna().values
        all_data_r.append(r_vals  if len(r_vals)  else [np.nan])
        all_data_nr.append(nr_vals if len(nr_vals) else [np.nan])

    def _draw_boxes(positions, data_list, keys):
        bps = ax.boxplot(
            data_list,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            notch=False,
            showfliers=True,
            flierprops=dict(marker='o', markersize=FLIER_SIZE,
                            markerfacecolor='none', markeredgewidth=0.5,
                            alpha=0.5),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
            medianprops=dict(color='white', linewidth=1.5),
            boxprops=dict(linewidth=0.5),
            zorder=2,
        )
        for i, (patch, key) in enumerate(zip(bps['boxes'], keys)):
            patch.set_facecolor(METHOD_COLORS[key])
            patch.set_alpha(BOX_ALPHA)
        return bps

    _draw_boxes(R_POS,  all_data_r,  [p[0] for p in MODEL_PAIRS])
    _draw_boxes(NR_POS, all_data_nr, [p[1] for p in MODEL_PAIRS])

    # Pair label ticks
    ax.set_xticks(CENTRES)
    if show_xlabel:
        ax.set_xticklabels([p[2] for p in MODEL_PAIRS], fontsize=6.8, rotation=25,
                           ha='right', rotation_mode='anchor')
    else:
        ax.set_xticklabels([] * len(MODEL_PAIRS))

    xleft  = R_POS[0]  - 0.35
    xright = NR_POS[-1] + 0.35
    ax.set_xlim(xleft, xright)

    if title:
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


# ── Fig F — density & degree (4 metrics × 4 datasets) ─────────────────────────
METRICS_F = [
    ('edge_density',     'Edge density',          True),
    ('pos_edge_density', 'Positive edge density', False),
    ('neg_edge_density', 'Negative edge density', False),
    ('max_total_degree', 'Max total degree',      False),
]

fig_f, axes_f = plt.subplots(
    nrows=len(METRICS_F), ncols=len(DATASET_ORDER),
    figsize=(14, 10),
    constrained_layout=True,
)

for row_i, (metric, ylabel_base, _) in enumerate(METRICS_F):
    for col_j, ds in enumerate(DATASET_ORDER):
        ax = axes_f[row_i, col_j]
        show_x = (row_i == len(METRICS_F) - 1)
        title  = DATASET_LABELS[ds] if row_i == 0 else None
        ylabel = ylabel_base         if col_j == 0 else None
        draw_panel(ax, ds, metric, ylabel=ylabel, title=title, show_xlabel=show_x)

# Shared legend
r_patches  = [mpatches.Patch(color=METHOD_COLORS[p[0]], label=p[2] + ' (E)', alpha=BOX_ALPHA)
               for p in MODEL_PAIRS]
nr_patches = [mpatches.Patch(color=METHOD_COLORS[p[1]], label=p[2] + ' (NE)', alpha=BOX_ALPHA)
               for p in MODEL_PAIRS]
gt_line    = mlines.Line2D([], [], color=GT_COLOR, linewidth=1.2, linestyle='--', label='GT mean')
gt_band    = mpatches.Patch(color=GT_COLOR, alpha=GT_ALPHA, label='GT ±1 SD')

# Two-column interleaved (R left, NR right)
handles = []
for r, nr in zip(r_patches, nr_patches):
    handles += [r, nr]
handles += [gt_line, gt_band]

fig_f.legend(handles=handles, loc='lower center',
             bbox_to_anchor=(0.5, -0.04),
             ncol=7, fontsize=7, frameon=False,
             handlelength=1.4, handleheight=0.9, borderpad=0.3, columnspacing=0.8)

fig_f.suptitle('FCM Graph-Structure Metrics: Edge Density & Degree\n'
               '(Dashed line = GT mean, shaded = GT ±1 SD)',
               fontsize=10, fontweight='bold', y=1.01)

out_f = OUT_DIR / 'figF_graph_density_degree.png'
fig_f.savefig(out_f, dpi=200, bbox_inches='tight')
plt.close(fig_f)
print(f'Saved → {out_f}')


# ── Fig G — node & edge counts (2 metrics × 4 datasets) ───────────────────────
METRICS_G = [
    ('n_nodes', 'Node count',  True),
    ('n_edges', 'Edge count', False),
]

fig_g, axes_g = plt.subplots(
    nrows=len(METRICS_G), ncols=len(DATASET_ORDER),
    figsize=(14, 5.5),
    constrained_layout=True,
)

for row_i, (metric, ylabel_base, _) in enumerate(METRICS_G):
    for col_j, ds in enumerate(DATASET_ORDER):
        ax = axes_g[row_i, col_j]
        show_x = (row_i == len(METRICS_G) - 1)
        title  = DATASET_LABELS[ds] if row_i == 0 else None
        ylabel = ylabel_base         if col_j == 0 else None
        draw_panel(ax, ds, metric, ylabel=ylabel, title=title, show_xlabel=show_x)

fig_g.legend(handles=handles, loc='lower center',
             bbox_to_anchor=(0.5, -0.07),
             ncol=7, fontsize=7, frameon=False,
             handlelength=1.4, handleheight=0.9, borderpad=0.3, columnspacing=0.8)

fig_g.suptitle('FCM Graph-Structure Metrics: Node & Edge Counts\n'
               '(Dashed line = GT mean, shaded = GT ±1 SD)',
               fontsize=10, fontweight='bold', y=1.02)

out_g = OUT_DIR / 'figG_graph_node_edge_counts.png'
fig_g.savefig(out_g, dpi=200, bbox_inches='tight')
plt.close(fig_g)
print(f'Saved → {out_g}')


# ── Combined-dataset helpers ───────────────────────────────────────────────────
# Add a synthetic 'all' label so draw_panel can pool all four datasets
df_all = df.copy()
df_all['dataset'] = 'all'
df_combined = pd.concat([df, df_all], ignore_index=True)
df_combined['dataset'] = pd.Categorical(
    df_combined['dataset'],
    categories=DATASET_ORDER + ['all'],
    ordered=True,
)

# Swap the global df so draw_panel reads from the combined frame
_orig_df = df
df = df_combined   # noqa: F811 — intentional global swap for reuse of draw_panel


def draw_panel_wide(ax, metric, ylabel=None, title=None, show_xlabel=True):
    """Wrapper around draw_panel for the 'all' dataset row."""
    return draw_panel(ax, 'all', metric,
                      ylabel=ylabel, title=title, show_xlabel=show_xlabel)


# ── Fig F2 — density & degree, all datasets combined ──────────────────────────
fig_f2, axes_f2 = plt.subplots(
    nrows=len(METRICS_F), ncols=1,
    figsize=(5.5, 10),
    constrained_layout=True,
)

for row_i, (metric, ylabel_base, _) in enumerate(METRICS_F):
    ax = axes_f2[row_i]
    show_x = (row_i == len(METRICS_F) - 1)
    draw_panel_wide(ax, metric,
                    ylabel=ylabel_base,
                    title=None,
                    show_xlabel=show_x)

fig_f2.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.05),
              ncol=4, fontsize=7, frameon=False,
              handlelength=1.4, handleheight=0.9, borderpad=0.3, columnspacing=0.8)

fig_f2.suptitle('FCM Graph-Structure Metrics: Edge Density & Degree\n'
                'All datasets combined  ·  (Dashed = GT mean, shaded = GT ±1 SD)',
                fontsize=10, fontweight='bold', y=1.01)

out_f2 = OUT_DIR / 'figF2_graph_density_degree_combined.png'
fig_f2.savefig(out_f2, dpi=200, bbox_inches='tight')
plt.close(fig_f2)
print(f'Saved → {out_f2}')


# ── Fig G2 — node & edge counts, all datasets combined ────────────────────────
fig_g2, axes_g2 = plt.subplots(
    nrows=len(METRICS_G), ncols=1,
    figsize=(5.5, 5.5),
    constrained_layout=True,
)

for row_i, (metric, ylabel_base, _) in enumerate(METRICS_G):
    ax = axes_g2[row_i]
    show_x = (row_i == len(METRICS_G) - 1)
    draw_panel_wide(ax, metric,
                    ylabel=ylabel_base,
                    title=None,
                    show_xlabel=show_x)

fig_g2.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.09),
              ncol=4, fontsize=7, frameon=False,
              handlelength=1.4, handleheight=0.9, borderpad=0.3, columnspacing=0.8)

fig_g2.suptitle('FCM Graph-Structure Metrics: Node & Edge Counts\n'
                'All datasets combined  ·  (Dashed = GT mean, shaded = GT ±1 SD)',
                fontsize=10, fontweight='bold', y=1.02)

out_g2 = OUT_DIR / 'figG2_graph_node_edge_counts_combined.png'
fig_g2.savefig(out_g2, dpi=200, bbox_inches='tight')
plt.close(fig_g2)
print(f'Saved → {out_g2}')

# Restore original df
df = _orig_df  # noqa: F811

print('\nDone.')
