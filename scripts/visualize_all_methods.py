"""
Publication-ready figures for all 8 AI methods vs ground truth.

Figure 1 – F1 by method, faceted by dataset (grouped boxplot)
Figure 2 – TP, PP, FP, FN by method, faceted by dataset (2×2 boxplot grid)
Figure 3 – GT vs AI nodes scatter, one panel per method (3×3 grid)
Figure 4 – GT vs AI edges scatter, one panel per method (3×3 grid)

Output: C:\\Users\\Nbrug\\Desktop\\fcm_visualizations_all_methods\\
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
from scipy import stats

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

DESKTOP  = Path(r'C:\Users\Nbrug\Desktop')
OUT_DIR  = DESKTOP / 'fcm_visualizations_all_methods'
OUT_DIR.mkdir(exist_ok=True)
DATA_CSV = DESKTOP / 'all_fcm_results_combined.csv'

# ── Method display names & colours ───────────────────────────────────────────
METHOD_ORDER = [
    'gpt5mini',
    'gpt5mini_nr',
    'gpt52',
    'gpt52_nr',
    'gemini25flash',
    'gemini25flash_nr',
    'gemini3flash',
    'gemini3flash_nr',
    'qwen',
    'qwen_nr',
    'mistral',
    'mistral_nr',
]

METHOD_LABELS = {
    'gpt5mini':         'GPT-5-mini\n(reasoning)',
    'gpt5mini_nr':      'GPT-5-mini\n(no reasoning)',
    'gpt52':            'GPT-5.2\n(reasoning)',
    'gpt52_nr':         'GPT-5.2\n(no reasoning)',
    'gemini25flash':    'Gemini 2.5 Flash\n(reasoning)',
    'gemini25flash_nr': 'Gemini 2.5 Flash\n(no reasoning)',
    'gemini3flash':     'Gemini 3 Flash\n(reasoning)',
    'gemini3flash_nr':  'Gemini 3 Flash\n(no reasoning)',
    'qwen':             'Qwen\n(reasoning)',
    'qwen_nr':          'Qwen\n(no reasoning)',
    'mistral':          'Mistral\n(reasoning)',
    'mistral_nr':       'Mistral\n(no reasoning)',
}

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

DATASET_ORDER  = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}
DATASET_COLORS = {
    'biodiversity': '#264653',
    'flpp':         '#2A9D8F',
    'osw':          '#E9C46A',
    'red_snapper':  '#E76F51',
}
DATASET_MARKERS = {
    'biodiversity': 'o',
    'flpp':         's',
    'osw':          '^',
    'red_snapper':  'D',
}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
df['method']  = pd.Categorical(df['method'],  categories=METHOD_ORDER,  ordered=True)
df['dataset'] = pd.Categorical(df['dataset'], categories=DATASET_ORDER, ordered=True)
print(f"Loaded {len(df)} rows  |  methods: {sorted(df['method'].unique())}")

# ── Derived soft metrics ─────────────────────────────────────────────────────
PP_WEIGHT = 0.6
df['soft_num']       = 2 * df['TP'] + PP_WEIGHT * df['PP']   # numerator shared by new_F1 and soft_F1
df['new_F1']         = df['soft_num'] / (2 * df['TP'] + df['PP'] + df['FP'] + df['FN'])
df['soft_F1']        = df['soft_num'] / (df['soft_num'] + df['FP'] + df['FN'])
df['soft_precision'] = (df['TP'] + PP_WEIGHT * df['PP']) / (df['TP'] + df['PP'] + df['FP'])
df['soft_recall']    = (df['TP'] + PP_WEIGHT * df['PP']) / (df['TP'] + df['PP'] + df['FN'])
# Guard against divide-by-zero
for col in ['new_F1', 'soft_F1', 'soft_precision', 'soft_recall']:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

SOFT_METRICS = [
    ('F1',             'F1 (original)'),
    ('new_F1',         'New F1\n(2TP+0.6PP)/(2TP+PP+FP+FN)'),
    ('soft_F1',        'Soft F1\n(2TP+0.6PP)/(2TP+0.6PP+FP+FN)'),
    ('soft_precision', 'Soft Precision\n(TP+0.6PP)/(TP+PP+FP)'),
    ('soft_recall',    'Soft Recall\n(TP+0.6PP)/(TP+PP+FN)'),
]

n_methods = len(METHOD_ORDER)


def r_with_se(a, b):
    """Return (r, SE) using Fisher z SE = (1-r^2)/sqrt(n-3).  Returns (nan, nan) if n<4."""
    mask = pd.notna(a) & pd.notna(b)
    n = mask.sum()
    if n < 4:
        return float('nan'), float('nan')
    r_val = stats.pearsonr(a[mask], b[mask])[0]
    se    = (1 - r_val ** 2) / np.sqrt(n - 3)
    return r_val, se


# =============================================================================
# FIGURE 1 – F1 by method, faceted by dataset
# =============================================================================
def fig1_f1_by_method():
    fig, axes = plt.subplots(2, 2, figsize=(22, 11), sharey=False)
    fig.suptitle('F1 Score by AI Method and Dataset', fontsize=14,
                 fontweight='bold', y=1.01)

    x_pos    = np.arange(n_methods)
    width    = 0.65

    for ax, ds in zip(axes.flat, DATASET_ORDER):
        sub = df[df['dataset'] == ds]

        vals  = [sub[sub['method'] == m]['F1'].dropna().values for m in METHOD_ORDER]
        bplot = ax.boxplot(
            vals,
            positions=x_pos,
            widths=width,
            patch_artist=True,
            showfliers=True,
            showmeans=True,
            flierprops=dict(marker='o', markersize=3, alpha=0.5, linestyle='none'),
            meanprops=dict(marker='D', markersize=5,
                           markerfacecolor='black', markeredgecolor='black'),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, m in zip(bplot['boxes'], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.85)

        ax.set_title(DATASET_LABELS[ds], fontsize=12, fontweight='bold', pad=6)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER],
                           fontsize=7, rotation=30, ha='right')
        ax.axhline(0.5, color='grey', linewidth=0.8, linestyle=':', alpha=0.6)

        # mean annotation above each box
        for i, m in enumerate(METHOD_ORDER):
            mean_val = sub[sub['method'] == m]['F1'].mean()
            if np.isfinite(mean_val):
                ax.text(i, mean_val + 0.04, f'{mean_val:.2f}',
                        ha='center', va='bottom', fontsize=6.5, color='#333333')

    # Legend
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m].replace('\n', ' '))
               for m in METHOD_ORDER]
    fig.legend(handles=handles, title='AI Method', title_fontsize=9,
               loc='lower center', ncol=6, frameon=True, fontsize=8,
               bbox_to_anchor=(0.5, -0.08), borderpad=0.8)

    fig.tight_layout()
    out = OUT_DIR / 'fig1_f1_by_method_per_dataset.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[OK] {out}')
    plt.close(fig)


# =============================================================================
# FIGURE 2 – TP, PP, FP, FN  (2×2 grid; grouped by method within each panel)
# =============================================================================
def fig2_tpppfpfn():
    metrics = ['TP', 'PP', 'FP', 'FN']
    metric_labels = {
        'TP': 'True Positives (TP)',
        'PP': 'Partial Positives (PP)',
        'FP': 'False Positives (FP)',
        'FN': 'False Negatives (FN)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Edge Classification Metrics by AI Method and Dataset',
                 fontsize=14, fontweight='bold', y=1.01)

    n_datasets = len(DATASET_ORDER)
    group_gap  = 1.5            # gap between dataset groups
    bar_width  = 0.18

    for ax, metric in zip(axes.flat, metrics):
        for g_idx, ds in enumerate(DATASET_ORDER):
            sub = df[df['dataset'] == ds]
            group_center = g_idx * (n_methods * bar_width + group_gap)

            for m_idx, method in enumerate(METHOD_ORDER):
                vals = sub[sub['method'] == method][metric].dropna().values
                x    = group_center + m_idx * bar_width

                if len(vals) == 0:
                    continue

                bplot = ax.boxplot(
                    vals,
                    positions=[x],
                    widths=bar_width * 0.85,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=False,
                    medianprops=dict(color='white', linewidth=1.5),
                    whiskerprops=dict(linewidth=0.9),
                    capprops=dict(linewidth=0.9),
                )
                for patch in bplot['boxes']:
                    patch.set_facecolor(METHOD_COLORS[method])
                    patch.set_alpha(0.85)

        # x-axis ticks at group centres
        group_centers = [g * (n_methods * bar_width + group_gap) +
                         (n_methods - 1) * bar_width / 2
                         for g in range(n_datasets)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASET_ORDER],
                           fontsize=10, fontweight='bold')
        ax.set_ylabel('Edge Count', fontsize=10)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold', pad=6)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    # Legend
    handles = [mpatches.Patch(color=METHOD_COLORS[m],
                               label=METHOD_LABELS[m].replace('\n', ' '))
               for m in METHOD_ORDER]
    fig.legend(handles=handles, title='AI Method', title_fontsize=9,
               loc='lower center', ncol=6, frameon=True, fontsize=8,
               bbox_to_anchor=(0.5, -0.06), borderpad=0.8)

    fig.tight_layout()
    out = OUT_DIR / 'fig2_tp_pp_fp_fn_by_method.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[OK] {out}')
    plt.close(fig)


# =============================================================================
# HELPER – one scatter panel per method
# =============================================================================
def _scatter_panels(x_col, y_col, x_label, y_label, fig_title, out_name):
    """
    3×3 grid (last cell = empty / legend).  One panel per method;
    points coloured by dataset, marker shape by dataset.
    Includes identity line and per-dataset Pearson r annotation.
    """
    ncols, nrows = 6, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(26, 9))
    fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.02)

    ax_list = axes.flat

    for idx, method in enumerate(METHOD_ORDER):
        ax  = ax_list[idx]
        sub = df[df['method'] == method]

        all_vals = pd.concat([sub[x_col], sub[y_col]]).dropna()
        ax_max   = all_vals.max() * 1.08 if len(all_vals) else 1

        # Identity line
        ax.plot([0, ax_max], [0, ax_max],
                'k--', linewidth=1.0, alpha=0.4, zorder=1)

        r_lines = []
        for ds in DATASET_ORDER:
            s = sub[sub['dataset'] == ds].dropna(subset=[x_col, y_col])
            if s.empty:
                continue
            ax.scatter(
                s[x_col], s[y_col],
                color=DATASET_COLORS[ds],
                marker=DATASET_MARKERS[ds],
                s=45, alpha=0.75, edgecolors='none', zorder=3,
            )
            if len(s) >= 2:
                r, p = stats.pearsonr(s[x_col], s[y_col])
                r_lines.append((ds, r))

        ax.set_xlim(0, ax_max)
        ax.set_ylim(0, ax_max)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(METHOD_LABELS[method].replace('\n', ' '), fontsize=9,
                     fontweight='bold', pad=5)
        ax.set_aspect('equal', adjustable='box')

        # Overall correlation across all datasets
        overall = sub.dropna(subset=[x_col, y_col])
        if len(overall) >= 2:
            r_all, _ = stats.pearsonr(overall[x_col], overall[y_col])
            ax.text(
                ax_max * 0.97, ax_max * 0.03,
                f'Overall r={r_all:.2f}',
                color='black', fontsize=8, va='bottom', ha='right',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#cccccc', alpha=0.85),
                zorder=6,
            )

        # Per-dataset correlation annotations
        if r_lines:
            y_pos  = ax_max * 0.98
            line_h = ax_max * 0.075
            for i, (ds, r) in enumerate(r_lines):
                ax.text(
                    ax_max * 0.03, y_pos - i * line_h,
                    f'{DATASET_LABELS[ds]}: r={r:.2f}',
                    color=DATASET_COLORS[ds], fontsize=7, va='top',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.55),
                    zorder=5,
                )

    # Legend panel (last axes cell if n_methods < nrows*ncols)
    n_cells = nrows * ncols
    for i in range(len(METHOD_ORDER), n_cells):
        ax_list[i].axis('off')

    # Shared legend in bottom-right empty cell (or below figure)
    ds_handles = [
        mlines.Line2D([], [], color=DATASET_COLORS[ds],
                      marker=DATASET_MARKERS[ds], linestyle='None',
                      markersize=7, label=DATASET_LABELS[ds])
        for ds in DATASET_ORDER
    ]
    fig.legend(handles=ds_handles, title='Dataset', title_fontsize=9,
               loc='lower center', ncol=4, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), borderpad=0.8)

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[OK] {out}')
    plt.close(fig)


# =============================================================================
# FIGURE 5 – Summary table: mean ± SD per method across all datasets
# =============================================================================
def fig5_soft_metrics_table():
    cols  = [m for m, _ in SOFT_METRICS]
    short = ['F1', 'New F1', 'Soft F1', 'Soft Prec.', 'Soft Rec.']

    # ── CSV summary ──────────────────────────────────────────────────────────
    rows = []
    for method in METHOD_ORDER:
        sub = df[df['method'] == method]
        row = {'method': METHOD_LABELS[method].replace('\n', ' ')}
        for col, lbl in zip(cols, short):
            vals = sub[col].dropna()
            row[f'{lbl} mean'] = round(vals.mean(), 3)
            row[f'{lbl} SD']   = round(vals.std(),  3)
        # Node correlations
        for ds in DATASET_ORDER:
            s = sub[sub['dataset'] == ds].dropna(subset=['gt_nodes', 'ai_nodes'])
            rv, se = r_with_se(s['gt_nodes'], s['ai_nodes'])
            row[f'r_nodes_{ds}']    = round(rv, 3)
            row[f'r_nodes_{ds}_SE'] = round(se, 3)
        overall_n = sub.dropna(subset=['gt_nodes', 'ai_nodes'])
        rv, se = r_with_se(overall_n['gt_nodes'], overall_n['ai_nodes'])
        row['r_nodes_all']    = round(rv, 3)
        row['r_nodes_all_SE'] = round(se, 3)
        # Edge correlations
        for ds in DATASET_ORDER:
            s = sub[sub['dataset'] == ds].dropna(subset=['gt_edges', 'ai_edges'])
            rv, se = r_with_se(s['gt_edges'], s['ai_edges'])
            row[f'r_edges_{ds}']    = round(rv, 3)
            row[f'r_edges_{ds}_SE'] = round(se, 3)
        overall_e = sub.dropna(subset=['gt_edges', 'ai_edges'])
        rv, se = r_with_se(overall_e['gt_edges'], overall_e['ai_edges'])
        row['r_edges_all']    = round(rv, 3)
        row['r_edges_all_SE'] = round(se, 3)
        rows.append(row)
    tbl = pd.DataFrame(rows)
    csv_out = OUT_DIR / 'table_soft_metrics_by_method.csv'
    tbl.to_csv(csv_out, index=False)
    print(f'[OK] {csv_out}')

    # ── Matplotlib table figure ───────────────────────────────────────────────
    # Build display strings  "mean ± SD"
    cell_text = []
    for method in METHOD_ORDER:
        sub = df[df['method'] == method]
        row_cells = [METHOD_LABELS[method].replace('\n', ' ')]
        for col in cols:
            vals = sub[col].dropna()
            row_cells.append(f"{vals.mean():.3f} ± {vals.std():.3f}")
        cell_text.append(row_cells)

    col_headers = ['Method'] + short
    row_colors  = [[METHOD_COLORS[m]] + ['#f7f7f7'] * len(cols) for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis('off')
    tbl_obj = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
    )
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(9)
    tbl_obj.scale(1, 1.6)

    # Style header row
    for j in range(len(col_headers)):
        tbl_obj[0, j].set_facecolor('#333333')
        tbl_obj[0, j].set_text_props(color='white', fontweight='bold')

    # Make method-name cells readable (dark text, lighter alpha)
    for i, method in enumerate(METHOD_ORDER, start=1):
        cell = tbl_obj[i, 0]
        cell.set_alpha(0.75)
        cell.set_text_props(color='white', fontweight='bold', fontsize=8)

    fig.suptitle('Soft Metrics Summary by AI Method (mean ± SD, all datasets)',
                 fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout()
    out = OUT_DIR / 'fig5_soft_metrics_table.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[OK] {out}')
    plt.close(fig)


# =============================================================================
# FIGURE 7 – Correlation table: r(nodes) and r(edges) per dataset + overall
# =============================================================================
def fig7_correlation_table():
    ds_short = [DATASET_LABELS[ds] for ds in DATASET_ORDER]
    col_headers = ['Method',
                   'r(nodes)\nBiodiv.', 'r(nodes)\nFLPP', 'r(nodes)\nGulf OSW', 'r(nodes)\nRed Snapper', 'r(nodes)\nAll',
                   'r(edges)\nBiodiv.', 'r(edges)\nFLPP', 'r(edges)\nGulf OSW', 'r(edges)\nRed Snapper', 'r(edges)\nAll']

    cell_text  = []
    cell_colors = []

    def r_color(r):
        """White-to-green colormap for r in [-1, 1]."""
        if np.isnan(r):
            return '#eeeeee'
        t = (r + 1) / 2          # map [-1,1] → [0,1]
        g = 0.45 + 0.45 * t      # green channel
        r_ch = 1.0 - 0.55 * t   # red channel  (fades out as r increases)
        b_ch = 1.0 - 0.55 * t   # blue channel
        return (r_ch, g, b_ch)

    for method in METHOD_ORDER:
        sub  = df[df['method'] == method]
        row_cells  = [METHOD_LABELS[method].replace('\n', ' ')]
        row_col    = [METHOD_COLORS[method]]

        # nodes per dataset
        for ds in DATASET_ORDER:
            s = sub[sub['dataset'] == ds].dropna(subset=['gt_nodes', 'ai_nodes'])
            rv, se = r_with_se(s['gt_nodes'], s['ai_nodes'])
            lbl = f'{rv:.3f}\n\u00b1{se:.3f}' if not np.isnan(rv) else 'n/a'
            row_cells.append(lbl)
            row_col.append(r_color(rv))
        # nodes overall
        sn = sub.dropna(subset=['gt_nodes', 'ai_nodes'])
        rv, se = r_with_se(sn['gt_nodes'], sn['ai_nodes'])
        row_cells.append(f'{rv:.3f}\n\u00b1{se:.3f}' if not np.isnan(rv) else 'n/a')
        row_col.append(r_color(rv))

        # edges per dataset
        for ds in DATASET_ORDER:
            s = sub[sub['dataset'] == ds].dropna(subset=['gt_edges', 'ai_edges'])
            rv, se = r_with_se(s['gt_edges'], s['ai_edges'])
            lbl = f'{rv:.3f}\n\u00b1{se:.3f}' if not np.isnan(rv) else 'n/a'
            row_cells.append(lbl)
            row_col.append(r_color(rv))
        # edges overall
        se_sub = sub.dropna(subset=['gt_edges', 'ai_edges'])
        rv, se = r_with_se(se_sub['gt_edges'], se_sub['ai_edges'])
        row_cells.append(f'{rv:.3f}\n\u00b1{se:.3f}' if not np.isnan(rv) else 'n/a')
        row_col.append(r_color(rv))

        cell_text.append(row_cells)
        cell_colors.append(row_col)

    n_cols = len(col_headers)
    fig, ax = plt.subplots(figsize=(22, 6.0))
    ax.axis('off')
    tbl_obj = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
    )
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(8.0)
    tbl_obj.scale(1, 2.4)

    # Header styling
    for j in range(n_cols):
        tbl_obj[0, j].set_facecolor('#333333')
        tbl_obj[0, j].set_text_props(color='white', fontweight='bold')

    # Method-name column styling
    for i, method in enumerate(METHOD_ORDER, start=1):
        cell = tbl_obj[i, 0]
        cell.set_alpha(0.75)
        cell.set_text_props(color='white', fontweight='bold', fontsize=7.5)

    # Divider: shade the "All" columns slightly differently
    for i in range(1, len(METHOD_ORDER) + 1):
        for j in [5, 10]:   # "All" columns (0-indexed)
            c = tbl_obj[i, j]
            cur = c.get_facecolor()
            # darken slightly
            c.set_facecolor((max(0, cur[0]-0.08), max(0, cur[1]-0.04), max(0, cur[2]-0.08)))

    fig.suptitle('Pearson r: GT vs AI Node & Edge Counts  (per dataset and overall)',
                 fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout()
    out = OUT_DIR / 'fig7_correlation_table.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[OK] {out}')
    plt.close(fig)


# =============================================================================
# FIGURE 6 – Boxplots: F1 / new_F1 / soft_precision / soft_recall  (2×2 grid)
# =============================================================================
def fig6_soft_metrics_boxplot():
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), sharey=False)
    fig.suptitle('Soft Metrics by AI Method (all datasets combined)',
                 fontsize=14, fontweight='bold', y=1.01)

    x_pos = np.arange(n_methods)
    width = 0.65

    for ax, (col, label) in zip(axes.flat, SOFT_METRICS):
        vals  = [df[df['method'] == m][col].dropna().values for m in METHOD_ORDER]
        bplot = ax.boxplot(
            vals,
            positions=x_pos,
            widths=width,
            patch_artist=True,
            showfliers=True,
            showmeans=True,
            flierprops=dict(marker='o', markersize=3, alpha=0.45, linestyle='none'),
            meanprops=dict(marker='D', markersize=5,
                           markerfacecolor='black', markeredgecolor='black'),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, m in zip(bplot['boxes'], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.85)

        ax.set_title(label.replace('\n', '\n'), fontsize=10, fontweight='bold', pad=6)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER],
                           fontsize=7, rotation=30, ha='right')
        ax.axhline(0.5, color='grey', linewidth=0.8, linestyle=':', alpha=0.6)

        # Mean annotation above each box
        for i, m in enumerate(METHOD_ORDER):
            mean_val = df[df['method'] == m][col].mean()
            if np.isfinite(mean_val):
                ax.text(i, mean_val + 0.04, f'{mean_val:.2f}',
                        ha='center', va='bottom', fontsize=6.5, color='#333333')

    # Hide unused panels (2×3 grid has 6 cells, we use 5)
    for ax in axes.flat[len(SOFT_METRICS):]:
        ax.axis('off')

    handles = [mpatches.Patch(color=METHOD_COLORS[m],
                               label=METHOD_LABELS[m].replace('\n', ' '))
               for m in METHOD_ORDER]
    fig.legend(handles=handles, title='AI Method', title_fontsize=9,
               loc='lower center', ncol=6, frameon=True, fontsize=8,
               bbox_to_anchor=(0.5, -0.08), borderpad=0.8)

    fig.tight_layout()
    out = OUT_DIR / 'fig6_soft_metrics_boxplot.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'[OK] {out}')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print(f'\nOutput dir: {OUT_DIR}\n')

    fig1_f1_by_method()
    fig2_tpppfpfn()

    _scatter_panels(
        x_col='gt_nodes', y_col='ai_nodes',
        x_label='Ground Truth Nodes',
        y_label='AI Nodes',
        fig_title='Node Count: Ground Truth vs AI-Generated (per method)',
        out_name='fig3_gt_vs_ai_nodes.png',
    )

    _scatter_panels(
        x_col='gt_edges', y_col='ai_edges',
        x_label='Ground Truth Edges',
        y_label='AI Edges',
        fig_title='Edge Count: Ground Truth vs AI-Generated (per method)',
        out_name='fig4_gt_vs_ai_edges.png',
    )

    fig5_soft_metrics_table()
    fig6_soft_metrics_boxplot()
    fig7_correlation_table()

    print('\nAll figures saved to:', OUT_DIR)
