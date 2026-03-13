"""
combined_graph_stats_table.py
Builds a unified graph-statistics + correlation summary table for ALL methods
(non-ACO and ACO), using:
  • fcm_graph_metrics_all.csv  (per-FCM graph metrics, non-ACO)
  • aco_graph_metrics_all.csv  (per-FCM graph metrics, ACO)
  • all_non-aco_fcm_results_combined.csv  (paired GT/AI counts for r)
  • all_aco_results_final.csv             (paired GT/AI counts for r)

Outputs (to fcm_visualizations_all_methods/):
  • unified_graph_stats_summary.csv      – per method × dataset, incl. correlations
  • fig_combined_graph_stats.png         – matplotlib table (pooled)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib   import Path
from scipy     import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
DESKTOP  = Path(r'C:\Users\Nbrug\Desktop')
NONACO_CSV      = DESKTOP / 'fcm_paired_analysis' / 'fcm_graph_metrics_all.csv'
ACO_CSV         = DESKTOP / 'fcm_visualizations_aco' / 'aco_graph_metrics_all.csv'
NONACO_PAIRS    = DESKTOP / 'all_non-aco_fcm_results_combined.csv'
ACO_PAIRS       = DESKTOP / 'all_aco_results_final.csv'
OUT_DIR         = DESKTOP / 'fcm_visualizations_all_methods'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Method labels & display order ─────────────────────────────────────────────
# Each entry: (method_key, source, display_label, hex_color)
METHOD_DEFS = [
    # ── GT
    ('gt',               'non-ACO', 'GT',                        '#555555'),
    # ── Non-ACO (with explanation)
    ('gpt5mini',         'non-ACO', 'GPT-5-mini (E)',            '#1f77b4'),
    ('gpt52',            'non-ACO', 'GPT-5.2 (E)',               '#2ca02c'),
    ('gemini25flash',    'non-ACO', 'Gemini 2.5 Flash (E)',      '#ff7f0e'),
    ('gemini3flash',     'non-ACO', 'Gemini 3 Flash (E)',        '#d62728'),
    ('qwen',             'non-ACO', 'Qwen (E)',                  '#9467bd'),
    ('mistral',          'non-ACO', 'Mistral (E)',               '#8c564b'),
    # ── Non-ACO (no explanation)
    ('gpt5mini_nr',      'non-ACO', 'GPT-5-mini (NE)',           '#aec7e8'),
    ('gpt52_nr',         'non-ACO', 'GPT-5.2 (NE)',              '#98df8a'),
    ('gemini25flash_nr', 'non-ACO', 'Gemini 2.5 Flash (NE)',     '#ffbb78'),
    ('gemini3flash_nr',  'non-ACO', 'Gemini 3 Flash (NE)',       '#ff9896'),
    ('qwen_nr',          'non-ACO', 'Qwen (NE)',                 '#c5b0d5'),
    ('mistral_nr',       'non-ACO', 'Mistral (NE)',              '#c49c94'),
    # ── ACO models
    ('aco-qwen',         'ACO',     'ACO-Qwen',                  '#17becf'),
    ('aco-mistral',      'ACO',     'ACO-Mistral',               '#bcbd22'),
    ('gpt-5-mini',       'ACO',     'GPT-5-mini [ACO]',         '#9edae5'),
    ('gpt-5.2',          'ACO',     'GPT-5.2 [ACO]',            '#dbdb8d'),
    ('gemini-2.5-flash', 'ACO',     'Gemini 2.5 Flash [ACO]',   '#f7b6d2'),
    ('gemini-3-flash',   'ACO',     'Gemini 3 Flash [ACO]',     '#c7c7c7'),
]

METHOD_ORDER  = [m[0] for m in METHOD_DEFS]
METHOD_LABEL  = {m[0]: m[2] for m in METHOD_DEFS}
METHOD_SOURCE = {m[0]: m[1] for m in METHOD_DEFS}
METHOD_COLOR  = {m[0]: m[3] for m in METHOD_DEFS}

# ── Load & harmonise non-ACO data ─────────────────────────────────────────────
df_na = pd.read_csv(NONACO_CSV)
# Normalise dataset names to common keys
DS_MAP_NONACO = {'biodiversity': 'biodiversity', 'flpp': 'flpp',
                 'osw': 'osw', 'red_snapper': 'red_snapper'}
df_na['dataset_key'] = df_na['dataset'].map(DS_MAP_NONACO).fillna(df_na['dataset'])
df_na['method_key']  = df_na['method']
df_na.loc[df_na['method_key'] == 'gt', 'method_key'] = 'gt'   # keep as 'gt' for non-ACO

# ── Load & harmonise ACO data ──────────────────────────────────────────────────
df_aco = pd.read_csv(ACO_CSV)
DS_MAP_ACO = {'biodiversity': 'biodiversity', 'flpp': 'flpp',
              'gulf-osw': 'osw', 'red-snapper': 'red_snapper'}
df_aco['dataset_key'] = df_aco['dataset'].map(DS_MAP_ACO).fillna(df_aco['dataset'])
# Drop GT from ACO source — it is identical to the non-ACO GT
df_aco['method_key'] = df_aco['method']
df_aco = df_aco[df_aco['method'] != 'gt'].copy()

# ── Shared metrics ─────────────────────────────────────────────────────────────
METRICS = ['n_nodes', 'n_edges', 'edge_density', 'neg_edge_density']
METRIC_LABELS = {
    'n_nodes':          'Nodes',
    'n_edges':          'Edges',
    'edge_density':     'Edge Density',
    'neg_edge_density': 'Neg. Edge Density',
}

DATASET_ORDER  = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}

# ── Build correlation lookup from paired GT/AI data ────────────────────────────
# Returns (r, n) for Pearson r, or (nan, 0) if insufficient data
def _pearson(a, b):
    mask = pd.notna(a) & pd.notna(b)
    n = int(mask.sum())
    if n < 3:
        return np.nan, n
    r, _ = stats.pearsonr(a[mask], b[mask])
    return round(float(r), 4), n

# Non-ACO paired data: method, dataset, gt_nodes, ai_nodes, gt_edges, ai_edges
pairs_na = pd.read_csv(NONACO_PAIRS)
DS_CANON_NA = {'biodiversity': 'biodiversity', 'flpp': 'flpp',
               'osw': 'osw', 'red_snapper': 'red_snapper'}
pairs_na['dataset_key'] = pairs_na['dataset'].map(DS_CANON_NA).fillna(pairs_na['dataset'])

# ACO paired data: model_name (== method_key), dataset_name, fcm1_nodes/edges, fcm2_nodes/edges
pairs_aco = pd.read_csv(ACO_PAIRS)
DS_CANON_ACO = {'biodiversity': 'biodiversity', 'flpp': 'flpp',
                'gulf-osw': 'osw', 'red-snapper': 'red_snapper'}
pairs_aco['dataset_key'] = pairs_aco['dataset_name'].map(DS_CANON_ACO).fillna(pairs_aco['dataset_name'])
pairs_aco['method_key']  = pairs_aco['model_name']

# Build lookup: (method_key, dataset_key) -> {r_nodes, r_edges, n_pairs}
corr_lookup = {}   # key -> {'r_nodes': float, 'r_edges': float, 'n_pairs': int}
f1_lookup   = {}   # key -> {'f1_mean': float, 'f1_sd': float}

for method_key, grp in pairs_na.groupby('method'):
    for ds_key, sg in grp.groupby('dataset_key'):
        r_nd, n_nd = _pearson(sg['gt_nodes'], sg['ai_nodes'])
        r_ed, _    = _pearson(sg['gt_edges'], sg['ai_edges'])
        corr_lookup[(method_key, ds_key)] = {'r_nodes': r_nd, 'r_edges': r_ed, 'n_pairs': n_nd}
        f1 = sg['F1'].dropna()
        f1_lookup[(method_key, ds_key)] = {'f1_mean': round(f1.mean(), 4), 'f1_sd': round(f1.std(), 4)}
    # pooled 'all' for this method
    r_nd, n_nd = _pearson(grp['gt_nodes'], grp['ai_nodes'])
    r_ed, _    = _pearson(grp['gt_edges'], grp['ai_edges'])
    corr_lookup[(method_key, 'all')] = {'r_nodes': r_nd, 'r_edges': r_ed, 'n_pairs': n_nd}
    f1 = grp['F1'].dropna()
    f1_lookup[(method_key, 'all')] = {'f1_mean': round(f1.mean(), 4), 'f1_sd': round(f1.std(), 4)}

for method_key, grp in pairs_aco.groupby('method_key'):
    for ds_key, sg in grp.groupby('dataset_key'):
        r_nd, n_nd = _pearson(sg['fcm1_nodes'], sg['fcm2_nodes'])
        r_ed, _    = _pearson(sg['fcm1_edges'], sg['fcm2_edges'])
        corr_lookup[(method_key, ds_key)] = {'r_nodes': r_nd, 'r_edges': r_ed, 'n_pairs': n_nd}
        f1 = sg['F1'].dropna()
        f1_lookup[(method_key, ds_key)] = {'f1_mean': round(f1.mean(), 4), 'f1_sd': round(f1.std(), 4)}
    r_nd, n_nd = _pearson(grp['fcm1_nodes'], grp['fcm2_nodes'])
    r_ed, _    = _pearson(grp['fcm1_edges'], grp['fcm2_edges'])
    corr_lookup[(method_key, 'all')] = {'r_nodes': r_nd, 'r_edges': r_ed, 'n_pairs': n_nd}
    f1 = grp['F1'].dropna()
    f1_lookup[(method_key, 'all')] = {'f1_mean': round(f1.mean(), 4), 'f1_sd': round(f1.std(), 4)}

# Map METHOD_DEF keys (non-prefixed) to ACO CSV model_name keys (always aco- prefixed)
ACO_KEY_ALIAS = {
    'gpt-5-mini':       'aco-gpt-5-mini',
    'gpt-5.2':          'aco-gpt-5.2',
    'gemini-2.5-flash': 'aco-gemini-2.5-flash',
    'gemini-3-flash':   'aco-gemini-3-flash',
}

# ── Combine both dataframes ────────────────────────────────────────────────────
shared = ['method_key', 'dataset_key'] + METRICS
df_all = pd.concat([
    df_na[shared].copy(),
    df_aco[shared].copy()
], ignore_index=True)
df_all = df_all[df_all['method_key'].isin(METHOD_ORDER)]

# ── Build per-method × per-dataset summary CSV ────────────────────────────────
rows = []
for mk in METHOD_ORDER:
    sub = df_all[df_all['method_key'] == mk]
    if sub.empty:
        continue
    for ds in DATASET_ORDER + ['all']:
        if ds == 'all':
            s = sub
            ds_label = 'All'
        else:
            s = sub[sub['dataset_key'] == ds]
            ds_label = DATASET_LABELS.get(ds, ds)
        if s.empty:
            continue
        row = {
            'method':  mk,
            'label':   METHOD_LABEL[mk],
            'source':  METHOD_SOURCE[mk],
            'dataset': ds_label,
            'n':       len(s),
        }
        for m in METRICS:
            v = s[m].dropna()
            row[f'{m}_mean'] = round(v.mean(), 4) if len(v) else np.nan
            row[f'{m}_sd']   = round(v.std(),  4) if len(v) > 1 else np.nan
        # Correlations (GT is never a source, so will be nan for 'gt')
        ds_lk  = 'all' if ds == 'all' else ds
        lk_key = ACO_KEY_ALIAS.get(mk, mk)   # resolve aco- alias if needed
        c = corr_lookup.get((lk_key, ds_lk), {})
        row['r_nodes']  = c.get('r_nodes', np.nan)
        row['r_edges']  = c.get('r_edges', np.nan)
        row['n_pairs']  = c.get('n_pairs', 0)
        # Soft F1
        f = f1_lookup.get((lk_key, ds_lk), {})
        row['f1_mean']  = f.get('f1_mean', np.nan)
        row['f1_sd']    = f.get('f1_sd',   np.nan)
        rows.append(row)

summary_df = pd.DataFrame(rows)
# canonical column order
unified_cols = [
    'method', 'label', 'source', 'dataset', 'n',
    'f1_mean', 'f1_sd',
    'n_nodes_mean', 'n_nodes_sd',
    'n_edges_mean', 'n_edges_sd',
    'r_nodes', 'r_edges',
    'edge_density_mean', 'edge_density_sd',
    'neg_edge_density_mean', 'neg_edge_density_sd',
    'n_pairs',
]
summary_df = summary_df[[c for c in unified_cols if c in summary_df.columns]]
unified_csv_out = OUT_DIR / 'unified_graph_stats_summary.csv'
summary_df.to_csv(unified_csv_out, index=False)
print(f'[OK] {unified_csv_out}')
# keep legacy filename too
csv_out = OUT_DIR / 'combined_graph_stats_summary.csv'
summary_df.to_csv(csv_out, index=False)
print(f'[OK] {csv_out}')

# ── Build the pooled-overview figure table ────────────────────────────────────
def fmt(val_mean, val_sd, scale=1.0, pct=False):
    """Format mean ± SD, optionally as percentage."""
    if pd.isna(val_mean):
        return 'n/a'
    m = val_mean * scale
    s = val_sd   * scale if not pd.isna(val_sd) else 0.0
    sfx = '%' if pct else ''
    return f'{m:.1f}{sfx}\n±{s:.1f}{sfx}'

def fmt_f1(val_mean, val_sd):
    """Format F1 mean ± SD with 2 decimal places."""
    if pd.isna(val_mean):
        return 'n/a'
    s = val_sd if not pd.isna(val_sd) else 0.0
    return f'{val_mean:.2f}\n±{s:.2f}'

col_headers = ['Method', 'Source',
               'Soft F1\nmean ± SD',
               'Nodes\nmean ± SD',
               'Edges\nmean ± SD',
               'Node\nCorr (r)',
               'Edge\nCorr (r)',
               'Edge Density\nmean ± SD (%)',
               'Neg. Density\nmean ± SD (%)']

cell_text   = []
cell_colors = []
# track raw numerics for ranking: list of (row_idx_1based, value) per highlight column
# col indices: F1=2, r_nodes=5, r_edges=6
rank_vals = {2: [], 5: [], 6: []}

for mk in METHOD_ORDER:
    sub = df_all[df_all['method_key'] == mk]
    if sub.empty:
        continue
    row_1idx = len(cell_text) + 1   # 1-based table row (header is 0)
    c_all  = corr_lookup.get((ACO_KEY_ALIAS.get(mk, mk), 'all'), {})
    f_all  = f1_lookup.get((ACO_KEY_ALIAS.get(mk, mk), 'all'), {})
    r_nodes_val = c_all.get('r_nodes', np.nan)
    r_edges_val = c_all.get('r_edges', np.nan)
    f1_val      = f_all.get('f1_mean', np.nan)
    r_nd_str = f"{r_nodes_val:.3f}" if not pd.isna(r_nodes_val) else '—'
    r_ed_str = f"{r_edges_val:.3f}" if not pd.isna(r_edges_val) else '—'
    row_cells = [
        METHOD_LABEL[mk],
        METHOD_SOURCE[mk],
        fmt_f1(f1_val, f_all.get('f1_sd', np.nan)),
        fmt(sub['n_nodes'].mean(),          sub['n_nodes'].std()),
        fmt(sub['n_edges'].mean(),          sub['n_edges'].std()),
        r_nd_str,
        r_ed_str,
        fmt(sub['edge_density'].mean(),     sub['edge_density'].std(),     scale=100, pct=True),
        fmt(sub['neg_edge_density'].mean(), sub['neg_edge_density'].std(), scale=100, pct=True),
    ]
    if not pd.isna(f1_val):
        rank_vals[2].append((row_1idx, f1_val))
    if not pd.isna(r_nodes_val):
        rank_vals[5].append((row_1idx, r_nodes_val))
    if not pd.isna(r_edges_val):
        rank_vals[6].append((row_1idx, r_edges_val))
    bg = METHOD_COLOR[mk]
    cell_text.append(row_cells)
    cell_colors.append([bg, '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5'])

n_rows = len(cell_text)
n_cols = len(col_headers)

fig, ax = plt.subplots(figsize=(26, 0.42 * n_rows + 1.2))
ax.axis('off')

tbl = ax.table(
    cellText=cell_text,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
    cellColours=cell_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 2.0)

# Header row styling
for j in range(n_cols):
    tbl[0, j].set_facecolor('#333333')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Method-name column: use the cell's bg color with white bold text
for i in range(1, n_rows + 1):
    tbl[i, 0].set_text_props(color='white', fontweight='bold', fontsize=8)
    tbl[i, 0].set_alpha(0.85)
    # Source column
    tbl[i, 1].set_facecolor('#f0f0f0')
    src = cell_text[i - 1][1]
    tbl[i, 1].set_text_props(
        color='#222222',
        fontweight='bold',
        fontstyle='italic',
        fontsize=8,
    )

# ── Bold top-1 / italicise top-2 for F1, r_nodes, r_edges ────────────────────
for col_j, entries in rank_vals.items():
    sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
    if len(sorted_entries) >= 1:
        r1, _ = sorted_entries[0]
        tbl[r1, col_j].set_text_props(fontweight='bold', color='#000000')
    if len(sorted_entries) >= 2:
        r2, _ = sorted_entries[1]
        tbl[r2, col_j].set_text_props(fontstyle='italic', color='#000000')

# Section divider lines between groups — thicker bottom edge on the last row of each group
GROUP_BREAKS = [1, 7, 13]   # after GT (1), after last E method (7), after last NE method (13)
for brk in GROUP_BREAKS:
    if brk < n_rows:
        for j in range(n_cols):
            tbl[brk, j].set_edgecolor('#555555')
            tbl[brk, j].set_linewidth(2.0)

fig.suptitle('Graph Structure Statistics — All Methods (pooled across datasets)',
             fontsize=12, fontweight='bold', y=0.99)
fig.tight_layout()

out_fig = OUT_DIR / 'fig_combined_graph_stats.png'
fig.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'[OK] {out_fig}')

# ── GT per-dataset summary table ──────────────────────────────────────────────
gt = df_na[df_na['method_key'] == 'gt'].copy()

DS_ORDER_FULL = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DS_LABELS_FULL = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'osw':          'Gulf OSW',
    'red_snapper':  'Red Snapper',
}

gt_rows = []
for ds in DS_ORDER_FULL + ['all']:
    s    = gt if ds == 'all' else gt[gt['dataset_key'] == ds]
    dlbl = 'All' if ds == 'all' else DS_LABELS_FULL[ds]
    def ms(col, scale=1, pct=False):
        v = s[col].dropna()
        sfx = '%' if pct else ''
        return f'{v.mean()*scale:.2f}{sfx}', f'{v.std()*scale:.2f}{sfx}'
    nn_m, nn_s   = ms('n_nodes')
    ne_m, ne_s   = ms('n_edges')
    ped_m, ped_s = ms('pos_edge_density', scale=100, pct=True)
    ned_m, ned_s = ms('neg_edge_density', scale=100, pct=True)
    gt_rows.append({
        'Dataset':             dlbl,
        'N FCMs':              len(s),
        'Nodes (mean)':        nn_m,
        'Nodes (SD)':          nn_s,
        'Edges (mean)':        ne_m,
        'Edges (SD)':          ne_s,
        'Pos. Edge Density (mean)': ped_m,
        'Pos. Edge Density (SD)':   ped_s,
        'Neg. Edge Density (mean)': ned_m,
        'Neg. Edge Density (SD)':   ned_s,
    })

gt_summary_df = pd.DataFrame(gt_rows)
gt_csv_out = OUT_DIR / 'gt_graph_stats_per_dataset.csv'
gt_summary_df.to_csv(gt_csv_out, index=False)
print(f'[OK] {gt_csv_out}')

# ── Figure: GT per-dataset table ──────────────────────────────────────────────
col_headers_gt = [
    'Dataset', 'N FCMs',
    'Nodes\nmean ± SD',
    'Edges\nmean ± SD',
    'Pos. Edge Density\nmean ± SD (%)',
    'Neg. Edge Density\nmean ± SD (%)',
]

cell_text_gt = []
for r in gt_rows:
    cell_text_gt.append([
        r['Dataset'],
        str(r['N FCMs']),
        f"{r['Nodes (mean)']} ± {r['Nodes (SD)']}",
        f"{r['Edges (mean)']} ± {r['Edges (SD)']}",
        f"{r['Pos. Edge Density (mean)']} ± {r['Pos. Edge Density (SD)']}",
        f"{r['Neg. Edge Density (mean)']} ± {r['Neg. Edge Density (SD)']}",
    ])

n_gt_rows = len(cell_text_gt)
n_gt_cols = len(col_headers_gt)

ROW_COLORS = ['#f0f4f8', '#dce8f5']   # alternating light blues
cell_colors_gt = []
for i, row in enumerate(cell_text_gt):
    if row[0] == 'All':
        cell_colors_gt.append(['#d0dce8'] * n_gt_cols)
    else:
        cell_colors_gt.append([ROW_COLORS[i % 2]] * n_gt_cols)

fig_gt, ax_gt = plt.subplots(figsize=(13, 0.55 * n_gt_rows + 1.2))
ax_gt.axis('off')

tbl_gt = ax_gt.table(
    cellText=cell_text_gt,
    colLabels=col_headers_gt,
    cellLoc='center',
    loc='center',
    cellColours=cell_colors_gt,
)
tbl_gt.auto_set_font_size(False)
tbl_gt.set_fontsize(10)
tbl_gt.scale(1, 2.2)

for j in range(n_gt_cols):
    tbl_gt[0, j].set_facecolor('#2c5f8a')
    tbl_gt[0, j].set_text_props(color='white', fontweight='bold')

# Bold the "All" row
for j in range(n_gt_cols):
    tbl_gt[n_gt_rows, j].set_text_props(fontweight='bold')

fig_gt.suptitle('Ground-Truth FCM Graph Statistics by Dataset',
                fontsize=13, fontweight='bold', y=0.98)
fig_gt.tight_layout()

out_gt = OUT_DIR / 'fig_gt_graph_stats_per_dataset.png'
fig_gt.savefig(out_gt, dpi=300, bbox_inches='tight')
plt.close(fig_gt)
print(f'[OK] {out_gt}')

print('\nDone.')
