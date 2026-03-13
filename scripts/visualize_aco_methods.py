"""
visualize_aco_methods.py
F1-score box plots and graph-metric plots for the ACO method set.
Uses the same colour palette and style as visualize_all_methods.py.

Models shown: gpt-5-mini, gpt-5.2, gemini-2.5-flash, gemini-3-flash, qwen, mistral
              (plus aco-mistral, aco-qwen when present)

Outputs (in fcm_visualizations_aco/):
  figACO_A_f1_by_dataset.png      — soft F1 per model × dataset (boxplot)
  figACO_B_f1_combined.png        — soft F1 all datasets combined
  figACO_C_density_degree.png     — edge density / ±density / max degree × dataset
  figACO_D_density_combined.png   — same, all datasets combined
  figACO_E_nodes_edges.png        — node & edge counts × dataset
  figACO_F_nodes_edges_combined.png — same, all datasets combined
"""

import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ──────────────────────────────────────────────────────────────────────
ALL_ACO_CSV    = Path(r'C:\Users\Nbrug\Desktop\all_aco_results_final.csv')
ACO_ADJ_DIR    = Path(r'C:\Users\Nbrug\Desktop\aco_adjacencies')
GT_DIR         = ACO_ADJ_DIR / 'gt'
OUT_DIR        = Path(r'C:\Users\Nbrug\Desktop\fcm_visualizations_aco')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':  'sans-serif',
    'font.size':    8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

# Colour per model (same hues, new labels)
MODEL_COLORS = {
    'aco-gpt-5-mini':        '#2E86AB',
    'aco-gpt-5.2':           '#3BB273',
    'aco-gemini-2.5-flash':  '#E87040',
    'aco-gemini-3-flash':    '#9B5DE5',
    'aco-qwen':              '#C84B9E',   # distinct magenta
    'aco-mistral':           '#8B4513',   # sienna
}

MODEL_LABELS = {
    'aco-gpt-5-mini':        'GPT-5-mini\n+ACO',
    'aco-gpt-5.2':           'GPT-5.2\n+ACO',
    'aco-gemini-2.5-flash':  'Gemini 2.5\nFlash+ACO',
    'aco-gemini-3-flash':    'Gemini 3\nFlash+ACO',
    'aco-qwen':              'ACO-Qwen',
    'aco-mistral':           'ACO-Mistral',
}

BOX_ALPHA  = 0.85
FLIER_SIZE = 2.0
GT_COLOR   = '#555555'
GT_ALPHA   = 0.15

# Models in display order
MODEL_ORDER = ['aco-gpt-5-mini', 'aco-gpt-5.2', 'aco-gemini-2.5-flash', 'aco-gemini-3-flash',
               'aco-qwen', 'aco-mistral']

# Dataset order used by aco results
DATASET_ORDER  = ['biodiversity', 'flpp', 'gulf-osw', 'red-snapper']
DATASET_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'gulf-osw':     'Gulf OSW',
    'red-snapper':  'Red Snapper',
}

# ── Load F1 data ───────────────────────────────────────────────────────────────
f1_df = pd.read_csv(ALL_ACO_CSV)

# Normalise biodiversity quality splits → 'biodiversity'
f1_df['dataset_name'] = f1_df['dataset_name'].replace({
    'biodiversity-high-quality':   'biodiversity',
    'biodiversity-medium-quality': 'biodiversity',
    'biodiversity-low-quality':    'biodiversity',
})

# Keep only the 8 target models that are present
available_models = [m for m in MODEL_ORDER if m in f1_df['model_name'].unique()]
f1_df = f1_df[f1_df['model_name'].isin(available_models)].copy()

print(f'Models in data: {sorted(f1_df["model_name"].unique())}')
print(f'Datasets:       {sorted(f1_df["dataset_name"].unique())}')


# ── Graph metrics: read ACO adjacencies ───────────────────────────────────────
def _to_num(v):
    if v == 0 or v == '0':
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        sv = str(v).strip()
        if sv in ('', 'nan', 'NaN'):
            return 0.0
        return 1.0   # 'Neutral' → +1 (same convention as graph_metrics_analysis.py)


def read_gt_csv(path: Path) -> np.ndarray | None:
    try:
        mat = pd.read_csv(path, index_col=0, header=0)
        mat = mat[[c for c in mat.columns if not str(c).startswith('Unnamed:')]]
        return mat.map(_to_num).values.astype(float)
    except Exception as e:
        print(f'  WARN GT read failed: {path.name} — {e}')
        return None


def read_aco_json(path: Path) -> np.ndarray | None:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        edges = data.get('edges', [])
        if not edges:
            nodes = []
        else:
            nodes = sorted({str(e['source']) for e in edges} |
                            {str(e['target']) for e in edges})
        if not nodes:
            return np.zeros((1, 1))
        n = len(nodes)
        idx = {nd: i for i, nd in enumerate(nodes)}
        mat = np.zeros((n, n))
        for e in edges:
            i, j = idx[str(e['source'])], idx[str(e['target'])]
            mat[i, j] = float(e.get('weight', 0.0))
        return mat
    except Exception as e:
        print(f'  WARN ACO JSON read failed: {path.name} — {e}')
        return None


def graph_metrics_from_mat(mat: np.ndarray) -> dict:
    if mat is None or mat.size == 0:
        return {}
    n = mat.shape[0]
    total_possible = n * (n - 1)   # exclude diagonal
    pos = (mat > 0).sum() - (np.diag(mat) > 0).sum()
    neg = (mat < 0).sum() - (np.diag(mat) < 0).sum()
    all_edges = pos + neg
    ed  = (all_edges / total_possible)  if total_possible > 0 else 0.0
    ped = (pos        / total_possible)  if total_possible > 0 else 0.0
    ned = (neg        / total_possible)  if total_possible > 0 else 0.0
    # Degree = row-wise + col-wise non-zeros (excluding diagonal)
    np.fill_diagonal(mat, 0)
    out_deg = (mat != 0).sum(axis=1)
    in_deg  = (mat != 0).sum(axis=0)
    total_deg = out_deg + in_deg
    return {
        'n_nodes':          n,
        'n_edges':          int(all_edges),
        'edge_density':     ed,
        'pos_edge_density': ped,
        'neg_edge_density': ned,
        'max_total_degree': int(total_deg.max()) if len(total_deg) else 0,
    }


def participant_to_gt_stem(dataset_folder: str, participant_id: str) -> str:
    if dataset_folder in ('biodiversity', 'flpp'):
        return participant_id
    elif dataset_folder == 'osw':
        return participant_id.split(' - ')[-1].strip()
    elif dataset_folder == 'red_snapper':
        parts = participant_id.split('_')
        return parts[1] if len(parts) > 1 else participant_id
    return participant_id


DS_FOLDER_MAP = {
    'biodiversity': 'biodiversity',
    'flpp':         'flpp',
    'gulf-osw':     'osw',
    'red-snapper':  'red_snapper',
}

def collect_graph_metrics() -> pd.DataFrame:
    """Compute graph metrics for GT + all 8 ACO models."""
    rows = []

    # GT
    for ds_name, ds_folder in DS_FOLDER_MAP.items():
        gt_ds_dir = GT_DIR / ds_folder
        if not gt_ds_dir.exists():
            continue
        for csv_f in sorted(gt_ds_dir.glob('*.csv')):
            mat = read_gt_csv(csv_f)
            if mat is None:
                continue
            m = graph_metrics_from_mat(mat)
            if m:
                rows.append({'model': 'gt', 'dataset': ds_name,
                             'file_id': csv_f.stem, **m})

    # AI models stored as CSVs (existing methods)
    AI_CSV_MODELS = {
        'gpt-5-mini': 'gpt-5-mini',
        'gpt-5.2':    'gpt-5.2',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-3-flash':   'gemini-3-flash',
        'qwen':    'qwen',
        'mistral': 'mistral',
    }
    # These CSVs don't exist separately for ACO — we derive node/edge stats
    # directly from the all_aco_results.csv for non-ACO models
    for _, row in f1_df.iterrows():
        m_name = row['model_name']
        if m_name not in AI_CSV_MODELS:
            continue
        ds_name = row['dataset_name']
        # Use fcm2_nodes/fcm2_edges (AI FCM) from scoring results
        n = int(row['fcm2_nodes']) if pd.notna(row['fcm2_nodes']) else 0
        e = int(row['fcm2_edges']) if pd.notna(row['fcm2_edges']) else 0
        total_p = n * (n - 1) if n > 1 else 1
        rows.append({
            'model':   m_name,
            'dataset': ds_name,
            'file_id': row['interview_file_name'],
            'n_nodes': n,
            'n_edges': e,
            'edge_density': e / total_p,
            'pos_edge_density': np.nan,
            'neg_edge_density': np.nan,
            'max_total_degree': np.nan,
        })

    # ACO models: parse JSON adjacencies
    for aco_key, aco_label in [('aco_qwen', 'aco-qwen'), ('aco_mistral', 'aco-mistral')]:
        aco_model_dir = ACO_ADJ_DIR / aco_key
        if not aco_model_dir.exists():
            continue
        for ds_dir in sorted(aco_model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds_folder = ds_dir.name
            ds_name   = {v: k for k, v in DS_FOLDER_MAP.items()}.get(ds_folder, ds_folder)
            for part_dir in sorted(ds_dir.iterdir()):
                if not part_dir.is_dir():
                    continue
                jsons = [f for f in part_dir.iterdir()
                         if f.suffix == '.json' and f.stem.endswith('_fcm')]
                if not jsons:
                    continue
                mat = read_aco_json(jsons[0])
                if mat is None:
                    continue
                m = graph_metrics_from_mat(mat)
                if m:
                    rows.append({'model': aco_label, 'dataset': ds_name,
                                 'file_id': participant_to_gt_stem(ds_folder, part_dir.name),
                                 **m})

    return pd.DataFrame(rows)


print('Computing graph metrics …')
gm_df = collect_graph_metrics()
print(f'Graph metrics: {len(gm_df)} rows  |  models: {sorted(gm_df["model"].unique())}')


# ── Box-plot helpers ───────────────────────────────────────────────────────────
def box_positions(n_models, box_gap=0.28, group_gap=0.9):
    positions, centres = [], []
    x = 0.0
    for _ in range(n_models):
        positions.append(x)
        centres.append(x)
        x += box_gap + group_gap
    return np.array(positions), np.array(centres)


def draw_boxes(ax, data_source_df, value_col, model_col, dataset_filter=None,
               show_gt_band=True, show_xlabel=True, ylabel=None, title=None,
               models=None):
    if models is None:
        models = [m for m in available_models]
    sub = data_source_df.copy()
    if dataset_filter:
        sub = sub[sub['dataset'] == dataset_filter] if 'dataset' in sub.columns \
              else sub[sub['dataset_name'] == dataset_filter]

    # GT band
    if show_gt_band and 'model' in sub.columns:
        gt_vals = sub.loc[sub['model'] == 'gt', value_col].dropna().values
        if len(gt_vals):
            ax.axhline(gt_vals.mean(), color=GT_COLOR, linewidth=1.2,
                       linestyle='--', zorder=1)
            ax.axhspan(gt_vals.mean() - gt_vals.std(),
                       gt_vals.mean() + gt_vals.std(),
                       color=GT_COLOR, alpha=GT_ALPHA, zorder=0)

    data_list, colors = [], []
    for m in models:
        col = model_col
        mask = (sub[col] == m) if col in sub.columns else pd.Series(False, index=sub.index)
        vals = sub.loc[mask, value_col].dropna().values
        data_list.append(vals if len(vals) else [np.nan])
        colors.append(MODEL_COLORS.get(m, '#999999'))

    pos, _ = box_positions(len(models))
    bps = ax.boxplot(
        data_list, positions=pos, widths=0.25,
        patch_artist=True, notch=False, showfliers=True,
        flierprops=dict(marker='o', markersize=FLIER_SIZE,
                        markerfacecolor='none', markeredgewidth=0.5, alpha=0.5),
        whiskerprops=dict(linewidth=0.7),
        capprops=dict(linewidth=0.7),
        medianprops=dict(color='white', linewidth=1.5),
        boxprops=dict(linewidth=0.5),
        zorder=2,
    )
    for patch, c in zip(bps['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(BOX_ALPHA)

    ax.set_xticks(pos)
    if show_xlabel:
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           fontsize=6.5, rotation=30, ha='right',
                           rotation_mode='anchor')
    else:
        ax.set_xticklabels([] * len(models))

    ax.set_xlim(pos[0] - 0.4, pos[-1] + 0.4)
    if title:
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# Shared legend handles
legend_handles = [
    mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m].replace('\n', ' '),
                   alpha=BOX_ALPHA)
    for m in available_models
] + [
    mlines.Line2D([], [], color=GT_COLOR, linewidth=1.2, linestyle='--', label='GT mean'),
    mpatches.Patch(color=GT_COLOR, alpha=GT_ALPHA, label='GT ±1 SD'),
]

# ── Fig A — F1 per dataset ─────────────────────────────────────────────────────
available_datasets = [d for d in DATASET_ORDER
                      if d in f1_df['dataset_name'].unique()]
n_ds = len(available_datasets)

fig_a, axes_a = plt.subplots(1, n_ds, figsize=(3.2 * n_ds, 4.5),
                               constrained_layout=True)
if n_ds == 1:
    axes_a = [axes_a]

for j, ds in enumerate(available_datasets):
    ax = axes_a[j]
    draw_boxes(ax, f1_df.rename(columns={'model_name': 'model',
                                          'dataset_name': 'dataset'}),
               value_col='F1', model_col='model',
               dataset_filter=ds,
               show_gt_band=False,
               show_xlabel=True,
               ylabel='Soft F1' if j == 0 else None,
               title=DATASET_LABELS.get(ds, ds))
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)

fig_a.legend(handles=legend_handles, loc='lower center',
             bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=7,
             frameon=False, handlelength=1.4)
fig_a.suptitle('ACO Methods — Soft F1 Score by Dataset', fontsize=10,
               fontweight='bold', y=1.01)
out = OUT_DIR / 'figACO_A_f1_by_dataset.png'
fig_a.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig_a)
print(f'Saved → {out}')

# ── Fig B — F1 all datasets combined ──────────────────────────────────────────
fig_b, ax_b = plt.subplots(1, 1, figsize=(6, 4.5), constrained_layout=True)
draw_boxes(ax_b, f1_df.rename(columns={'model_name': 'model',
                                        'dataset_name': 'dataset'}),
           value_col='F1', model_col='model',
           show_gt_band=False, show_xlabel=True,
           ylabel='Soft F1', title='All datasets combined')
ax_b.set_ylim(0, 1)
ax_b.axhline(0.5, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)
fig_b.legend(handles=legend_handles, loc='lower center',
             bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=7,
             frameon=False, handlelength=1.4)
fig_b.suptitle('ACO Methods — Soft F1 Score (All Datasets)',
               fontsize=10, fontweight='bold', y=1.01)
out = OUT_DIR / 'figACO_B_f1_combined.png'
fig_b.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig_b)
print(f'Saved → {out}')


# ── Figs C/D — edge density & degree ──────────────────────────────────────────
GRAPH_METRICS_F = [
    ('edge_density',     'Edge density'),
    ('pos_edge_density', 'Positive edge density'),
    ('neg_edge_density', 'Negative edge density'),
    ('max_total_degree', 'Max total degree'),
]

aco_models_gm  = [m for m in available_models if m.startswith('aco')]
non_aco_models = [m for m in available_models if not m.startswith('aco')]

# For non-ACO models we only have n_nodes/n_edges from scoring results.
# For GT and ACO models we have full metrics.
# So for density/degree plots, show only GT + ACO models.
gm_models = ['gt'] + aco_models_gm
gm_avail  = [m for m in gm_models if m in gm_df['model'].unique()]

gm_ds_avail = [d for d in DATASET_ORDER
               if d in gm_df['dataset'].unique()]

# Fig C — per dataset
fig_c, axes_c = plt.subplots(len(GRAPH_METRICS_F), len(gm_ds_avail),
                               figsize=(3.5 * len(gm_ds_avail), 10),
                               constrained_layout=True)

for row_i, (metric, ylabel_base) in enumerate(GRAPH_METRICS_F):
    for col_j, ds in enumerate(gm_ds_avail):
        ax = axes_c[row_i, col_j]
        draw_boxes(ax, gm_df, value_col=metric, model_col='model',
                   dataset_filter=ds, show_gt_band=True,
                   show_xlabel=(row_i == len(GRAPH_METRICS_F) - 1),
                   ylabel=ylabel_base if col_j == 0 else None,
                   title=DATASET_LABELS.get(ds, ds) if row_i == 0 else None,
                   models=[m for m in gm_avail if m != 'gt'])

gm_legend = [
    mpatches.Patch(color=MODEL_COLORS.get(m, '#999'), label=MODEL_LABELS.get(m, m),
                   alpha=BOX_ALPHA)
    for m in gm_avail if m != 'gt'
] + [
    mlines.Line2D([], [], color=GT_COLOR, linewidth=1.2, linestyle='--', label='GT mean'),
    mpatches.Patch(color=GT_COLOR, alpha=GT_ALPHA, label='GT ±1 SD'),
]
fig_c.legend(handles=gm_legend, loc='lower center',
             bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=7,
             frameon=False, handlelength=1.4)
fig_c.suptitle('ACO Methods — Graph Metrics: Edge Density & Degree\n'
               '(Dashed = GT mean, shaded = GT ±1 SD)',
               fontsize=10, fontweight='bold', y=1.01)
out = OUT_DIR / 'figACO_C_density_degree.png'
fig_c.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig_c)
print(f'Saved → {out}')

# Fig D — combined
fig_d, axes_d = plt.subplots(len(GRAPH_METRICS_F), 1,
                               figsize=(5.5, 10), constrained_layout=True)
for row_i, (metric, ylabel_base) in enumerate(GRAPH_METRICS_F):
    ax = axes_d[row_i]
    draw_boxes(ax, gm_df, value_col=metric, model_col='model',
               show_gt_band=True,
               show_xlabel=(row_i == len(GRAPH_METRICS_F) - 1),
               ylabel=ylabel_base,
               models=[m for m in gm_avail if m != 'gt'])

fig_d.legend(handles=gm_legend, loc='lower center',
             bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7,
             frameon=False, handlelength=1.4)
fig_d.suptitle('ACO Methods — Graph Metrics: Edge Density & Degree\n'
               'All datasets combined',
               fontsize=10, fontweight='bold', y=1.01)
out = OUT_DIR / 'figACO_D_density_combined.png'
fig_d.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig_d)
print(f'Saved → {out}')


# ── Figs E/F — node & edge counts ─────────────────────────────────────────────
GRAPH_METRICS_G = [
    ('n_nodes', 'Node count'),
    ('n_edges', 'Edge count'),
]
# For these we can include all 8 models (n_nodes/n_edges from scoring CSV)
# Map f1_df fcm2 columns to gm_df for non-ACO models (already loaded above).
all_gm_models = [m for m in available_models if m != 'gt']
all_gm_avail  = ['gt'] + [m for m in all_gm_models if m in gm_df['model'].unique()]

ne_legend = [
    mpatches.Patch(color=MODEL_COLORS.get(m, '#999'), label=MODEL_LABELS.get(m, m),
                   alpha=BOX_ALPHA)
    for m in all_gm_avail if m != 'gt'
] + [
    mlines.Line2D([], [], color=GT_COLOR, linewidth=1.2, linestyle='--', label='GT mean'),
    mpatches.Patch(color=GT_COLOR, alpha=GT_ALPHA, label='GT ±1 SD'),
]

fig_e, axes_e = plt.subplots(len(GRAPH_METRICS_G), len(gm_ds_avail),
                               figsize=(3.5 * len(gm_ds_avail), 6),
                               constrained_layout=True)
for row_i, (metric, ylabel_base) in enumerate(GRAPH_METRICS_G):
    for col_j, ds in enumerate(gm_ds_avail):
        ax = axes_e[row_i, col_j]
        draw_boxes(ax, gm_df, value_col=metric, model_col='model',
                   dataset_filter=ds, show_gt_band=True,
                   show_xlabel=(row_i == len(GRAPH_METRICS_G) - 1),
                   ylabel=ylabel_base if col_j == 0 else None,
                   title=DATASET_LABELS.get(ds, ds) if row_i == 0 else None,
                   models=[m for m in all_gm_avail if m != 'gt'])

fig_e.legend(handles=ne_legend, loc='lower center',
             bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=7,
             frameon=False, handlelength=1.4)
fig_e.suptitle('ACO Methods — Graph Metrics: Node & Edge Counts\n'
               '(Dashed = GT mean, shaded = GT ±1 SD)',
               fontsize=10, fontweight='bold', y=1.01)
out = OUT_DIR / 'figACO_E_nodes_edges.png'
fig_e.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig_e)
print(f'Saved → {out}')

# Fig F — combined
fig_f, axes_f = plt.subplots(len(GRAPH_METRICS_G), 1,
                               figsize=(5.5, 6), constrained_layout=True)
for row_i, (metric, ylabel_base) in enumerate(GRAPH_METRICS_G):
    ax = axes_f[row_i]
    draw_boxes(ax, gm_df, value_col=metric, model_col='model',
               show_gt_band=True,
               show_xlabel=(row_i == len(GRAPH_METRICS_G) - 1),
               ylabel=ylabel_base,
               models=[m for m in all_gm_avail if m != 'gt'])

fig_f.legend(handles=ne_legend, loc='lower center',
             bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=7,
             frameon=False, handlelength=1.4)
fig_f.suptitle('ACO Methods — Graph Metrics: Node & Edge Counts\n'
               'All datasets combined',
               fontsize=10, fontweight='bold', y=1.01)
out = OUT_DIR / 'figACO_F_nodes_edges_combined.png'
fig_f.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig_f)
print(f'Saved → {out}')

print('\nAll ACO visualizations done.')
