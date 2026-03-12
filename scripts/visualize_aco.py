# visualize_aco.py
# Produces publication-quality figures for ACO results:
#   figH_aco_f1_boxplots.png     - soft-F1 box plots by model x dataset
#   figI_aco_graph_metrics.png   - graph metrics (density + degree) by model x dataset
#   figJ_aco_node_edge_counts.png - node + edge counts by model x dataset
#   figH2/I2/J2 - combined (all datasets pooled) versions

import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ACO_CSV    = Path(r'C:\Users\Nbrug\Desktop\all_aco_results.csv')
ACO_ROOT   = Path(r'C:\Users\Nbrug\Desktop\aco_adjacencies')
OUT_DIR    = Path(r'C:\Users\Nbrug\Desktop\fcm_visualizations_aco')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         8,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
})

# ACO uses same colour palette concept: one colour per model
MODEL_COLORS = {
    'gpt-5-mini':       '#2E86AB',
    'gpt-5.2':          '#3BB273',
    'gemini-2.5-flash': '#E87040',
    'gemini-3-flash':   '#9B5DE5',
    'qwen':             '#E84855',
    'mistral':          '#FF9F1C',
    'aco-qwen':         '#C04000',
    'aco-mistral':      '#1B6CA8',
}

GT_COLOR  = '#555555'
GT_ALPHA  = 0.15
BOX_ALPHA = 0.85
FLIER_SIZE = 2.0

# ── Focus models & display labels ─────────────────────────────────────────────
FOCUS_MODELS = [
    ('gpt-5-mini',       'GPT-5-mini'),
    ('gpt-5.2',          'GPT-5.2'),
    ('gemini-2.5-flash', 'Gemini\n2.5 Flash'),
    ('gemini-3-flash',   'Gemini\n3 Flash'),
    ('aco-qwen',         'ACO-Qwen'),
    ('aco-mistral',      'ACO-Mistral'),
]
MODEL_KEYS   = [m[0] for m in FOCUS_MODELS]
MODEL_LABELS = [m[1] for m in FOCUS_MODELS]

DATASET_ORDER = ['biodiversity', 'flpp', 'gulf-osw', 'red-snapper']
DS_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'gulf-osw':     'Gulf OSW',
    'red-snapper':  'Red Snapper',
}

# ── Box position helpers ───────────────────────────────────────────────────────
N_MODELS = len(FOCUS_MODELS)
BOX_W    = 0.55
GAP      = 1.4
POSITIONS = np.arange(N_MODELS) * GAP

def draw_boxes(ax, data_list, positions, colours, flier_alpha=0.5):
    bps = ax.boxplot(
        data_list,
        positions=positions,
        widths=BOX_W,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=FLIER_SIZE,
                        markerfacecolor='none', markeredgewidth=0.5,
                        alpha=flier_alpha),
        whiskerprops=dict(linewidth=0.7),
        capprops=dict(linewidth=0.7),
        medianprops=dict(color='white', linewidth=1.5),
        boxprops=dict(linewidth=0.5),
        zorder=2,
    )
    for patch, col in zip(bps['boxes'], colours):
        patch.set_facecolor(col)
        patch.set_alpha(BOX_ALPHA)
    return bps

def setup_ax(ax, ylabel=None, title=None, show_xlabels=True):
    ax.set_xticks(POSITIONS)
    if show_xlabels:
        ax.set_xticklabels(MODEL_LABELS, fontsize=6.8, rotation=25,
                           ha='right', rotation_mode='anchor')
    else:
        ax.set_xticklabels(['']*N_MODELS)
    ax.set_xlim(POSITIONS[0] - 0.6, POSITIONS[-1] + 0.6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5)
    if title:
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Load F1 data ───────────────────────────────────────────────────────────────
df = pd.read_csv(ACO_CSV)
# Normalise dataset names: merge biodiversity sub-types
df['dataset_clean'] = df['dataset_name'].str.replace(
    r'^biodiversity.*', 'biodiversity', regex=True)

# ── Section 1: F1 box plots ────────────────────────────────────────────────────
def make_f1_fig(datasets, out_name, suptitle):
    n_ds = len(datasets)
    fig, axes = plt.subplots(nrows=1, ncols=n_ds,
                              figsize=(2.8*n_ds, 4.0),
                              constrained_layout=True)
    if n_ds == 1:
        axes = [axes]

    for col_j, ds in enumerate(datasets):
        ax = axes[col_j]
        sub = df[df['dataset_clean'] == ds]

        data_list = []
        colours   = []
        for m_key in MODEL_KEYS:
            vals = sub.loc[sub['model_name'] == m_key, 'F1'].dropna().values
            data_list.append(vals if len(vals) else [np.nan])
            colours.append(MODEL_COLORS.get(m_key, '#888888'))

        draw_boxes(ax, data_list, POSITIONS, colours)
        setup_ax(ax, ylabel='Soft F1' if col_j == 0 else None,
                 title=DS_LABELS.get(ds, ds))
        ax.axhline(0.5, color='grey', linewidth=0.8, linestyle=':', alpha=0.5, zorder=1)
        ax.set_ylim(-0.02, 1.02)

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=lbl.replace('\n', ' '), alpha=BOX_ALPHA)
               for m, lbl in FOCUS_MODELS]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.05 if n_ds > 1 else -0.12),
               ncol=min(6, N_MODELS), fontsize=7, frameon=False,
               handlelength=1.4, columnspacing=0.8)
    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.01)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')

make_f1_fig(DATASET_ORDER, 'figH_aco_f1_boxplots.png',
            'ACO — Soft F1 scores by model and dataset')

# Combined (all datasets pooled) — one panel per model
def make_f1_combined_fig(out_name, suptitle):
    """Single panel with all datasets pooled, one box per model."""
    fig, ax = plt.subplots(figsize=(3.5, 4.0), constrained_layout=True)

    data_list, colours = [], []
    for m_key in MODEL_KEYS:
        vals = df.loc[df['model_name'] == m_key, 'F1'].dropna().values
        data_list.append(vals if len(vals) else [np.nan])
        colours.append(MODEL_COLORS.get(m_key, '#888888'))

    draw_boxes(ax, data_list, POSITIONS, colours)
    setup_ax(ax, ylabel='Soft F1', title='All datasets (pooled)')
    ax.axhline(0.5, color='grey', linewidth=0.8, linestyle=':', alpha=0.5, zorder=1)
    ax.set_ylim(-0.02, 1.02)

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=lbl.replace('\n', ' '), alpha=BOX_ALPHA)
               for m, lbl in FOCUS_MODELS]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.12),
               ncol=min(6, N_MODELS), fontsize=7, frameon=False,
               handlelength=1.4, columnspacing=0.8)
    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.01)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')

make_f1_combined_fig('figH2_aco_f1_combined.png',
                     'ACO — Soft F1 (all datasets combined)')

# ── Section 2: Graph metrics from ACO adjacency JSONs ─────────────────────────

def normalize_bio(name):
    name = re.sub(r'\s*\(\d+\)$', '', name.strip())
    name = re.sub(r'\(\d+\)$', '', name.strip())
    return name.rstrip('_').strip()

def normalize_osw(name):
    if ' - ' in name:
        return name.split(' - ')[-1].strip()
    if ' -' in name:
        return name.split(' -')[-1].strip()
    return name.strip()

def normalize_rs(name):
    parts = name.split('_')
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1]
    return parts[0]

def get_gt_id(raw, dataset):
    if dataset == 'biodiversity': return normalize_bio(raw)
    if dataset == 'osw':          return normalize_osw(raw)
    if dataset == 'red_snapper':  return normalize_rs(raw)
    return raw

INTERNAL_DS = ['biodiversity', 'flpp', 'osw', 'red_snapper']
DS_CLEAN    = {'biodiversity': 'biodiversity', 'flpp': 'flpp',
               'osw': 'gulf-osw', 'red_snapper': 'red-snapper'}

def aco_json_metrics(json_path):
    data = json.loads(json_path.read_text(encoding='utf-8'))
    edges = data.get('edges', [])
    id2concept = {n['id']: n.get('concepts', n['id']) for n in data.get('nodes', [])}
    nodes = set()
    for e in edges:
        nodes.add(id2concept.get(e['source'], e['source']))
        nodes.add(id2concept.get(e['target'], e['target']))
    n_nodes = len(nodes)
    n_edges = len([e for e in edges if float(e.get('weight', 0)) != 0])
    if n_nodes < 2:
        return dict(n_nodes=n_nodes, n_edges=0,
                    edge_density=0.0, pos_edge_density=0.0,
                    neg_edge_density=0.0, max_total_degree=0.0)
    max_edges = n_nodes * (n_nodes - 1)
    pos = sum(1 for e in edges if float(e.get('weight', 0)) > 0)
    neg = sum(1 for e in edges if float(e.get('weight', 0)) < 0)
    # degree
    deg = {}
    for e in edges:
        s = id2concept.get(e['source'], e['source'])
        t = id2concept.get(e['target'], e['target'])
        w = float(e.get('weight', 0))
        if w != 0:
            deg[s] = deg.get(s, 0) + 1
            deg[t] = deg.get(t, 0) + 1
    max_deg = max(deg.values()) if deg else 0
    return dict(
        n_nodes=n_nodes, n_edges=n_edges,
        edge_density=n_edges / max_edges,
        pos_edge_density=pos / max_edges,
        neg_edge_density=neg / max_edges,
        max_total_degree=float(max_deg),
    )

def gt_csv_metrics(csv_path):
    import numpy as np
    mat = pd.read_csv(csv_path, index_col=0, header=0)
    mat = mat[[c for c in mat.columns if not str(c).startswith('Unnamed:')]]
    def _to_num(v):
        if v == 0 or v == '0': return 0.0
        try: return float(v)
        except: return 1.0 if str(v).strip() not in ('', 'nan') else 0.0
    arr = mat.map(_to_num).values.astype(float)
    # make square
    n = min(arr.shape)
    arr = arr[:n, :n]
    n_nodes = arr.shape[0]
    if n_nodes < 2:
        return dict(n_nodes=n_nodes, n_edges=0, edge_density=0.0,
                    pos_edge_density=0.0, neg_edge_density=0.0, max_total_degree=0.0)
    max_edges = n_nodes * (n_nodes - 1)
    n_edges = int((arr != 0).sum())
    pos = int((arr > 0).sum())
    neg = int((arr < 0).sum())
    degree = (arr != 0).sum(axis=0) + (arr != 0).sum(axis=1)
    max_deg = float(degree.max()) if n_edges > 0 else 0.0
    return dict(n_nodes=n_nodes, n_edges=n_edges,
                edge_density=n_edges/max_edges,
                pos_edge_density=pos/max_edges,
                neg_edge_density=neg/max_edges,
                max_total_degree=max_deg)

# Collect metrics rows
metric_rows = []

# GT
for ds in INTERNAL_DS:
    gt_dir = ACO_ROOT / 'gt' / ds
    for csv_p in sorted(gt_dir.glob('*.csv')):
        m = gt_csv_metrics(csv_p)
        m['method'] = 'gt'; m['dataset'] = DS_CLEAN[ds]; m['file_id'] = csv_p.stem
        metric_rows.append(m)

# ACO models
ACO_MODEL_DIRS = [
    ('aco_mistral', 'aco-mistral'),
    ('aco_qwen',    'aco-qwen'),
]
for model_dir, model_key in ACO_MODEL_DIRS:
    for ds in INTERNAL_DS:
        ai_dir = ACO_ROOT / model_dir / ds
        if not ai_dir.exists():
            continue
        for p_dir in sorted(p for p in ai_dir.iterdir() if p.is_dir()):
            jsons = list(p_dir.glob('*_fcm.json'))
            if not jsons:
                continue
            m = aco_json_metrics(jsons[0])
            gt_key = get_gt_id(p_dir.name, ds)
            m['method'] = model_key; m['dataset'] = DS_CLEAN[ds]; m['file_id'] = gt_key
            metric_rows.append(m)

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(OUT_DIR / 'aco_graph_metrics_all.csv', index=False)
print(f'Metrics: {len(metrics_df)} rows, methods: {sorted(metrics_df["method"].unique())}')

# ── Graph metric plots ─────────────────────────────────────────────────────────
GT_COLOR  = '#555555'
GT_ALPHA  = 0.15

ACO_COLORS = {
    'aco-mistral': '#FF9F1C',
    'aco-qwen':    '#E84855',
}
ACO_MODEL_LIST = [('aco-mistral', 'ACO\nMistral'), ('aco-qwen', 'ACO\nQwen')]
ACO_KEYS   = [m[0] for m in ACO_MODEL_LIST]
ACO_LABELS = [m[1] for m in ACO_MODEL_LIST]
ACO_POS    = np.arange(len(ACO_MODEL_LIST)) * GAP

def draw_metric_panel(ax, dataset, metric, df_m, ylabel=None, title=None, show_xlabel=True):
    sub = df_m[df_m['dataset'] == dataset]
    gt_vals = sub.loc[sub['method'] == 'gt', metric].dropna().values
    if len(gt_vals):
        gm, gs = gt_vals.mean(), gt_vals.std()
        ax.axhline(gm, color=GT_COLOR, linewidth=1.2, linestyle='--', zorder=1)
        ax.axhspan(gm - gs, gm + gs, color=GT_COLOR, alpha=GT_ALPHA, zorder=0)
    data_list = []
    colours   = []
    for mk in ACO_KEYS:
        v = sub.loc[sub['method'] == mk, metric].dropna().values
        data_list.append(v if len(v) else [np.nan])
        colours.append(ACO_COLORS[mk])
    draw_boxes(ax, data_list, ACO_POS, colours)
    ax.set_xticks(ACO_POS)
    if show_xlabel:
        ax.set_xticklabels(ACO_LABELS, fontsize=7)
    else:
        ax.set_xticklabels(['']*len(ACO_KEYS))
    ax.set_xlim(ACO_POS[0]-0.5, ACO_POS[-1]+0.5)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5)
    if title:  ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

METRICS_DENSITY = [
    ('edge_density',     'Edge density'),
    ('pos_edge_density', 'Positive edge density'),
    ('neg_edge_density', 'Negative edge density'),
    ('max_total_degree', 'Max total degree'),
]
METRICS_COUNTS = [
    ('n_nodes', 'Node count'),
    ('n_edges', 'Edge count'),
]

CLEAN_DS_LIST = ['biodiversity', 'flpp', 'gulf-osw', 'red-snapper']
DS_LABEL_LIST = ['Biodiversity', 'FLPP', 'Gulf OSW', 'Red Snapper']

def make_metric_fig(metric_list, out_name, suptitle):
    fig, axes = plt.subplots(nrows=len(metric_list), ncols=len(CLEAN_DS_LIST),
                              figsize=(10, 2.5*len(metric_list)),
                              constrained_layout=True)
    if len(metric_list) == 1:
        axes = [axes]
    for row_i, (metric, ylabel) in enumerate(metric_list):
        for col_j, (ds, ds_lbl) in enumerate(zip(CLEAN_DS_LIST, DS_LABEL_LIST)):
            ax = axes[row_i][col_j]
            show_x = (row_i == len(metric_list)-1)
            ttl = ds_lbl if row_i == 0 else None
            yl  = ylabel if col_j == 0 else None
            draw_metric_panel(ax, ds, metric, metrics_df, ylabel=yl, title=ttl, show_xlabel=show_x)

    handles = [mpatches.Patch(color=ACO_COLORS[m], label=lbl.replace('\n', ' '), alpha=BOX_ALPHA)
               for m, lbl in ACO_MODEL_LIST]
    handles += [mlines.Line2D([], [], color=GT_COLOR, linestyle='--', linewidth=1.2, label='GT mean'),
                mpatches.Patch(color=GT_COLOR, alpha=GT_ALPHA, label='GT ±1 SD')]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.04),
               ncol=4, fontsize=7, frameon=False, handlelength=1.4, columnspacing=0.8)
    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.01)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')

make_metric_fig(METRICS_DENSITY, 'figI_aco_graph_density_degree.png',
                'ACO — Graph-Structure Metrics: Edge Density & Degree\n(dashed = GT mean, shaded = GT ±1 SD)')
make_metric_fig(METRICS_COUNTS,  'figJ_aco_node_edge_counts.png',
                'ACO — Graph-Structure Metrics: Node & Edge Counts\n(dashed = GT mean, shaded = GT ±1 SD)')

# Combined (all datasets pooled)
def make_metric_fig_combined(metric_list, out_name, suptitle):
    df_all = metrics_df.copy()
    df_all['dataset'] = 'all'
    df_c = pd.concat([metrics_df, df_all], ignore_index=True)

    fig, axes = plt.subplots(nrows=len(metric_list), ncols=1,
                              figsize=(4.0, 2.5*len(metric_list)),
                              constrained_layout=True)
    if len(metric_list) == 1:
        axes = [axes]
    for row_i, (metric, ylabel) in enumerate(metric_list):
        ax = axes[row_i]
        show_x = (row_i == len(metric_list)-1)
        draw_metric_panel(ax, 'all', metric, df_c, ylabel=ylabel, show_xlabel=show_x)

    handles = [mpatches.Patch(color=ACO_COLORS[m], label=lbl.replace('\n', ' '), alpha=BOX_ALPHA)
               for m, lbl in ACO_MODEL_LIST]
    handles += [mlines.Line2D([], [], color=GT_COLOR, linestyle='--', linewidth=1.2, label='GT mean'),
                mpatches.Patch(color=GT_COLOR, alpha=GT_ALPHA, label='GT ±1 SD')]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.06),
               ncol=4, fontsize=7, frameon=False, handlelength=1.4, columnspacing=0.8)
    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.01)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')

make_metric_fig_combined(METRICS_DENSITY, 'figI2_aco_graph_density_combined.png',
                         'ACO — Edge Density & Degree (all datasets combined)')
make_metric_fig_combined(METRICS_COUNTS,  'figJ2_aco_node_edge_combined.png',
                         'ACO — Node & Edge Counts (all datasets combined)')

# ── Section 3: Node & edge count correlations (GT vs ACO) ─────────────────────
from scipy import stats as sp_stats

# Dataset style matching visualize_all_methods.py
SCATTER_DS_ORDER  = ['biodiversity', 'flpp', 'gulf-osw', 'red-snapper']
SCATTER_DS_LABELS = {
    'biodiversity': 'Biodiversity',
    'flpp':         'FLPP',
    'gulf-osw':     'Gulf OSW',
    'red-snapper':  'Red Snapper',
}
SCATTER_DS_COLORS = {
    'biodiversity': '#264653',
    'flpp':         '#2A9D8F',
    'gulf-osw':     '#E9C46A',
    'red-snapper':  '#E76F51',
}
SCATTER_DS_MARKERS = {
    'biodiversity': 'o',
    'flpp':         's',
    'gulf-osw':     '^',
    'red-snapper':  'D',
}

gt_m = metrics_df[metrics_df['method'] == 'gt'][
    ['dataset', 'file_id', 'n_nodes', 'n_edges']].copy()

corr_rows = []

def _scatter_corr_panels(metric, ax_label, out_name, fig_title):
    """One panel per ACO model; points coloured/shaped by dataset.
    Matches _scatter_panels style from visualize_all_methods.py."""
    ncols = len(ACO_MODEL_LIST)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))
    fig.suptitle(fig_title, fontsize=13, fontweight='bold', y=1.02)

    for idx, (model_key, model_label) in enumerate(ACO_MODEL_LIST):
        ax = axes[idx]
        aco_m = metrics_df[metrics_df['method'] == model_key][
            ['dataset', 'file_id', metric]].copy()
        paired = gt_m.merge(aco_m, on=['dataset', 'file_id'],
                            suffixes=('_gt', '_aco'))

        all_vals = pd.concat([paired[f'{metric}_gt'],
                               paired[f'{metric}_aco']]).dropna()
        ax_max = all_vals.max() * 1.08 if len(all_vals) else 1

        # Identity line
        ax.plot([0, ax_max], [0, ax_max], 'k--', linewidth=1.0, alpha=0.4, zorder=1)

        r_lines = []
        for ds in SCATTER_DS_ORDER:
            sub = paired[paired['dataset'] == ds].dropna(
                subset=[f'{metric}_gt', f'{metric}_aco'])
            if sub.empty:
                continue
            ax.scatter(sub[f'{metric}_gt'], sub[f'{metric}_aco'],
                       color=SCATTER_DS_COLORS[ds],
                       marker=SCATTER_DS_MARKERS[ds],
                       s=45, alpha=0.75, edgecolors='none', zorder=3)
            if len(sub) >= 2:
                r, p = sp_stats.pearsonr(sub[f'{metric}_gt'],
                                          sub[f'{metric}_aco'])
                r_lines.append((ds, r, p, len(sub)))
                corr_rows.append(dict(model=model_key, dataset=ds, metric=metric,
                                      n=len(sub), r=round(r, 4), p=round(p, 4)))

        ax.set_xlim(0, ax_max)
        ax.set_ylim(0, ax_max)
        ax.set_xlabel(f'GT {ax_label}', fontsize=9)
        ax.set_ylabel(f'ACO {ax_label}', fontsize=9)
        ax.set_title(model_label, fontsize=9, fontweight='bold', pad=5)
        ax.set_aspect('equal', adjustable='box')

        # Overall r (bottom-right)
        overall = paired.dropna(subset=[f'{metric}_gt', f'{metric}_aco'])
        if len(overall) >= 2:
            r_all, _ = sp_stats.pearsonr(overall[f'{metric}_gt'],
                                          overall[f'{metric}_aco'])
            ax.text(ax_max * 0.97, ax_max * 0.03,
                    f'Overall r={r_all:.2f}',
                    color='black', fontsize=8, va='bottom', ha='right',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='#cccccc', alpha=0.85),
                    zorder=6)

        # Per-dataset r (top-left)
        if r_lines:
            y_pos  = ax_max * 0.98
            line_h = ax_max * 0.075
            for i, (ds, r, p, n) in enumerate(r_lines):
                ax.text(ax_max * 0.03, y_pos - i * line_h,
                        f'{SCATTER_DS_LABELS[ds]}: r={r:.2f}',
                        color=SCATTER_DS_COLORS[ds], fontsize=7, va='top',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec='none', alpha=0.55),
                        zorder=5)

    # Dataset legend
    ds_handles = [
        mlines.Line2D([], [], color=SCATTER_DS_COLORS[ds],
                      marker=SCATTER_DS_MARKERS[ds], linestyle='None',
                      markersize=7, label=SCATTER_DS_LABELS[ds])
        for ds in SCATTER_DS_ORDER
    ]
    fig.legend(handles=ds_handles, title='Dataset', title_fontsize=9,
               loc='lower center', ncol=4, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), borderpad=0.8)

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')

_scatter_corr_panels('n_nodes', 'Node count',
                     'figK_aco_node_corr.png',
                     'ACO vs GT — Node Count Correlation')
_scatter_corr_panels('n_edges', 'Edge count',
                     'figK2_aco_edge_corr.png',
                     'ACO vs GT — Edge Count Correlation')

# Save correlation summary
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT_DIR / 'aco_node_edge_correlations.csv', index=False)
print('\nCorrelation summary:')
print(corr_df.to_string(index=False))

print('\nDone.')
