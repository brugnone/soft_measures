"""
FCM Graph Metrics Analysis
==========================
1. Copies / organises all adjacency CSVs (GT + 12 AI models) into a unified
   directory tree under DESKTOP/fcm_adjacency_organised/
2. Computes per-FCM graph metrics:
     - n_nodes, n_edges
     - edge_density         = |nonzero edges| / (n*(n-1))
     - pos_edge_density     = |positive edges| / (n*(n-1))
     - neg_edge_density     = |negative edges| / (n*(n-1))
     - max_total_degree     = max node (out_degree + in_degree)
     - mean_total_degree
3. Aggregates per model × dataset and prints summary statistics with
   comparisons vs GT using Mann-Whitney U tests.
4. Saves full per-FCM CSV and summary CSV to DESKTOP/fcm_paired_analysis/
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DESKTOP  = Path(r'C:\Users\Nbrug\Desktop')
ORG_DIR  = DESKTOP / 'fcm_adjacency_organised'   # unified adjacency store
OUT_DIR  = DESKTOP / 'fcm_paired_analysis'
OUT_DIR.mkdir(exist_ok=True)

# Dataset name mappings (AI folder name → canonical key → GT folder name)
DATASETS = {
    'biodiversity': {'ai_folder': 'Biodiversity', 'gt_folder': 'biodiversity_gt'},
    'flpp':         {'ai_folder': 'FLPP',          'gt_folder': 'flpp_gt'},
    'osw':          {'ai_folder': 'Gulf OSW',       'gt_folder': 'osw_gt'},
    'red_snapper':  {'ai_folder': 'Red snapper',    'gt_folder': 'red_snapper_gt'},
}

# Dataset-specific GT subdirectory (some have an extra level)
GT_SUBDIR = {
    'biodiversity': 'biodiversity-data',
    'flpp':         'flpp-data',
    'osw':          None,
    'red_snapper':  None,
}

# OSW name normalisation (AI folder → GT stem)
OSW_NAME_MAP = {'DougP': 'DoP', 'MarianaS': 'MaS', 'MichelleS': 'MiS', 'NREL2': 'NREL'}

# All 12 AI models: (method_key, label, ai_batch_dir)
AI_MODELS = [
    ('gpt5mini',       'GPT-5-mini (E)',         DESKTOP / 'fcm_ai_20260225'),
    ('gpt5mini_nr',    'GPT-5-mini (NE)',         DESKTOP / 'fcm_ai_20260227_160145'),
    ('gpt52',          'GPT-5.2 (E)',             DESKTOP / 'fcm_adjacency_matrices_fcm_interviews_gpt52_20260227_174543'),
    ('gpt52_nr',       'GPT-5.2 (NE)',            DESKTOP / 'no reasoning results' / 'fcm_adjacency_matrices_fcm_interviews_gpt52_nr_20260303_192229'),
    ('gemini25flash',  'Gemini 2.5 Flash (E)',    DESKTOP / 'fcm_adjacency_matrices_fcm_interviews_gemini25flash_20260301_182916'),
    ('gemini25flash_nr','Gemini 2.5 Flash (NE)',  DESKTOP / 'no reasoning results' / 'fcm_adjacency_matrices_fcm_interviews_gemini25flash_nr_20260303_192146'),
    ('gemini3flash',   'Gemini 3 Flash (E)',      DESKTOP / 'fcm_adjacency_matrices_fcm_interviews_gemini3flash_20260301_182933'),
    ('gemini3flash_nr','Gemini 3 Flash (NE)',     DESKTOP / 'no reasoning results' / 'fcm_adjacency_matrices_fcm_interviews_gemini3flash_nr_20260303_192210'),
    ('qwen',           'Qwen (E)',                DESKTOP / 'fcm_adjacency_matrices_fcm_interviews_20260306_231820_qwen'),
    ('qwen_nr',        'Qwen (NE)',               DESKTOP / 'qwen3_14b_nr_20260310_194137' / 'fcm_adjacency_matrices_fcm_interviews_qwen3_14b_nr_20260310_194137'),
    ('mistral',        'Mistral (E)',             DESKTOP / 'mistral'),
    ('mistral_nr',     'Mistral (NE)',            DESKTOP / 'mistral_24b_nr'),
]

MODEL_ORDER = [m[0] for m in AI_MODELS]
MODEL_LABEL = {m[0]: m[1] for m in AI_MODELS}

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_adj(path: Path) -> np.ndarray | None:
    """
    Read adjacency CSV → numpy float array.
    Matches score_fcms.py convention:
      - numeric values kept as-is
      - '0' string → 0.0
      - any other text (e.g. 'Neutral') → 1.0  (positive edge, per scoring code)
    Returns full (possibly non-square) matrix.
    """
    try:
        mat = pd.read_csv(path, index_col=0, header=0)
        # Drop unnamed trailing columns
        mat = mat[[c for c in mat.columns if not str(c).startswith('Unnamed:')]]

        def _to_num(v):
            if v == 0 or v == '0':
                return 0.0
            try:
                f = float(v)
                return f
            except (ValueError, TypeError):
                sv = str(v).strip()
                if sv == '' or sv.lower() == 'nan':
                    return 0.0
                # Non-numeric non-zero text → treat as +1 (matches scoring code)
                return 1.0

        return mat.map(_to_num).values.astype(float)
    except Exception as e:
        print(f'  WARN read_adj failed: {path.name} — {e}')
        return None

def graph_metrics(mat: np.ndarray) -> dict:
    """
    Compute graph metrics from (possibly non-square) adjacency matrix.
    Rows = source nodes, columns = target nodes.
    n_nodes = max(n_rows, n_cols) — the full node set.
    """
    nr, nc = mat.shape
    n_nodes = max(nr, nc)
    if n_nodes < 2:
        return {}
    max_edges = n_nodes * (n_nodes - 1)   # directed, no self-loops

    # Zero out diagonal cells (self-loops) for square subblock
    sq = min(nr, nc)
    mat_clean = mat.copy()
    for i in range(sq):
        mat_clean[i, i] = 0.0

    nz  = (mat_clean != 0) & ~np.isnan(mat_clean)
    pos = mat_clean > 0
    neg = mat_clean < 0

    n_edges = int(nz.sum())
    n_pos   = int(pos.sum())
    n_neg   = int(neg.sum())

    # Out-degree = row sums (over nc columns), In-degree = col sums (over nr rows)
    out_deg = nz.sum(axis=1)          # shape (nr,)
    in_deg  = nz.sum(axis=0)          # shape (nc,)

    # Build full total-degree array over n_nodes
    total_deg = np.zeros(n_nodes)
    total_deg[:nr] += out_deg
    total_deg[:nc] += in_deg

    return {
        'n_nodes':          n_nodes,
        'n_edges':          n_edges,
        'edge_density':     n_edges  / max_edges if max_edges > 0 else np.nan,
        'pos_edge_density': n_pos    / max_edges if max_edges > 0 else np.nan,
        'neg_edge_density': n_neg    / max_edges if max_edges > 0 else np.nan,
        'max_total_degree': int(total_deg.max()),
        'mean_total_degree': float(total_deg.mean()),
    }

def extract_file_id(folder_name: str, dataset: str) -> str:
    """Normalise participant folder name → GT file stem."""
    if dataset == 'red_snapper':
        parts = folder_name.split('_')
        return parts[1] if parts[0].isdigit() else parts[0]
    elif dataset == 'osw':
        return OSW_NAME_MAP.get(folder_name, folder_name)
    return folder_name

# ── 1. Collect GT metrics ─────────────────────────────────────────────────────
print('=' * 70)
print('Computing GT graph metrics...')
gt_rows = []

for ds_key, ds_info in DATASETS.items():
    gt_dir = DESKTOP / 'fcm_gt' / ds_info['gt_folder']
    subdir = GT_SUBDIR[ds_key]
    csv_dir = gt_dir / subdir if subdir else gt_dir

    for csv_path in sorted(csv_dir.glob('*.csv')):
        mat = read_adj(csv_path)
        if mat is None:
            continue
        m = graph_metrics(mat)
        m.update({'method': 'gt', 'label': 'Ground Truth',
                  'dataset': ds_key, 'file_id': csv_path.stem})
        gt_rows.append(m)

gt_df = pd.DataFrame(gt_rows)
print(f'  GT: {len(gt_df)} FCMs across datasets: {dict(gt_df.groupby("dataset").size())}')

# ── 2. Organise AI adjacency files + collect AI metrics ──────────────────────
ORG_DIR.mkdir(exist_ok=True)
print('\nProcessing AI models...')

ai_rows = []

for method_key, label, ai_base in AI_MODELS:
    if not ai_base.exists():
        print(f'  MISSING: {label} — {ai_base}')
        continue

    model_org = ORG_DIR / method_key
    model_org.mkdir(exist_ok=True)

    total = 0
    for ds_key, ds_info in DATASETS.items():
        ai_ds_dir = ai_base / ds_info['ai_folder']
        if not ai_ds_dir.exists():
            continue

        ds_org = model_org / ds_key
        ds_org.mkdir(exist_ok=True)

        for participant_dir in sorted(ai_ds_dir.iterdir()):
            if not participant_dir.is_dir():
                continue
            csvs = list(participant_dir.glob('*.csv'))
            if not csvs:
                continue

            file_id = extract_file_id(participant_dir.name, ds_key)
            src_csv = csvs[0]

            # Copy to organised structure
            dest = ds_org / f'{file_id}.csv'
            shutil.copy2(src_csv, dest)

            mat = read_adj(src_csv)
            if mat is None:
                continue
            m = graph_metrics(mat)
            m.update({'method': method_key, 'label': label,
                      'dataset': ds_key, 'file_id': file_id})
            ai_rows.append(m)
            total += 1

    print(f'  {label:<28}: {total} FCMs copied & measured')

# Also copy GT into organised dir
gt_org = ORG_DIR / 'gt'
gt_org.mkdir(exist_ok=True)
for ds_key, ds_info in DATASETS.items():
    gt_dir = DESKTOP / 'fcm_gt' / ds_info['gt_folder']
    subdir = GT_SUBDIR[ds_key]
    csv_dir = gt_dir / subdir if subdir else gt_dir
    ds_org = gt_org / ds_key
    ds_org.mkdir(exist_ok=True)
    for f in csv_dir.glob('*.csv'):
        shutil.copy2(f, ds_org / f.name)
print(f'  {"Ground Truth":<28}: {len(gt_df)} FCMs copied')

# ── 3. Combine and save ───────────────────────────────────────────────────────
all_df = pd.concat([gt_df, pd.DataFrame(ai_rows)], ignore_index=True)
out_csv = OUT_DIR / 'fcm_graph_metrics_all.csv'
all_df.to_csv(out_csv, index=False)
print(f'\nFull metrics CSV saved → {out_csv} ({len(all_df)} rows)')

# ── 4. Summary statistics ─────────────────────────────────────────────────────
METRICS = ['edge_density', 'pos_edge_density', 'neg_edge_density',
           'max_total_degree', 'n_nodes', 'n_edges']
METRIC_LABELS = {
    'edge_density':      'Edge Density',
    'pos_edge_density':  '+Edge Density',
    'neg_edge_density':  '−Edge Density',
    'max_total_degree':  'Max Degree',
    'n_nodes':           'Nodes',
    'n_edges':           'Edges',
}

GROUPS = [('gt', 'Ground Truth')] + [(k, v) for k, v in MODEL_LABEL.items()]
DATASETS_LIST = ['biodiversity', 'flpp', 'osw', 'red_snapper', 'ALL']
DS_LABEL = {'biodiversity': 'Biodiv.', 'flpp': 'FLPP',
            'osw': 'OSW', 'red_snapper': 'RedSnap', 'ALL': 'ALL'}

def get_vals(df, method, dataset, metric):
    rows = df[df['method'] == method] if dataset == 'ALL' else \
           df[(df['method'] == method) & (df['dataset'] == dataset)]
    return rows[metric].dropna().values

summary_rows = []

for metric in METRICS:
    for m_key, m_label in GROUPS:
        for ds in DATASETS_LIST:
            v = get_vals(all_df, m_key, ds, metric)
            if len(v) == 0:
                continue
            summary_rows.append({
                'metric': metric, 'method': m_key, 'label': m_label,
                'dataset': ds, 'mean': round(v.mean(), 4), 'sd': round(v.std(), 4), 'n': len(v),
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / 'fcm_graph_metrics_summary.csv', index=False)

# ── 5. Clean publication table ────────────────────────────────────────────────
# Format: rows = model, columns = metric × dataset
# sig markers vs GT; R vs NR column-pair contrasts appended

MODEL_PAIRS_KEYS = [
    ('gpt5mini',       'gpt5mini_nr',      'GPT-5-mini'),
    ('gpt52',          'gpt52_nr',         'GPT-5.2'),
    ('gemini25flash',  'gemini25flash_nr',  'Gemini 2.5 Flash'),
    ('gemini3flash',   'gemini3flash_nr',   'Gemini 3 Flash'),
    ('qwen',           'qwen_nr',           'Qwen'),
    ('mistral',        'mistral_nr',        'Mistral'),
]

DS_SHORT = {'biodiversity': 'Bio', 'flpp': 'FLPP', 'osw': 'OSW',
            'red_snapper': 'RS',   'ALL': 'ALL'}
METRIC_SHORT = {
    'edge_density':     'EdgeDens',
    'pos_edge_density': '+EdgeDens',
    'neg_edge_density': '−EdgeDens',
    'max_total_degree': 'MaxDeg',
    'n_nodes':          'Nodes',
    'n_edges':          'Edges',
}

def sig_stars(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '.'
    return ''

table_rows = []

for ds in DATASETS_LIST:
    gt_v_all = {m: get_vals(all_df, 'gt', ds, m) for m in METRICS}

    # GT row first
    row = {'Dataset': DS_SHORT[ds], 'Model': 'GT', 'Condition': '—'}
    for metric in METRICS:
        v = gt_v_all[metric]
        row[METRIC_SHORT[metric]] = f'{v.mean():.3f}' if len(v) else '—'
        row[METRIC_SHORT[metric] + '_SD'] = f'{v.std():.3f}' if len(v) else '—'
    table_rows.append(row)

    for r_key, nr_key, label in MODEL_PAIRS_KEYS:
        for cond, key in [('R', r_key), ('NR', nr_key)]:
            row = {'Dataset': DS_SHORT[ds], 'Model': label, 'Condition': cond}
            for metric in METRICS:
                v      = get_vals(all_df, key, ds, metric)
                gt_v   = gt_v_all[metric]
                r_v    = get_vals(all_df, r_key,  ds, metric)
                nr_v   = get_vals(all_df, nr_key, ds, metric)

                if len(v) == 0:
                    row[METRIC_SHORT[metric]]        = '—'
                    row[METRIC_SHORT[metric] + '_SD'] = '—'
                    row[METRIC_SHORT[metric] + '_vs_GT'] = ''
                    row[METRIC_SHORT[metric] + '_R_vs_NR'] = ''
                    continue

                mean_, sd_ = v.mean(), v.std()
                row[METRIC_SHORT[metric]]        = f'{mean_:.3f}'
                row[METRIC_SHORT[metric] + '_SD'] = f'{sd_:.3f}'

                # vs GT
                if len(gt_v) >= 3:
                    _, p_gt = stats.mannwhitneyu(v, gt_v, alternative='two-sided')
                    row[METRIC_SHORT[metric] + '_vs_GT'] = sig_stars(p_gt)
                else:
                    row[METRIC_SHORT[metric] + '_vs_GT'] = ''

                # R vs NR (same for both rows — show on NR row only)
                if cond == 'NR' and len(r_v) >= 3 and len(nr_v) >= 3:
                    _, p_rnr = stats.mannwhitneyu(r_v, nr_v, alternative='two-sided')
                    row[METRIC_SHORT[metric] + '_R_vs_NR'] = sig_stars(p_rnr)
                else:
                    row[METRIC_SHORT[metric] + '_R_vs_NR'] = ''

            table_rows.append(row)

table_df = pd.DataFrame(table_rows)
table_csv = OUT_DIR / 'fcm_graph_metrics_table.csv'
table_df.to_csv(table_csv, index=False)

# ── 6. Print clean table ──────────────────────────────────────────────────────
PRINT_METRICS = ['edge_density', 'pos_edge_density', 'neg_edge_density', 'max_total_degree']
COL_W = 18

def fmt_cell(mean_str, sd_str, gt_sig, rnr_sig):
    if mean_str == '—':
        return '—'.center(COL_W)
    core = f'{mean_str}±{sd_str}'
    markers = (gt_sig or '') + ('/' + rnr_sig if rnr_sig else '')
    return f'{core}{markers}'.ljust(COL_W)

print()
print('=' * 120)
print('GRAPH METRICS SUMMARY TABLE')
print('sig vs GT: * p<.05  ** p<.01  *** p<.001  . p<.10')
print('R vs NR (after /): shown on NR row  e.g. 0.047±0.026*/.  means sig vs GT + trending R≠NR')
print('=' * 120)

for ds in DATASETS_LIST:
    ds_rows = [r for r in table_rows if r['Dataset'] == DS_SHORT[ds]]
    print()
    print(f'  ── {DS_SHORT[ds]} ──')
    hdr = f'  {"Model":<22} {"Cond":<4}' + ''.join(f'  {METRIC_SHORT[m]:<{COL_W}}' for m in PRINT_METRICS)
    print(hdr)
    print('  ' + '-' * (26 + (COL_W + 2) * len(PRINT_METRICS)))

    for row in ds_rows:
        model_str = f'{row["Model"]:<22}'
        cond_str  = f'{row["Condition"]:<4}'
        cells = []
        for m in PRINT_METRICS:
            ms = METRIC_SHORT[m]
            mean_ = row.get(ms, '—')
            sd_   = row.get(ms + '_SD', '—')
            gt_s  = row.get(ms + '_vs_GT', '')
            rnr_s = row.get(ms + '_R_vs_NR', '')
            cells.append('  ' + fmt_cell(mean_, sd_, gt_s, rnr_s))
        print(f'  {model_str} {cond_str}' + ''.join(cells))

print()
print(f'Full table CSV → {table_csv}')
print(f'Summary CSV    → {OUT_DIR / "fcm_graph_metrics_summary.csv"}')
print(f'Per-FCM CSV    → {OUT_DIR / "fcm_graph_metrics_all.csv"}')
print(f'Adjacency tree → {ORG_DIR}')

# ── 7. GT-contrast table (AI vs GT only, no R-vs-NR mixing) ──────────────────
ALL_METRICS = ['edge_density', 'pos_edge_density', 'neg_edge_density',
               'max_total_degree', 'n_nodes', 'n_edges']
METRIC_LABEL = {
    'edge_density':     'EdgeDens',
    'pos_edge_density': '+EdgeDens',
    'neg_edge_density': '−EdgeDens',
    'max_total_degree': 'MaxDeg',
    'n_nodes':          'Nodes',
    'n_edges':          'Edges',
}

def fmt_gt(mean_str, sd_str, gt_sig):
    """Format a cell as mean±SD + GT-significance only."""
    if mean_str == '—' or mean_str is None:
        return '—'
    sig = gt_sig if gt_sig and gt_sig != 'ns' else ''
    return f'{mean_str}±{sd_str}{sig}'

gt_rows_wide = []
model_rows_wide = []

for ds in DATASETS_LIST:
    ds_label = DS_SHORT[ds]

    # GT baseline row for this dataset
    gt_row = {'Dataset': ds_label, 'Model': 'GT', 'Condition': '—'}
    for metric in ALL_METRICS:
        ml = METRIC_LABEL[metric]
        gt_v = get_vals(all_df, 'gt', ds, metric)
        if len(gt_v) >= 1:
            gt_row[ml] = f'{gt_v.mean():.3f}±{gt_v.std():.3f}'
        else:
            gt_row[ml] = '—'
    gt_rows_wide.append(gt_row)

    for r_key, nr_key, label in MODEL_PAIRS_KEYS:
        gt_v = get_vals(all_df, 'gt', ds, 'edge_density')  # just to reuse variable
        for cond, key in [('R', r_key), ('NR', nr_key)]:
            row = {'Dataset': ds_label, 'Model': label, 'Condition': cond}
            for metric in ALL_METRICS:
                ml = METRIC_LABEL[metric]
                v       = get_vals(all_df, key,  ds, metric)
                gt_vals = get_vals(all_df, 'gt', ds, metric)
                if len(v) < 1:
                    row[ml] = '—'
                    continue
                mean_s = f'{v.mean():.3f}'
                sd_s   = f'{v.std():.3f}'
                if len(gt_vals) >= 3 and len(v) >= 3:
                    _, p_gt = stats.mannwhitneyu(v, gt_vals, alternative='two-sided')
                    sig = sig_stars(p_gt)
                else:
                    sig = ''
                row[ml] = fmt_gt(mean_s, sd_s, sig)
            model_rows_wide.append(row)

contrast_rows = []
for ds in DATASETS_LIST:
    ds_label = DS_SHORT[ds]
    contrast_rows += [r for r in gt_rows_wide   if r['Dataset'] == ds_label]
    contrast_rows += [r for r in model_rows_wide if r['Dataset'] == ds_label]

contrast_df = pd.DataFrame(contrast_rows,
                            columns=['Dataset', 'Model', 'Condition'] +
                                    [METRIC_LABEL[m] for m in ALL_METRICS])
contrast_csv = OUT_DIR / 'fcm_graph_metrics_gt_contrasts.csv'
contrast_df.to_csv(contrast_csv, index=False)

# pretty-print
GT_COL_W = 18
print()
print('=' * 130)
print('GT-CONTRAST TABLE  (mean±SD  sig vs GT:  *** p<.001  ** p<.01  * p<.05  . p<.10)')
print('=' * 130)
for ds in DATASETS_LIST:
    ds_label = DS_SHORT[ds]
    rows_ds = [r for r in contrast_rows if r['Dataset'] == ds_label]
    print()
    print(f'  ── {ds_label} ──')
    hdr = f'  {"Model":<22} {"Cond":<5}' + \
          ''.join(f'  {METRIC_LABEL[m]:<{GT_COL_W}}' for m in ALL_METRICS)
    print(hdr)
    print('  ' + '-' * (27 + (GT_COL_W + 2) * len(ALL_METRICS)))
    for row in rows_ds:
        cells = ''.join(f'  {str(row[METRIC_LABEL[m]]):<{GT_COL_W}}' for m in ALL_METRICS)
        print(f'  {row["Model"]:<22} {row["Condition"]:<5}{cells}')

print()
print(f'GT-contrast CSV → {contrast_csv}')
print('\nDone.')

