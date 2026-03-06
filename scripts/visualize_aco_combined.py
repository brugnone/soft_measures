"""
Compare all LLM-interview methods vs LLM+ACO methods on Red Snapper.

  Interview-based (5 batches):
    GPT-5-mini + explanation, GPT-5-mini, GPT-5.2 + explanation,
    Gemini 2.5 Flash, Gemini 3 Flash

  LLM + ACO (6 models):
    GPT-4o, GPT-4.1 Mini, GPT-4.1, GPT-5.2,
    Gemini 2.5 Flash, Gemini 3 Flash

Metrics: F1, TP, PP, FP, FN

Output: Desktop/fcm_visualizations_all5/aco_vs_interview_red_snapper.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
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

# ── Interview-based batches (Red Snapper subset) ───────────────────────────────
INTERVIEW_SOURCES = [
    (
        DESKTOP / 'fcm_comparison_results_CORRECT' /
        'all_fcm_comparisons_CORRECT.csv',
        'GPT-5-mini\n+explanation',
    ),
    (
        DESKTOP / 'fcm_comparison_results_20260227' /
        'all_fcm_comparisons_20260227.csv',
        'GPT-5-mini',
    ),
    (
        DESKTOP / 'fcm_comparison_results_gpt52_20260227' /
        'all_fcm_comparisons_fcm_adjacency_matrices_fcm_interviews_gpt52_20260227_174543.csv',
        'GPT-5.2\n+explanation',
    ),
    (
        DESKTOP / 'fcm_comparison_results_fcm_adjacency_matrices_fcm_interviews_gemini25flash_20260301_182916' /
        'all_fcm_comparisons_fcm_adjacency_matrices_fcm_interviews_gemini25flash_20260301_182916.csv',
        'Gemini 2.5 Flash\n(interview)',
    ),
    (
        DESKTOP / 'fcm_comparison_results_fcm_adjacency_matrices_fcm_interviews_gemini3flash_20260301_182933' /
        'all_fcm_comparisons_fcm_adjacency_matrices_fcm_interviews_gemini3flash_20260301_182933.csv',
        'Gemini 3 Flash\n(interview)',
    ),
]

ACO_DIR = DESKTOP / 'llm_plus_ant_colony_optimization_results'
ACO_SOURCES = [
    (ACO_DIR / 'gpt-4o.csv',        'GPT-4o\n+ACO'),
    (ACO_DIR / 'gpt-4.1-mini.csv',  'GPT-4.1 Mini\n+ACO'),
    (ACO_DIR / 'gpt-4.1.csv',       'GPT-4.1\n+ACO'),
    (ACO_DIR / 'gpt-5.2.csv',       'GPT-5.2\n+ACO'),
    (ACO_DIR / 'gemini-2.5-flash.csv', 'Gemini 2.5 Flash\n+ACO'),
    (ACO_DIR / 'gemini-3-flash.csv',   'Gemini 3 Flash\n+ACO'),
]

# Colors: muted blues/greens for interview, warm palette for ACO
INTERVIEW_COLORS = ['#2E86AB', '#E84855', '#3BB273', '#F4A261', '#E040FB']
ACO_COLORS       = ['#B5838D', '#6D6875', '#E9C46A', '#457B9D', '#A8DADC', '#F77F00']

# ── Load data ──────────────────────────────────────────────────────────────────
records = []   # list of (label, group, df)

for (path, label), color in zip(INTERVIEW_SOURCES, INTERVIEW_COLORS):
    df = pd.read_csv(path)
    df = df[df['dataset'] == 'red_snapper'].copy()
    records.append({'label': label, 'group': 'LLM Interview', 'color': color, 'df': df})
    print(f'  [interview] {label.replace(chr(10)," ")}: {len(df)} rows')

for (path, label), color in zip(ACO_SOURCES, ACO_COLORS):
    df = pd.read_csv(path)
    # Normalise column names to match interview format
    df = df.rename(columns={'fcm1_edges': 'gt_edges', 'fcm2_edges': 'ai_edges',
                             'fcm1_nodes': 'gt_nodes', 'fcm2_nodes': 'ai_nodes'})
    records.append({'label': label, 'group': 'LLM + ACO', 'color': color, 'df': df})
    print(f'  [aco]       {label.replace(chr(10)," ")}: {len(df)} rows')

n_interview = len(INTERVIEW_SOURCES)
n_aco       = len(ACO_SOURCES)
n_total     = n_interview + n_aco

labels = [r['label'] for r in records]
colors = [r['color'] for r in records]
groups = [r['group'] for r in records]

print(f'\nTotal methods: {n_total}  (interview: {n_interview}, ACO: {n_aco})')

# ── Helper: draw one metric panel ─────────────────────────────────────────────
def draw_metric(ax, metric, ylabel, records, divider_after):
    """Boxplot panel: x = method, y = metric value."""
    positions = np.arange(1, n_total + 1)

    for i, rec in enumerate(records):
        vals = rec['df'][metric].dropna().values
        bp = ax.boxplot(
            vals,
            positions=[positions[i]],
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=2.5, alpha=0.5,
                            markerfacecolor=rec['color'], markeredgewidth=0),
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor=rec['color'], alpha=0.8),
            whiskerprops=dict(color=rec['color']),
            capprops=dict(color=rec['color']),
        )

    # Divider between interview and ACO groups
    ax.axvline(divider_after + 0.5, color='#888888', linewidth=1.0,
               linestyle='--', alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.5, ha='center')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_xlim(0.3, n_total + 0.7)


# ── Figure layout: F1 on top row (full width), TP/PP/FP/FN on bottom ──────────
fig = plt.figure(figsize=(16, 9))
gs  = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.35,
                        top=0.91, bottom=0.22, left=0.06, right=0.98)

ax_f1 = fig.add_subplot(gs[0, :2])   # F1 spans left half of top row
ax_jc = fig.add_subplot(gs[0, 2:])   # Jaccard spans right half of top row
ax_tp = fig.add_subplot(gs[1, 0])
ax_pp = fig.add_subplot(gs[1, 1])
ax_fp = fig.add_subplot(gs[1, 2])
ax_fn = fig.add_subplot(gs[1, 3])

for ax, metric, ylabel in [
    (ax_f1, 'F1',      'F1 Score'),
    (ax_jc, 'Jaccard', 'Jaccard Similarity'),
    (ax_tp, 'TP',      'True Positives (TP)'),
    (ax_pp, 'PP',      'Partial Positives (PP)'),
    (ax_fp, 'FP',      'False Positives (FP)'),
    (ax_fn, 'FN',      'False Negatives (FN)'),
]:
    draw_metric(ax, metric, ylabel, records, divider_after=n_interview)

# ── Legend: group labels only ──────────────────────────────────────────────────
interview_patch = mpatches.Patch(color='#888888', alpha=0, label='—— LLM Interview')
aco_patch        = mpatches.Patch(color='#888888', alpha=0, label='—— LLM + ACO')

interview_handles = [mpatches.Patch(color=c, alpha=0.8, label=l.replace('\n', ' '))
                     for c, l in zip(INTERVIEW_COLORS, [r['label'] for r in records[:n_interview]])]
aco_handles       = [mpatches.Patch(color=c, alpha=0.8, label=l.replace('\n', ' '))
                     for c, l in zip(ACO_COLORS, [r['label'] for r in records[n_interview:]])]

leg_interview = fig.legend(
    handles=interview_handles,
    title='LLM Interview',
    title_fontsize=9,
    loc='lower left',
    ncol=1,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.01, 0.01),
    borderpad=0.6,
)
leg_interview.get_title().set_fontweight('bold')

leg_aco = fig.legend(
    handles=aco_handles,
    title='LLM + ACO',
    title_fontsize=9,
    loc='lower right',
    ncol=1,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.99, 0.01),
    borderpad=0.6,
)
leg_aco.get_title().set_fontweight('bold')
fig.add_artist(leg_interview)

fig.suptitle('Red Snapper: LLM Interview vs LLM + ACO',
             fontsize=13, fontweight='bold')

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = DESKTOP / 'fcm_visualizations_all5'
out_dir.mkdir(exist_ok=True)
out_path = out_dir / 'aco_vs_interview_red_snapper.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
kb = out_path.stat().st_size / 1024
print(f'\n[OK] {out_path}  ({kb:.1f} KB)')

# ── Summary table ──────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('MEAN F1 BY METHOD  (Red Snapper)')
print('=' * 65)
for rec in records:
    d = rec['df']
    label = rec['label'].replace('\n', ' ')
    group = rec['group']
    f1 = d['F1']
    print(f"  [{group}]  {label:<35} F1={f1.mean():.3f} +/- {f1.std():.3f}  (n={len(f1)})")
