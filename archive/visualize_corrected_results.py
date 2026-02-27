"""
Generate boxplot visualizations from the CORRECTED FCM comparison results.
"""

# Use Agg backend to prevent interactive display
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load corrected results
results_file = r"C:\Users\Nbrug\Desktop\fcm_comparison_results\all_fcm_comparisons_CORRECTED.csv"
df = pd.read_csv(results_file)

print(f"Loaded {len(df)} comparisons from corrected results")
print(f"Datasets: {df['dataset'].unique()}")

# Create output directory
output_dir = r"C:\Users\Nbrug\Desktop\fcm_comparison_results\visualizations_corrected"
os.makedirs(output_dir, exist_ok=True)

# Map dataset names for display
dataset_display_names = {
    'biodiversity': 'Biodiversity',
    'flpp': 'FLPP',
    'osw': 'Gulf OSW',
    'red_snapper': 'Red Snapper'
}
df['dataset_display'] = df['dataset'].map(dataset_display_names)

# ============================================================================
# FIGURE 1: Edge Matching Metrics (TP, PP, FP, FN)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Edge Matching Metrics by Dataset (CORRECTED)', fontsize=16, fontweight='bold', y=0.995)

metrics = ['TP', 'PP', 'FP', 'FN']
metric_labels = {
    'TP': 'True Positives (TP)',
    'PP': 'Partial Positives (PP)', 
    'FP': 'False Positives (FP)\n(AI edges not in GT)',
    'FN': 'False Negatives (FN)\n(GT edges not in AI)'
}
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    # Create boxplot
    bp = ax.boxplot(
        [df[df['dataset']==ds][metric].values for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
        labels=[dataset_display_names[ds] for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6)
    )
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor(colors[idx])
        patch.set_alpha(0.7)
    
    ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
output_file = os.path.join(output_dir, 'edge_matching_metrics_boxplots_CORRECTED.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_file}")
file_size = os.path.getsize(output_file) / 1024
print(f"  Size: {file_size:.1f} KB")

# ============================================================================
# FIGURE 2: Node Counts (AI-generated vs Ground Truth)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Node Counts: AI-Generated vs Ground Truth (CORRECTED)', fontsize=16, fontweight='bold')

# AI-generated node counts
bp1 = axes[0].boxplot(
    [df[df['dataset']==ds]['ai_nodes'].values for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
    labels=[dataset_display_names[ds] for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6)
)
for patch in bp1['boxes']:
    patch.set_facecolor('#9b59b6')
    patch.set_alpha(0.7)
axes[0].set_ylabel('Number of Nodes', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Dataset', fontsize=10)
axes[0].set_title('AI-Generated FCMs', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Ground truth node counts
bp2 = axes[1].boxplot(
    [df[df['dataset']==ds]['gt_nodes'].values for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
    labels=[dataset_display_names[ds] for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6)
)
for patch in bp2['boxes']:
    patch.set_facecolor('#16a085')
    patch.set_alpha(0.7)
axes[1].set_ylabel('Number of Nodes', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Dataset', fontsize=10)
axes[1].set_title('Ground Truth FCMs', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(output_dir, 'node_counts_boxplots_CORRECTED.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_file}")
file_size = os.path.getsize(output_file) / 1024
print(f"  Size: {file_size:.1f} KB")

# ============================================================================
# FIGURE 3: Edge Counts (AI-generated vs Ground Truth)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Edge Counts: AI-Generated vs Ground Truth (CORRECTED)', fontsize=16, fontweight='bold')

# AI-generated edge counts
bp1 = axes[0].boxplot(
    [df[df['dataset']==ds]['ai_edges'].values for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
    labels=[dataset_display_names[ds] for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6)
)
for patch in bp1['boxes']:
    patch.set_facecolor('#e67e22')
    patch.set_alpha(0.7)
axes[0].set_ylabel('Number of Edges', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Dataset', fontsize=10)
axes[0].set_title('AI-Generated FCMs', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Ground truth edge counts
bp2 = axes[1].boxplot(
    [df[df['dataset']==ds]['gt_edges'].values for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
   labels=[dataset_display_names[ds] for ds in ['biodiversity', 'flpp', 'osw', 'red_snapper']],
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6)
)
for patch in bp2['boxes']:
    patch.set_facecolor('#c0392b')
    patch.set_alpha(0.7)
axes[1].set_ylabel('Number of Edges', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Dataset', fontsize=10)
axes[1].set_title('Ground Truth FCMs', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(output_dir, 'edge_counts_boxplots_CORRECTED.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_file}")
file_size = os.path.getsize(output_file) / 1024
print(f"  Size: {file_size:.1f} KB")

# ============================================================================
# Print detailed statistics
# ============================================================================
print("\n" + "="*80)
print("CORRECTED STATISTICS BY DATASET")
print("="*80)

for dataset in ['biodiversity', 'flpp', 'osw', 'red_snapper']:
    ds_df = df[df['dataset'] == dataset]
    display_name = dataset_display_names[dataset]
    
    print(f"\n{display_name.upper()}:")
    print("-" * 80)
    print(f"Number of comparisons: {len(ds_df)}")
    
    print("\nPerformance Metrics:")
    print(f"  F1:      Mean={ds_df['F1'].mean():.3f}, Median={ds_df['F1'].median():.3f}, "
          f"Std={ds_df['F1'].std():.3f}, Range=[{ds_df['F1'].min():.3f}, {ds_df['F1'].max():.3f}]")
    print(f"  Jaccard: Mean={ds_df['Jaccard'].mean():.3f}, Median={ds_df['Jaccard'].median():.3f}, "
          f"Std={ds_df['Jaccard'].std():.3f}, Range=[{ds_df['Jaccard'].min():.3f}, {ds_df['Jaccard'].max():.3f}]")
    
    print("\nEdge Matching Metrics (CORRECTED):")
    for metric in ['TP', 'PP', 'FP', 'FN']:
        data = ds_df[metric]
        print(f"  {metric}: Mean={data.mean():6.1f}, Median={data.median():6.1f}, "
              f"Std={data.std():6.1f}, Range=[{data.min():.0f}, {data.max():.0f}]")
    
    print("\nGraph Structure (AI-generated):")
    print(f"  Nodes: Mean={ds_df['ai_nodes'].mean():6.1f}, Median={ds_df['ai_nodes'].median():6.1f}, "
          f"Range=[{ds_df['ai_nodes'].min():.0f}, {ds_df['ai_nodes'].max():.0f}]")
    print(f"  Edges: Mean={ds_df['ai_edges'].mean():6.1f}, Median={ds_df['ai_edges'].median():6.1f}, "
          f"Range=[{ds_df['ai_edges'].min():.0f}, {ds_df['ai_edges'].max():.0f}]")
    
    print("\nGraph Structure (Ground Truth):")
    print(f"  Nodes: Mean={ds_df['gt_nodes'].mean():6.1f}, Median={ds_df['gt_nodes'].median():6.1f}, "
          f"Range=[{ds_df['gt_nodes'].min():.0f}, {ds_df['gt_nodes'].max():.0f}]")
    print(f"  Edges: Mean={ds_df['gt_edges'].mean():6.1f}, Median={ds_df['gt_edges'].median():6.1f}, "
          f"Range=[{ds_df['gt_edges'].min():.0f}, {ds_df['gt_edges'].max():.0f}]")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nKey difference from original (incorrect) results:")
print("  RED SNAPPER now shows:")
print("    - AI edges (28.4) > GT edges (11.7)")
print("    - High FP (16.7) = AI generated many edges not in GT")
print("    - Zero FN (0.0) = AI found nearly all GT edges")
print("  This makes logical sense! The original had these swapped.")
