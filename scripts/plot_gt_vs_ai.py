"""
Generate scatter plots comparing Ground Truth vs AI-generated FCM sizes.
Shows node counts and edge counts with reference diagonal line.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Publication-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3

# Load data
results_file = r"C:\Users\Nbrug\Desktop\fcm_comparison_results_CORRECT\all_fcm_comparisons_CORRECT.csv"
df = pd.read_csv(results_file)

print(f"Loaded {len(df)} comparisons from corrected results")
print(f"Datasets: {df['dataset'].value_counts().to_dict()}")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Marker shapes for datasets
dataset_markers = {
    'biodiversity': 'o',      # circle
    'flpp': 's',              # square
    'osw': '^',               # triangle
    'red_snapper': 'D'        # diamond
}

# Use viridis colormap for F1 scores (0.0 to 1.0)
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=1)

# Plot 1: Node counts
ax1 = axes[0]
for dataset in sorted(df['dataset'].unique()):
    subset = df[df['dataset'] == dataset]
    scatter1 = ax1.scatter(subset['gt_nodes'], subset['ai_nodes'], 
               c=subset['F1'], cmap=cmap, norm=norm,
               alpha=0.7, s=80, 
               marker=dataset_markers.get(dataset, 'o'),
               label=dataset.replace('_', ' ').title(),
               edgecolors='black', linewidth=0.7)

# Add diagonal reference line (y=x)
max_nodes = max(df['gt_nodes'].max(), df['ai_nodes'].max())
min_nodes = min(df['gt_nodes'].min(), df['ai_nodes'].min())
ax1.plot([min_nodes, max_nodes], [min_nodes, max_nodes], 
         'k--', alpha=0.5, linewidth=1.5, label='Perfect Agreement', zorder=0)

ax1.set_xlabel('Ground Truth Node Count', fontsize=12, fontweight='bold')
ax1.set_ylabel('AI-Generated Node Count', fontsize=12, fontweight='bold')
ax1.set_title('Node Count Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal', adjustable='box')

# Plot 2: Edge counts
ax2 = axes[1]
for dataset in sorted(df['dataset'].unique()):
    subset = df[df['dataset'] == dataset]
    scatter2 = ax2.scatter(subset['gt_edges'], subset['ai_edges'], 
               c=subset['F1'], cmap=cmap, norm=norm,
               alpha=0.7, s=80,
               marker=dataset_markers.get(dataset, 'o'),
               label=dataset.replace('_', ' ').title(),
               edgecolors='black', linewidth=0.7)

# Add diagonal reference line (y=x)
max_edges = max(df['gt_edges'].max(), df['ai_edges'].max())
min_edges = min(df['gt_edges'].min(), df['ai_edges'].min())
ax2.plot([min_edges, max_edges], [min_edges, max_edges], 
         'k--', alpha=0.5, linewidth=1.5, label='Perfect Agreement', zorder=0)

ax2.set_xlabel('Ground Truth Edge Count', fontsize=12, fontweight='bold')
ax2.set_ylabel('AI-Generated Edge Count', fontsize=12, fontweight='bold')
ax2.set_title('Edge Count Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_aspect('equal', adjustable='box')

# Add shared colorbar for F1 scores
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar.set_label('F1 Score', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Save figure
output_dir = Path(r"C:\Users\Nbrug\Desktop\fcm_comparison_results_CORRECT\visualizations")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "gt_vs_ai_scatter.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
file_size_kb = output_file.stat().st_size / 1024
print(f"\n[OK] Saved: {output_file}")
print(f"     Size: {file_size_kb:.1f} KB")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for dataset in sorted(df['dataset'].unique()):
    subset = df[df['dataset'] == dataset]
    print(f"\n{dataset.upper()}:")
    print(f"  Nodes - GT: {subset['gt_nodes'].mean():.1f} ± {subset['gt_nodes'].std():.1f}, "
          f"AI: {subset['ai_nodes'].mean():.1f} ± {subset['ai_nodes'].std():.1f}")
    print(f"  Edges - GT: {subset['gt_edges'].mean():.1f} ± {subset['gt_edges'].std():.1f}, "
          f"AI: {subset['ai_edges'].mean():.1f} ± {subset['ai_edges'].std():.1f}")
    
    # Correlation
    node_corr = subset[['gt_nodes', 'ai_nodes']].corr().iloc[0, 1]
    edge_corr = subset[['gt_edges', 'ai_edges']].corr().iloc[0, 1]
    print(f"  Correlation - Nodes: r={node_corr:.3f}, Edges: r={edge_corr:.3f}")

print("\n" + "="*60)
print("ALL DATASETS COMBINED:")
print("="*60)
print(f"  Nodes - GT: {df['gt_nodes'].mean():.1f} ± {df['gt_nodes'].std():.1f}, "
      f"AI: {df['ai_nodes'].mean():.1f} ± {df['ai_nodes'].std():.1f}")
print(f"  Edges - GT: {df['gt_edges'].mean():.1f} ± {df['gt_edges'].std():.1f}, "
      f"AI: {df['ai_edges'].mean():.1f} ± {df['ai_edges'].std():.1f}")

node_corr = df[['gt_nodes', 'ai_nodes']].corr().iloc[0, 1]
edge_corr = df[['gt_edges', 'ai_edges']].corr().iloc[0, 1]
print(f"  Correlation - Nodes: r={node_corr:.3f}, Edges: r={edge_corr:.3f}")

print("\n" + "="*60)
print("DONE!")
