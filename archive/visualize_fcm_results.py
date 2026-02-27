"""
Create visualizations of FCM scoring results:
- Boxplots of TP, PP, FP, FN by dataset
- Boxplots of node and edge counts by dataset
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load combined results
results_file = r"C:\Users\Nbrug\Desktop\fcm_comparison_results\all_fcm_comparisons_complete.csv"
df = pd.read_csv(results_file)

print(f"Loaded {len(df)} comparisons from {len(df['dataset'].unique())} datasets")
print(f"Datasets: {', '.join(df['dataset'].unique())}")

# Create output directory for plots
output_dir = r"C:\Users\Nbrug\Desktop\fcm_comparison_results\visualizations"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Figure 1: TP, PP, FP, FN Boxplots
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Edge Matching Metrics by Dataset', fontsize=16, fontweight='bold')

metrics = ['TP', 'PP', 'FP', 'FN']
metric_labels = {
    'TP': 'True Positives (TP)',
    'PP': 'Partial Positives (PP)',
    'FP': 'False Positives (FP)',
    'FN': 'False Negatives (FN)'
}
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Create boxplot
    box_parts = ax.boxplot([df[df['dataset'] == ds][metric].dropna() 
                             for ds in df['dataset'].unique()],
                            labels=df['dataset'].unique(),
                            patch_artist=True,
                            showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='red', 
                                         markersize=6, markeredgecolor='darkred'))
    
    # Color the boxes
    for patch in box_parts['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_xlabel('Dataset', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add median values as text
    for i, ds in enumerate(df['dataset'].unique()):
        median_val = df[df['dataset'] == ds][metric].median()
        ax.text(i+1, ax.get_ylim()[1]*0.95, f'Med: {median_val:.0f}', 
                ha='center', fontsize=8, bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'edge_matching_metrics_boxplots.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {plot1_path}")

# ============================================================================
# Figure 2: Node Counts (AI vs Ground Truth)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Node Counts by Dataset: AI-Generated vs Ground Truth', 
             fontsize=16, fontweight='bold')

# AI-generated FCM nodes
ax = axes[0]
box_parts = ax.boxplot([df[df['dataset'] == ds]['fcm1_nodes'].dropna() 
                         for ds in df['dataset'].unique()],
                        labels=df['dataset'].unique(),
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                     markersize=6, markeredgecolor='darkred'))
for patch in box_parts['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)

ax.set_title('AI-Generated FCMs', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Nodes', fontsize=10)
ax.set_xlabel('Dataset', fontsize=10)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# Add statistics
for i, ds in enumerate(df['dataset'].unique()):
    data = df[df['dataset'] == ds]['fcm1_nodes']
    median_val = data.median()
    mean_val = data.mean()
    ax.text(i+1, ax.get_ylim()[1]*0.95, 
            f'Med: {median_val:.0f}\nMean: {mean_val:.1f}', 
            ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ground truth FCM nodes
ax = axes[1]
box_parts = ax.boxplot([df[df['dataset'] == ds]['fcm2_nodes'].dropna() 
                         for ds in df['dataset'].unique()],
                        labels=df['dataset'].unique(),
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                     markersize=6, markeredgecolor='darkred'))
for patch in box_parts['boxes']:
    patch.set_facecolor('#2ecc71')
    patch.set_alpha(0.7)

ax.set_title('Ground Truth FCMs', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Nodes', fontsize=10)
ax.set_xlabel('Dataset', fontsize=10)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# Add statistics
for i, ds in enumerate(df['dataset'].unique()):
    data = df[df['dataset'] == ds]['fcm2_nodes']
    median_val = data.median()
    mean_val = data.mean()
    ax.text(i+1, ax.get_ylim()[1]*0.95, 
            f'Med: {median_val:.0f}\nMean: {mean_val:.1f}', 
            ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plot2_path = os.path.join(output_dir, 'node_counts_boxplots.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot2_path}")

# ============================================================================
# Figure 3: Edge Counts (AI vs Ground Truth)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Edge Counts by Dataset: AI-Generated vs Ground Truth', 
             fontsize=16, fontweight='bold')

# AI-generated FCM edges
ax = axes[0]
box_parts = ax.boxplot([df[df['dataset'] == ds]['fcm1_edges'].dropna() 
                         for ds in df['dataset'].unique()],
                        labels=df['dataset'].unique(),
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                     markersize=6, markeredgecolor='darkred'))
for patch in box_parts['boxes']:
    patch.set_facecolor('#9b59b6')
    patch.set_alpha(0.7)

ax.set_title('AI-Generated FCMs', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Edges', fontsize=10)
ax.set_xlabel('Dataset', fontsize=10)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# Add statistics
for i, ds in enumerate(df['dataset'].unique()):
    data = df[df['dataset'] == ds]['fcm1_edges']
    median_val = data.median()
    mean_val = data.mean()
    ax.text(i+1, ax.get_ylim()[1]*0.95, 
            f'Med: {median_val:.0f}\nMean: {mean_val:.1f}', 
            ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ground truth FCM edges
ax = axes[1]
box_parts = ax.boxplot([df[df['dataset'] == ds]['fcm2_edges'].dropna() 
                         for ds in df['dataset'].unique()],
                        labels=df['dataset'].unique(),
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                     markersize=6, markeredgecolor='darkred'))
for patch in box_parts['boxes']:
    patch.set_facecolor('#e67e22')
    patch.set_alpha(0.7)

ax.set_title('Ground Truth FCMs', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Edges', fontsize=10)
ax.set_xlabel('Dataset', fontsize=10)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

# Add statistics
for i, ds in enumerate(df['dataset'].unique()):
    data = df[df['dataset'] == ds]['fcm2_edges']
    median_val = data.median()
    mean_val = data.mean()
    ax.text(i+1, ax.get_ylim()[1]*0.95, 
            f'Med: {median_val:.0f}\nMean: {mean_val:.1f}', 
            ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plot3_path = os.path.join(output_dir, 'edge_counts_boxplots.png')
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot3_path}")

# ============================================================================
# Summary Statistics Table
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS BY DATASET")
print("=" * 80)

for dataset in df['dataset'].unique():
    ds_df = df[df['dataset'] == dataset]
    print(f"\n{dataset.upper()}:")
    print("-" * 80)
    
    print("\nEdge Matching Metrics:")
    for metric in ['TP', 'PP', 'FP', 'FN']:
        data = ds_df[metric]
        print(f"  {metric:3s}: Mean={data.mean():6.1f}, Median={data.median():6.1f}, "
              f"Std={data.std():6.1f}, Range=[{data.min():.0f}, {data.max():.0f}]")
    
    print("\nGraph Structure (AI-Generated):")
    print(f"  Nodes: Mean={ds_df['fcm1_nodes'].mean():6.1f}, "
          f"Median={ds_df['fcm1_nodes'].median():6.1f}, "
          f"Range=[{ds_df['fcm1_nodes'].min():.0f}, {ds_df['fcm1_nodes'].max():.0f}]")
    print(f"  Edges: Mean={ds_df['fcm1_edges'].mean():6.1f}, "
          f"Median={ds_df['fcm1_edges'].median():6.1f}, "
          f"Range=[{ds_df['fcm1_edges'].min():.0f}, {ds_df['fcm1_edges'].max():.0f}]")
    
    print("\nGraph Structure (Ground Truth):")
    print(f"  Nodes: Mean={ds_df['fcm2_nodes'].mean():6.1f}, "
          f"Median={ds_df['fcm2_nodes'].median():6.1f}, "
          f"Range=[{ds_df['fcm2_nodes'].min():.0f}, {ds_df['fcm2_nodes'].max():.0f}]")
    print(f"  Edges: Mean={ds_df['fcm2_edges'].mean():6.1f}, "
          f"Median={ds_df['fcm2_edges'].median():6.1f}, "
          f"Range=[{ds_df['fcm2_edges'].min():.0f}, {ds_df['fcm2_edges'].max():.0f}]")

print("\n" + "=" * 80)
print(f"All plots saved to: {output_dir}")
print("=" * 80)
