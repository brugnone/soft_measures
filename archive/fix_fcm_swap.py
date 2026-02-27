"""
Fix the FCM1/FCM2 swap in the comparison results.

The scoring function treats fcm1 as ground truth and fcm2 as prediction,
but files were passed with fcm1=AI and fcm2=GT (backwards).

This script creates a corrected version with proper labels.
"""

import pandas as pd
import os

# Load the misaligned results
results_file = r"C:\Users\Nbrug\Desktop\fcm_comparison_results\all_fcm_comparisons_complete.csv"
df = pd.read_csv(results_file)

print("=" * 80)
print("CORRECTING FCM1/FCM2 LABELING")
print("=" * 80)

print(f"\nLoaded {len(df)} comparisons")
print(f"\nProblem: Files were passed as fcm1=AI, fcm2=GT")
print("But scoring treats fcm1=GT(reference), fcm2=AI(prediction)")
print("\nThis results in:")
print("  - fcm1_nodes, fcm1_edges = actually AI metrics")
print("  - fcm2_nodes, fcm2_edges = actually GT metrics")
print("  - FN counts AI edges not in GT (should be GT edges not in AI)")
print("  - FP counts GT edges not in AI (should be AI edges not in GT)")

# Create corrected dataframe with swapped and renamed columns
df_corrected = df.copy()

# Swap the node and edge counts with corrected names
df_corrected['ai_nodes'] = df['fcm1_nodes']
df_corrected['ai_edges'] = df['fcm1_edges']
df_corrected['gt_nodes'] = df['fcm2_nodes']
df_corrected['gt_edges'] = df['fcm2_edges']

# Swap FP and FN to match the corrected interpretation
# Original: FP = unmatched fcm2 (GT), FN = unmatched fcm1 (AI)
# Corrected: FP = unmatched AI, FN = unmatched GT
df_corrected['FP'] = df['FN']  # FP should be unmatched AI edges
df_corrected['FN'] = df['FP']  # FN should be unmatched GT edges

# Keep TP and PP as they are (symmetric)
# Keep F1 and Jaccard as they are (symmetric)

# Drop the old fcm1/fcm2 columns
df_corrected = df_corrected.drop(columns=['fcm1_nodes', 'fcm1_edges', 'fcm2_nodes', 'fcm2_edges'])

# Reorder columns for clarity
column_order = [
    'dataset', 'file_id', 'ai_file', 'gt_file', 'Model', 'data',
    'F1', 'Jaccard', 
    'TP', 'PP', 'FP', 'FN',
    'threshold', 'tp_scale', 'pp_scale',
    'ai_nodes', 'ai_edges', 'gt_nodes', 'gt_edges'
]
df_corrected = df_corrected[column_order]

# Verify the correction
print("\n" + "-" * 80)
print("VERIFICATION (first few rows):")
print("-" * 80)

for dataset in df_corrected['dataset'].unique():
    ds_data = df_corrected[df_corrected['dataset'] == dataset].iloc[0]
    tp, pp, fp, fn = int(ds_data['TP']), int(ds_data['PP']), int(ds_data['FP']), int(ds_data['FN'])
    ai_edges = int(ds_data['ai_edges'])
    gt_edges = int(ds_data['gt_edges'])
    
    sum_ai = tp + pp + fp
    sum_gt = tp + pp + fn
    
    print(f"\n{dataset}:")
    print(f"  AI edges: {ai_edges}, GT edges: {gt_edges}")
    print(f"  TP + PP + FP = {sum_ai} (should equal AI edges: {ai_edges}) -> {'✓' if sum_ai == ai_edges else 'X'}")
    print(f"  TP + PP + FN = {sum_gt} (should equal GT edges: {gt_edges}) -> {'✓' if sum_gt == gt_edges else 'X'}")

# Save corrected results
output_file = r"C:\Users\Nbrug\Desktop\fcm_comparison_results\all_fcm_comparisons_CORRECTED.csv"
df_corrected.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print(f"CORRECTED results saved to: {output_file}")
print("=" * 80)

# Print statistics with corrected interpretation
print("\n" + "=" * 80)
print("CORRECTED STATISTICS BY DATASET")
print("=" * 80)

for dataset in df_corrected['dataset'].unique():
    ds_df = df_corrected[df_corrected['dataset'] == dataset]
    print(f"\n{dataset.upper()}:")
    print("-" * 80)
    
    print("\nEdge Matching Metrics (CORRECTED):")
    for metric in ['TP', 'PP', 'FP', 'FN']:
        data = ds_df[metric]
        print(f"  {metric:3s}: Mean={data.mean():6.1f}, Median={data.median():6.1f}, "
              f"Std={data.std():6.1f}, Range=[{data.min():.0f}, {data.max():.0f}]")
    
    print("\nGraph Structure:")
    print(f"  AI Nodes:  Mean={ds_df['ai_nodes'].mean():6.1f}, "
          f"Median={ds_df['ai_nodes'].median():6.1f}, "
          f"Range=[{ds_df['ai_nodes'].min():.0f}, {ds_df['ai_nodes'].max():.0f}]")
    print(f"  AI Edges:  Mean={ds_df['ai_edges'].mean():6.1f}, "
          f"Median={ds_df['ai_edges'].median():6.1f}, "
          f"Range=[{ds_df['ai_edges'].min():.0f}, {ds_df['ai_edges'].max():.0f}]")
    print(f"  GT Nodes:  Mean={ds_df['gt_nodes'].mean():6.1f}, "
          f"Median={ds_df['gt_nodes'].median():6.1f}, "
          f"Range=[{ds_df['gt_nodes'].min():.0f}, {ds_df['gt_nodes'].max():.0f}]")
    print(f"  GT Edges:  Mean={ds_df['gt_edges'].mean():6.1f}, "
          f"Median={ds_df['gt_edges'].median():6.1f}, "
          f"Range=[{ds_df['gt_edges'].min():.0f}, {ds_df['gt_edges'].max():.0f}]")

print("\n" + "=" * 80)
print("KEY INSIGHT: Red Snapper now makes sense!")
print("  - AI-generated FCMs have MORE edges than GT (~28 vs ~11)")
print("  - High FN means many GT edges were not found by AI")
print("  - High FP means AI generated many edges not in GT")
print("  - Zero FP in original was because all small GT matched large AI")
print("=" * 80)
