"""
Combine all FCM scoring results into a single comprehensive CSV file.
Includes both old results and new results from Feb 25, 2026.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def load_all_individual_results(results_base_dir, dataset_name):
    """
    Load all individual scoring result files for a dataset.
    
    Args:
        results_base_dir: Base results directory
        dataset_name: Dataset name (e.g., 'biodiversity', 'flpp')
    
    Returns:
        DataFrame with all results for the dataset
    """
    dataset_dir = os.path.join(results_base_dir, dataset_name)
    result_files = glob.glob(os.path.join(dataset_dir, '**', '*_scoring_results.csv'), recursive=True)
    
    all_results = []
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            # Add dataset column if not present
            if 'dataset' not in df.columns:
                df['dataset'] = dataset_name
            # Add file_id from filename if not present
            if 'file_id' not in df.columns:
                file_id = Path(result_file).stem.replace('_scoring_results', '')
                df['file_id'] = file_id
            # Add ai_file and gt_file if not present (use 'data' column as fallback)
            if 'ai_file' not in df.columns and 'data' in df.columns:
                df['ai_file'] = df['data'].apply(lambda x: f"{x}.csv")
            if 'gt_file' not in df.columns and 'data' in df.columns:
                df['gt_file'] = df['data'].apply(lambda x: f"{x}.csv")
            
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
            continue
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def combine_all_results(results_base_dir, output_file):
    """
    Combine all FCM scoring results into a single CSV.
    
    Args:
        results_base_dir: Base directory with all results
        output_file: Path to save combined results
    """
    print("=" * 80)
    print("COMBINING ALL FCM SCORING RESULTS")
    print("=" * 80)
    
    datasets = ['biodiversity', 'flpp', 'osw', 'red_snapper']
    all_data = []
    
    for dataset in datasets:
        print(f"\nLoading {dataset} results...")
        df = load_all_individual_results(results_base_dir, dataset)
        if not df.empty:
            print(f"  Found {len(df)} comparisons")
            all_data.append(df)
        else:
            print(f"  No results found")
    
    # Combine all datasets
    if all_data:
        print(f"\n{'-' * 80}")
        print("Combining all datasets...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure consistent column order
        desired_columns = [
            'dataset', 'file_id', 'ai_file', 'gt_file', 'Model', 'data',
            'F1', 'Jaccard', 'TP', 'PP', 'FP', 'FN',
            'threshold', 'tp_scale', 'pp_scale',
            'fcm1_nodes', 'fcm1_edges', 'fcm2_nodes', 'fcm2_edges'
        ]
        
        # Keep only columns that exist
        columns_to_use = [col for col in desired_columns if col in combined_df.columns]
        combined_df = combined_df[columns_to_use]
        
        # Remove duplicates based on dataset and file_id
        print(f"Total rows before deduplication: {len(combined_df)}")
        combined_df = combined_df.drop_duplicates(subset=['dataset', 'file_id'], keep='last')
        print(f"Total rows after deduplication: {len(combined_df)}")
        
        # Sort by dataset and file_id
        combined_df = combined_df.sort_values(['dataset', 'file_id'])
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        print(f"\n{'-' * 80}")
        print(f"Combined results saved to: {output_file}")
        print(f"Total comparisons: {len(combined_df)}")
        
        # Print summary by dataset
        print(f"\n{'=' * 80}")
        print("SUMMARY BY DATASET")
        print(f"{'=' * 80}\n")
        
        for dataset in datasets:
            dataset_df = combined_df[combined_df['dataset'] == dataset]
            if not dataset_df.empty:
                f1_mean = dataset_df['F1'].mean()
                f1_min = dataset_df['F1'].min()
                f1_max = dataset_df['F1'].max()
                jaccard_mean = dataset_df['Jaccard'].mean()
                
                print(f"{dataset}:")
                print(f"  Comparisons: {len(dataset_df)}")
                print(f"  F1 Score:    {f1_mean:.4f} (range: {f1_min:.4f} - {f1_max:.4f})")
                print(f"  Jaccard:     {jaccard_mean:.4f}\n")
        
        print(f"{'=' * 80}")
        print(f"TOTAL: {len(combined_df)} AI vs Ground Truth Comparisons")
        print(f"{'=' * 80}")
    else:
        print("\nNo results found to combine.")


if __name__ == "__main__":
    RESULTS_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_comparison_results"
    OUTPUT_FILE = os.path.join(RESULTS_BASE_DIR, "all_fcm_comparisons_complete.csv")
    
    combine_all_results(RESULTS_BASE_DIR, OUTPUT_FILE)
