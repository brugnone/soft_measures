"""
Score only new FCM files from fcm_ai_20260225 that haven't been scored yet.

This script:
1. Identifies already-scored files from existing results
2. Finds new files in fcm_ai_20260225 directory
3. Scores only the new files against ground truth
4. Appends results to existing result directories
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Set
from score_fcms import score_fcm_with_scorer, ScoreCalculator
import glob


def get_already_scored_files(results_dir: str, dataset: str) -> Set[str]:
    """
    Get set of filenames that have already been scored for a dataset.
    
    Args:
        results_dir: Root results directory
        dataset: Dataset name (e.g., 'biodiversity', 'flpp')
    
    Returns:
        Set of already-scored base filenames (without _scoring_results suffix)
    """
    scored_files = set()
    dataset_results_dir = os.path.join(results_dir, dataset)
    
    if os.path.exists(dataset_results_dir):
        for csv_file in glob.glob(os.path.join(dataset_results_dir, '**', '*_scoring_results.csv'), 
                                  recursive=True):
            # Extract original filename from result filename
            base_name = Path(csv_file).stem.replace('_scoring_results', '') + '.csv'
            scored_files.add(base_name)
    
    return scored_files


def find_new_file_pairs(ai_dataset_dir: str, gt_dataset_dir: str, 
                        already_scored: Set[str], verbose: bool = False) -> List[Tuple[str, str, str]]:
    """
    Find matching file pairs that haven't been scored yet.
    
    Args:
        ai_dataset_dir: Directory with AI FCMs
        gt_dataset_dir: Directory with ground truth FCMs  
        already_scored: Set of already-scored AI filenames
        verbose: Print matching information
    
    Returns:
        List of tuples (file_id, ai_path, gt_path) for new files only
    """
    # Find all CSV files in AI directory
    ai_files = {}
    for csv_file in glob.glob(os.path.join(ai_dataset_dir, '**', '*.csv'), recursive=True):
        basename = Path(csv_file).name
        
        # Skip if already scored
        if basename in already_scored:
            continue
        
        # Determine file ID for matching with GT
        parent_dir = Path(csv_file).parent.name
        if parent_dir != Path(ai_dataset_dir).name:
            # File is in a subdirectory, use directory name as ID
            file_id = parent_dir
            # Extract GT-style ID from directory name patterns
            if '_' in file_id:
                parts = file_id.split('_')
                # For Red Snapper pattern (number_CODE_date)
                if len(parts) >= 2 and parts[0].isdigit():
                    file_id = parts[1]
                # For OSW pattern (CODE_description)
                elif len(parts) >= 1:
                    file_id = parts[0]
        else:
            # File is directly in dataset dir, use basename
            file_id = Path(csv_file).stem
            # Try to extract GT-style ID from filename patterns
            if '_' in file_id:
                parts = file_id.split('_')
                # For Red Snapper pattern (number_CODE_date)
                if len(parts) >= 2 and parts[0].isdigit():
                    file_id = parts[1]
                # For OSW pattern (CODE_description)
                elif len(parts) >= 1:
                    file_id = parts[0]
        
        ai_files[file_id] = csv_file
    
    # Find all CSV files in GT directory
    gt_files = {}
    for csv_file in glob.glob(os.path.join(gt_dataset_dir, '**', '*.csv'), recursive=True):
        file_id = Path(csv_file).stem
        gt_files[file_id] = csv_file
    
    # Find matching pairs
    matching_ids = set(ai_files.keys()) & set(gt_files.keys())
    pairs = [(file_id, ai_files[file_id], gt_files[file_id]) 
             for file_id in sorted(matching_ids)]
    
    if verbose:
        print(f"  Found {len(ai_files)} new AI files, {len(gt_files)} GT files")
        print(f"  Matched {len(pairs)} new file pairs")
        if len(pairs) < len(ai_files):
            ai_only = set(ai_files.keys()) - matching_ids
            if ai_only:
                print(f"  New AI files without GT match: {len(ai_only)}")
                print(f"    {', '.join(sorted(list(ai_only))[:5])}{'...' if len(ai_only) > 5 else ''}")
    
    return pairs


def score_new_files(
    ai_base_dir: str,
    gt_base_dir: str,
    results_base_dir: str,
    datasets: Dict[str, str],  # Maps dataset dir name to results dir name
    threshold: float = 0.6,
    tp_scale: float = 1.0,
    pp_scale: float = 0.6,
    verbose: bool = True
):
    """
    Score only new FCMs that haven't been scored yet.
    
    Args:
        ai_base_dir: Base directory with AI-generated FCMs (fcm_ai_20260225)
        gt_base_dir: Base directory with ground truth FCMs  
        results_base_dir: Base directory for saving results
        datasets: Dictionary mapping AI dir names to results dir names
        threshold: Similarity threshold for matching nodes
        tp_scale: Scale factor for true positive matches
        pp_scale: Scale factor for partial positive matches
        verbose: Print progress information
    """
    print("=" * 80)
    print("SCORING NEW FCM FILES")
    print("=" * 80)
    
    # Initialize scorer once for efficiency
    if verbose:
        print("\nInitializing semantic scorer...")
    
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    scorer = ScoreCalculator(
        threshold=threshold, 
        model_name=model_name,
        data="multi_dataset",  # Generic data name for multi-dataset scoring
        tp_scale=tp_scale, 
        pp_scale=pp_scale
    )
    
    all_results = []
    dataset_summaries = {}
    
    for ai_dataset_name, results_dataset_name in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {ai_dataset_name}")
        print(f"{'=' * 80}")
        
        ai_dataset_dir = os.path.join(ai_base_dir, ai_dataset_name)
        
        # Determine GT directory name (usually with _gt suffix)
        gt_dataset_name = results_dataset_name + "_gt"
        gt_dataset_dir = os.path.join(gt_base_dir, gt_dataset_name)
        
        if not os.path.exists(ai_dataset_dir):
            print(f"  WARNING: AI directory not found: {ai_dataset_dir}")
            continue
        
        if not os.path.exists(gt_dataset_dir):
            print(f"  WARNING: GT directory not found: {gt_dataset_dir}")
            continue
        
        # Get already-scored files
        already_scored = get_already_scored_files(results_base_dir, results_dataset_name)
        print(f"\n  Already scored: {len(already_scored)} files")
        
        # Find new file pairs
        file_pairs = find_new_file_pairs(ai_dataset_dir, gt_dataset_dir, 
                                        already_scored, verbose=verbose)
        
        if not file_pairs:
            print(f"  No new files to score for {ai_dataset_name}")
            continue
        
        print(f"\n  Processing {len(file_pairs)} new file pairs...")
        
        # Create output directory for this dataset
        output_dir = os.path.join(results_base_dir, results_dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Score each pair
        dataset_results = []
        for idx, (file_id, ai_file, gt_file) in enumerate(file_pairs, 1):
            if verbose:
                print(f"\n  [{idx}/{len(file_pairs)}] {file_id}")
                print(f"    AI: {Path(ai_file).name}")
                print(f"    GT: {Path(gt_file).name}")
            
            try:
                # Score the FCM pair
                result_df = score_fcm_with_scorer(
                    fcm1_path=gt_file,   # GT = reference, encoded WITHOUT prompt
                    fcm2_path=ai_file,   # AI = prediction, encoded WITH prompt
                    scorer=scorer,
                    output_dir=output_dir,
                    output_format='csv',
                    verbose=False
                )
                
                # Extract results from DataFrame
                results_dict = {
                    'dataset': results_dataset_name,
                    'file_id': file_id,
                    'ai_file': Path(ai_file).name,
                    'gt_file': Path(gt_file).name,
                    'Model': result_df['Model'].iloc[0],
                    'data': file_id,  # Use file_id as data name
                    'F1': result_df['F1'].iloc[0],
                    'Jaccard': result_df['Jaccard'].iloc[0],
                    'TP': result_df['TP'].iloc[0],
                    'PP': result_df['PP'].iloc[0],
                    'FP': result_df['FP'].iloc[0],
                    'FN': result_df['FN'].iloc[0],
                    'threshold': result_df['threshold'].iloc[0],
                    'tp_scale': result_df['tp_scale'].iloc[0],
                    'pp_scale': result_df['pp_scale'].iloc[0],
                    'fcm1_nodes': result_df['fcm1_nodes'].iloc[0],
                    'fcm1_edges': result_df['fcm1_edges'].iloc[0],
                    'fcm2_nodes': result_df['fcm2_nodes'].iloc[0],
                    'fcm2_edges': result_df['fcm2_edges'].iloc[0]
                }
                
                dataset_results.append(results_dict)
                all_results.append(results_dict)
                
                if verbose:
                    print(f"    F1: {results_dict['F1']:.4f}, Jaccard: {results_dict['Jaccard']:.4f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        # Save dataset summary if we have results
        if dataset_results:
            df_dataset = pd.DataFrame(dataset_results)
            summary_file = os.path.join(output_dir, f"{results_dataset_name}_new_files_summary.csv")
            df_dataset.to_csv(summary_file, index=False)
            print(f"\n  Saved {len(dataset_results)} results to {summary_file}")
            
            # Calculate summary statistics
            dataset_summaries[results_dataset_name] = {
                'count': len(dataset_results),
                'f1_mean': df_dataset['F1'].mean(),
                'f1_std': df_dataset['F1'].std(),
                'f1_min': df_dataset['F1'].min(),
                'f1_max': df_dataset['F1'].max(),
                'jaccard_mean': df_dataset['Jaccard'].mean(),
                'jaccard_std': df_dataset['Jaccard'].std()
            }
    
    # Save combined results
    if all_results:
        print(f"\n{'=' * 80}")
        print("SUMMARY OF NEW FILES")
        print(f"{'=' * 80}\n")
        
        df_all = pd.DataFrame(all_results)
        combined_file = os.path.join(results_base_dir, "new_files_20260225_results.csv")
        df_all.to_csv(combined_file, index=False)
        print(f"Combined results saved to: {combined_file}")
        
        # Print summary statistics
        print("\nDataset Statistics:")
        print("-" * 80)
        for dataset_name, stats in dataset_summaries.items():
            print(f"\n{dataset_name}:")
            print(f"  New comparisons: {stats['count']}")
            print(f"  F1 Score:   {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f} (range: {stats['f1_min']:.4f} - {stats['f1_max']:.4f})")
            print(f"  Jaccard:    {stats['jaccard_mean']:.4f} ± {stats['jaccard_std']:.4f}")
        
        print(f"\n{'=' * 80}")
        print(f"TOTAL NEW COMPARISONS: {len(all_results)}")
        print(f"{'=' * 80}")
    else:
        print("\nNo new files to score.")


if __name__ == "__main__":
    # Configuration
    AI_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_ai_20260225"
    GT_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_gt"
    RESULTS_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_comparison_results"
    
    # Map AI directory names to results directory names
    DATASETS = {
        "Biodiversity": "biodiversity",
        "FLPP": "flpp",
        "Gulf OSW": "osw",
        "Red snapper": "red_snapper"
    }
    
    # Run scoring
    score_new_files(
        ai_base_dir=AI_BASE_DIR,
        gt_base_dir=GT_BASE_DIR,
        results_base_dir=RESULTS_BASE_DIR,
        datasets=DATASETS,
        threshold=0.6,
        tp_scale=1.0,
        pp_scale=0.6,
        verbose=True
    )
