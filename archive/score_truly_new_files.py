"""
Score only truly new FCM files by comparing old and new AI directories.

This script:
1. Compares old fcm_ai directory with new fcm_ai_20260225 directory
2. Identifies files that exist in new but not in old (by filename)
3. Scores only the truly new files against ground truth
4. Appends results to existing result directories
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Set
from score_fcms import score_fcm_with_scorer, ScoreCalculator
import glob


def get_old_ai_files(old_ai_dir: str, dataset: str) -> Set[str]:
    """
    Get set of filenames from old AI directory.
    
    Args:
        old_ai_dir: Old AI base directory (fcm_ai)
        dataset: Dataset name with suffix (e.g., 'biodiversity_ai', 'flpp_ai')
    
    Returns:
        Set of CSV filenames from old directory
    """
    old_files = set()
    old_dataset_dir = os.path.join(old_ai_dir, dataset)
    
    if os.path.exists(old_dataset_dir):
        for csv_file in glob.glob(os.path.join(old_dataset_dir, '**', '*.csv'), 
                                  recursive=True):
            old_files.add(Path(csv_file).name)
    
    return old_files


def find_truly_new_file_pairs(new_ai_dataset_dir: str, gt_dataset_dir: str, 
                               old_ai_files: Set[str], verbose: bool = False) -> List[Tuple[str, str, str]]:
    """
    Find matching file pairs for truly new files only.
    
    Args:
        new_ai_dataset_dir: Directory with new AI FCMs
        gt_dataset_dir: Directory with ground truth FCMs  
        old_ai_files: Set of filenames from old AI directory
        verbose: Print matching information
    
    Returns:
        List of tuples (file_id, ai_path, gt_path) for truly new files only
    """
    # Find all CSV files in new AI directory
    ai_files = {}
    for csv_file in glob.glob(os.path.join(new_ai_dataset_dir, '**', '*.csv'), recursive=True):
        basename = Path(csv_file).name
        
        # Skip if file existed in old directory
        if basename in old_ai_files:
            continue
        
        # Determine file ID for matching with GT
        parent_dir = Path(csv_file).parent.name
        if parent_dir != Path(new_ai_dataset_dir).name:
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
        print(f"  Found {len(ai_files)} truly new AI files")
        print(f"  Matched {len(pairs)} new file pairs with GT")
        if len(pairs) < len(ai_files):
            ai_only = set(ai_files.keys()) - matching_ids
            if ai_only:
                print(f"  New AI files without GT match: {len(ai_only)}")
                print(f"    {', '.join(sorted(list(ai_only))[:5])}{'...' if len(ai_only) > 5 else ''}")
    
    return pairs


def score_truly_new_files(
    old_ai_base_dir: str,
    new_ai_base_dir: str,
    gt_base_dir: str,
    results_base_dir: str,
    datasets: Dict[str, Tuple[str, str]],  # Maps: new_dir_name -> (old_dir_suffix, results_dir_name)
    threshold: float = 0.6,
    tp_scale: float = 1.0,
    pp_scale: float = 0.6,
    verbose: bool = True
):
    """
    Score only truly new FCMs by comparing old and new AI directories.
    
    Args:
        old_ai_base_dir: Base directory with old AI FCMs (fcm_ai)
        new_ai_base_dir: Base directory with new AI FCMs (fcm_ai_20260225)
        gt_base_dir: Base directory with ground truth FCMs  
        results_base_dir: Base directory for saving results
        datasets: Dictionary mapping new AI dir names to (old dir suffix, results dir name)
        threshold: Similarity threshold for matching nodes
        tp_scale: Scale factor for true positive matches
        pp_scale: Scale factor for partial positive matches
        verbose: Print progress information
    """
    print("=" * 80)
    print("SCORING TRULY NEW FCM FILES")
    print("=" * 80)
    
    # Initialize scorer once for efficiency
    if verbose:
        print("\nInitializing semantic scorer...")
    
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    scorer = ScoreCalculator(
        threshold=threshold, 
        model_name=model_name,
        data="multi_dataset",
        tp_scale=tp_scale, 
        pp_scale=pp_scale
    )
    
    all_results = []
    dataset_summaries = {}
    
    for new_dataset_name, (old_dataset_suffix, results_dataset_name) in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {new_dataset_name}")
        print(f"{'=' * 80}")
        
        new_ai_dataset_dir = os.path.join(new_ai_base_dir, new_dataset_name)
        
        # Determine GT directory name (usually with _gt suffix)
        gt_dataset_name = results_dataset_name + "_gt"
        gt_dataset_dir = os.path.join(gt_base_dir, gt_dataset_name)
        
        if not os.path.exists(new_ai_dataset_dir):
            print(f"  WARNING: New AI directory not found: {new_ai_dataset_dir}")
            continue
        
        if not os.path.exists(gt_dataset_dir):
            print(f"  WARNING: GT directory not found: {gt_dataset_dir}")
            continue
        
        # Get old AI files
        old_ai_files = get_old_ai_files(old_ai_base_dir, old_dataset_suffix)
        print(f"\n  Old AI directory had: {len(old_ai_files)} files")
        
        # Find truly new file pairs
        file_pairs = find_truly_new_file_pairs(new_ai_dataset_dir, gt_dataset_dir, 
                                               old_ai_files, verbose=verbose)
        
        if not file_pairs:
            print(f"  No truly new files to score for {new_dataset_name}")
            continue
        
        print(f"\n  Processing {len(file_pairs)} truly new file pairs...")
        
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
                    'data': file_id,
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
            summary_file = os.path.join(output_dir, f"{results_dataset_name}_new_20260225_summary.csv")
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
        print("SUMMARY OF TRULY NEW FILES")
        print(f"{'=' * 80}\n")
        
        df_all = pd.DataFrame(all_results)
        combined_file = os.path.join(results_base_dir, "truly_new_files_20260225_results.csv")
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
        print(f"TOTAL TRULY NEW COMPARISONS: {len(all_results)}")
        print(f"{'=' * 80}")
    else:
        print("\nNo truly new files to score.")


if __name__ == "__main__":
    # Configuration
    OLD_AI_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_ai"
    NEW_AI_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_ai_20260225"
    GT_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_gt"
    RESULTS_BASE_DIR = r"C:\Users\Nbrug\Desktop\fcm_comparison_results"
    
    # Map new AI directory names to (old AI directory suffix, results directory name)
    DATASETS = {
        "Biodiversity": ("biodiversity_ai", "biodiversity"),
        "FLPP": ("flpp_ai", "flpp"),
        "Gulf OSW": ("osw_ai", "osw"),
        "Red snapper": ("red_snapper_ai", "red_snapper")
    }
    
    # Run scoring
    score_truly_new_files(
        old_ai_base_dir=OLD_AI_BASE_DIR,
        new_ai_base_dir=NEW_AI_BASE_DIR,
        gt_base_dir=GT_BASE_DIR,
        results_base_dir=RESULTS_BASE_DIR,
        datasets=DATASETS,
        threshold=0.6,
        tp_scale=1.0,
        pp_scale=0.6,
        verbose=True
    )
