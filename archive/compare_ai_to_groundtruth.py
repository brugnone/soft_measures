"""
Compare AI-generated FCMs with ground truth FCMs across multiple datasets.

This script handles different directory structures between AI and ground truth FCMs,
matches files by ID, and reports results grouped by dataset.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from score_fcms import score_fcm_with_scorer, ScoreCalculator
import glob


def find_dataset_file_pairs(ai_dataset_dir: str, gt_dataset_dir: str, 
                            verbose: bool = False) -> List[Tuple[str, str, str]]:
    """
    Find matching file pairs between AI and ground truth directories.
    
    Handles different naming conventions:
    - OSW: AI has "AE_IEA-Wind CM - AE.csv" vs GT has "AE.csv"
    - Red Snapper: AI has "357392_BeFa_7_26_21.csv" vs GT has "BeFa.csv"
    - Uses directory name as primary ID for AI files when CSV names don't match
    
    Args:
        ai_dataset_dir: Directory with AI FCMs (may have subdirectories)
        gt_dataset_dir: Directory with ground truth FCMs  
        verbose: Print matching information
    
    Returns:
        List of tuples (file_id, ai_path, gt_path)
    """
    import re
    
    # Find all CSV files in AI directory (may be in subdirectories)
    ai_files = {}
    for csv_file in glob.glob(os.path.join(ai_dataset_dir, '**', '*.csv'), recursive=True):
        # For AI files, try to use the parent directory name as ID
        parent_dir = Path(csv_file).parent.name
        if parent_dir != Path(ai_dataset_dir).name:
            # File is in a subdirectory, use directory name as ID
            file_id = parent_dir
            # Try to extract GT-style ID from directory name patterns
            # Pattern: "PREFIX_CODE_SUFFIX" -> extract CODE
            # Examples: "357392_BeFa_7_26_21" -> "BeFa", "AE" -> "AE"
            if '_' in file_id:
                parts = file_id.split('_')
                # For Red Snapper pattern (number_CODE_date)
                if len(parts) >= 2 and parts[0].isdigit():
                    file_id = parts[1]
                # For OSW pattern (CODE_description)
                elif len(parts) >= 1:
                    file_id = parts[0]
        else:
            # File is directly in ai_dataset_dir, use basename
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
    
    # Find all CSV files in GT directory (may be in subdirectories)
    gt_files = {}
    for csv_file in glob.glob(os.path.join(gt_dataset_dir, '**', '*.csv'), recursive=True):
        file_id = Path(csv_file).stem
        gt_files[file_id] = csv_file
    
    # Find matching pairs
    matching_ids = set(ai_files.keys()) & set(gt_files.keys())
    pairs = [(file_id, ai_files[file_id], gt_files[file_id]) 
             for file_id in sorted(matching_ids)]
    
    if verbose:
        print(f"  Found {len(ai_files)} AI files, {len(gt_files)} GT files")
        print(f"  Matched {len(pairs)} file pairs")
        if len(pairs) < len(ai_files) or len(pairs) < len(gt_files):
            ai_only = set(ai_files.keys()) - matching_ids
            gt_only = set(gt_files.keys()) - matching_ids
            if ai_only:
                print(f"  AI-only files: {len(ai_only)}")
            if gt_only:
                print(f"  GT-only files: {len(gt_only)}")
    
    return pairs


def compare_ai_to_groundtruth(
    ai_base_dir: str,
    gt_base_dir: str,
    output_dir: str = "fcm_comparison_results",
    datasets: Optional[List[str]] = None,
    threshold: float = 0.6,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    tp_scale: float = 1.0,
    pp_scale: float = 0.6,
    batch_size: int = 2,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Compare AI-generated FCMs with ground truth FCMs across multiple datasets.
    
    Args:
        ai_base_dir: Base directory containing AI FCM datasets
        gt_base_dir: Base directory containing ground truth FCM datasets
        output_dir: Directory to save results
        datasets: List of dataset names to process (None = auto-detect)
        threshold: Similarity threshold for matching
        model_name: Embedding model to use
        tp_scale: True positive scale factor
        pp_scale: Partial positive scale factor
        batch_size: Batch size for processing
        seed: Random seed for reproducibility
        verbose: Print detailed information
    
    Returns:
        Dictionary mapping dataset names to result DataFrames
    """
    # Auto-detect datasets if not specified
    if datasets is None:
        ai_datasets = [d for d in os.listdir(ai_base_dir) 
                      if os.path.isdir(os.path.join(ai_base_dir, d))]
        gt_datasets = [d for d in os.listdir(gt_base_dir)
                      if os.path.isdir(os.path.join(gt_base_dir, d))]
        
        # Match datasets by removing _ai and _gt suffixes
        ai_stems = {d.replace('_ai', ''): d for d in ai_datasets}
        gt_stems = {d.replace('_gt', ''): d for d in gt_datasets}
        
        datasets = sorted(set(ai_stems.keys()) & set(gt_stems.keys()))
        
        if verbose:
            print(f"Auto-detected {len(datasets)} datasets: {', '.join(datasets)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize scorer once for all comparisons (efficient!)
    if verbose:
        print(f"\n{'='*70}")
        print(f"INITIALIZING SCORER")
        print(f"{'='*70}")
        print(f"Loading model weights (this happens once for all comparisons)...")
    
    scorer = ScoreCalculator(
        threshold=threshold,
        model_name=model_name,
        data="ai_vs_gt",
        tp_scale=tp_scale,
        pp_scale=pp_scale,
        seed=seed
    )
    scorer.batch_size = batch_size
    
    if verbose:
        print("Scorer initialized\n")
    
    # Process each dataset
    all_dataset_results = {}
    
    for dataset_idx, dataset_name in enumerate(datasets, 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"DATASET {dataset_idx}/{len(datasets)}: {dataset_name.upper()}")
            print(f"{'='*70}")
        
        # Construct directory paths
        ai_dir = os.path.join(ai_base_dir, f"{dataset_name}_ai")
        gt_dir = os.path.join(gt_base_dir, f"{dataset_name}_gt")
        
        if not os.path.exists(ai_dir):
            print(f"⚠ Warning: AI directory not found: {ai_dir}")
            continue
        if not os.path.exists(gt_dir):
            print(f"⚠ Warning: GT directory not found: {gt_dir}")
            continue
        
        if verbose:
            print(f"AI Directory: {ai_dir}")
            print(f"GT Directory: {gt_dir}")
        
        # Find matching file pairs
        pairs = find_dataset_file_pairs(ai_dir, gt_dir, verbose=verbose)
        
        if not pairs:
            print(f"⚠ No matching files found for {dataset_name}")
            continue
        
        # Create dataset output directory
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Process each pair
        dataset_results = []
        
        if verbose:
            print(f"\nProcessing {len(pairs)} file pair(s)...\n")
        
        for pair_idx, (file_id, ai_path, gt_path) in enumerate(pairs, 1):
            if verbose:
                print(f"[{pair_idx}/{len(pairs)}] Comparing: {file_id}")
                print("-" * 70)
            
            try:
                # Create output directory for this pair
                pair_output_dir = os.path.join(dataset_output_dir, file_id)
                os.makedirs(pair_output_dir, exist_ok=True)
                
                # Score the pair using pre-initialized scorer
                result = score_fcm_with_scorer(
                    fcm1_path=ai_path,
                    fcm2_path=gt_path,
                    scorer=scorer,
                    output_dir=pair_output_dir,
                    output_format='csv',
                    verbose=verbose
                )
                
                # Add metadata
                result.insert(0, 'dataset', dataset_name)
                result.insert(1, 'file_id', file_id)
                result.insert(2, 'ai_file', os.path.basename(ai_path))
                result.insert(3, 'gt_file', os.path.basename(gt_path))
                
                dataset_results.append(result)
                
            except Exception as e:
                print(f"✗ Error processing {file_id}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
            
            if verbose:
                print()
        
        # Combine results for this dataset
        if dataset_results:
            combined_results = pd.concat(dataset_results, ignore_index=True)
            all_dataset_results[dataset_name] = combined_results
            
            # Save dataset-specific results
            csv_file = os.path.join(dataset_output_dir, f"{dataset_name}_results.csv")
            combined_results.to_csv(csv_file, index=False)
            
            if verbose:
                print(f"{'='*70}")
                print(f"{dataset_name.upper()} SUMMARY")
                print(f"{'='*70}")
                print(f"Files compared: {len(combined_results)}")
                print(f"Average F1 Score: {combined_results['F1'].mean():.4f}")
                print(f"Average Jaccard Score: {combined_results['Jaccard'].mean():.4f}")
                print(f"Results saved to: {csv_file}")
    
    # Create overall summary
    if all_dataset_results:
        combined_all = pd.concat(list(all_dataset_results.values()), ignore_index=True)
        
        # Save combined results
        combined_csv = os.path.join(output_dir, "all_datasets_results.csv")
        combined_all.to_csv(combined_csv, index=False)
        
        # Create summary by dataset
        summary = combined_all.groupby('dataset').agg({
            'F1': ['mean', 'std', 'min', 'max', 'count'],
            'Jaccard': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        summary_csv = os.path.join(output_dir, "dataset_summary.csv")
        summary.to_csv(summary_csv)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"OVERALL SUMMARY")
            print(f"{'='*70}")
            print(f"\nResults by dataset:")
            print(summary)
            print(f"\nAll results saved to: {output_dir}")
            print(f"  - Combined results: {combined_csv}")
            print(f"  - Summary: {summary_csv}")
    
    return all_dataset_results


def main():
    """Main function to compare AI vs ground truth FCMs."""
    
    # Directories
    ai_base_dir = r'C:\Users\Nbrug\Desktop\fcm_ai'
    gt_base_dir = r'C:\Users\Nbrug\Desktop\fcm_gt'
    output_dir = r'C:\Users\Nbrug\Desktop\fcm_comparison_results'
    
    print("=" * 70)
    print("AI vs GROUND TRUTH FCM COMPARISON")
    print("=" * 70)
    print(f"\nAI Base Directory: {ai_base_dir}")
    print(f"GT Base Directory: {gt_base_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Run comparison
    results = compare_ai_to_groundtruth(
        ai_base_dir=ai_base_dir,
        gt_base_dir=gt_base_dir,
        output_dir=output_dir,
        datasets=None,  # Auto-detect
        threshold=0.6,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        tp_scale=1.0,
        pp_scale=0.6,
        batch_size=2,
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("✓ COMPARISON COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
