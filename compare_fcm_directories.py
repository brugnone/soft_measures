"""
Compare FCMs from two directories and score matching files.

This script takes two directories containing FCM files (CSV or JSON),
finds files with matching names in both directories, scores them,
and outputs the results to a specified directory.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from score_fcms import score_fcm


def find_matching_files(dir1: str, dir2: str, extensions: List[str] = ['.csv', '.json']) -> List[Tuple[str, str, str]]:
    """
    Find files with matching names in two directories.
    
    Args:
        dir1: First directory path
        dir2: Second directory path
        extensions: List of file extensions to match (default: ['.csv', '.json'])
    
    Returns:
        List of tuples (filename_stem, path_in_dir1, path_in_dir2)
    """
    if not os.path.exists(dir1):
        raise FileNotFoundError(f"Directory 1 not found: {dir1}")
    if not os.path.exists(dir2):
        raise FileNotFoundError(f"Directory 2 not found: {dir2}")
    
    # Get all files with specified extensions from each directory
    dir1_files = {}
    dir2_files = {}
    
    for file in os.listdir(dir1):
        file_path = os.path.join(dir1, file)
        if os.path.isfile(file_path):
            stem, ext = os.path.splitext(file)
            if ext.lower() in extensions:
                dir1_files[stem] = file_path
    
    for file in os.listdir(dir2):
        file_path = os.path.join(dir2, file)
        if os.path.isfile(file_path):
            stem, ext = os.path.splitext(file)
            if ext.lower() in extensions:
                dir2_files[stem] = file_path
    
    # Find matching stems
    matching_stems = set(dir1_files.keys()) & set(dir2_files.keys())
    
    matches = [(stem, dir1_files[stem], dir2_files[stem]) for stem in sorted(matching_stems)]
    
    return matches


def compare_directories(
    dir1: str,
    dir2: str,
    output_dir: Optional[str] = None,
    output_format: str = "both",
    threshold: float = 0.6,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    tp_scale: float = 1.0,
    pp_scale: float = 1.1,
    batch_size: int = 2,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare all matching FCM files from two directories.
    
    Args:
        dir1: First directory containing FCM files
        dir2: Second directory containing FCM files
        output_dir: Directory to save results. If None, creates 'comparison_results' in current directory
        output_format: Output format - 'csv', 'json', or 'both' (default: both)
        threshold: Similarity threshold for matching (default: 0.6)
        model_name: Embedding model to use (default: Qwen/Qwen3-Embedding-0.6B)
        tp_scale: Scale factor for true positives (default: 1.0)
        pp_scale: Scale factor for partial positives (default: 1.1)
        batch_size: Batch size for processing (lower = less VRAM)
        seed: Random seed for reproducibility
        verbose: Print detailed information
    
    Returns:
        pd.DataFrame: Combined results for all file pairs
    """
    # Set output directory
    if output_dir is None:
        output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching files
    if verbose:
        print(f"Searching for matching files in:")
        print(f"  Directory 1: {dir1}")
        print(f"  Directory 2: {dir2}")
    
    matches = find_matching_files(dir1, dir2)
    
    if not matches:
        print("No matching files found!")
        return pd.DataFrame()
    
    if verbose:
        print(f"\nFound {len(matches)} matching file pair(s):")
        for stem, path1, path2 in matches:
            print(f"  - {stem}: {os.path.basename(path1)} <-> {os.path.basename(path2)}")
        print()
    
    # Score each pair
    all_results = []
    
    for i, (stem, path1, path2) in enumerate(matches, 1):
        if verbose:
            print(f"[{i}/{len(matches)}] Scoring: {stem}")
            print("=" * 60)
        
        try:
            # Create a subdirectory for this pair's results
            pair_output_dir = os.path.join(output_dir, stem)
            os.makedirs(pair_output_dir, exist_ok=True)
            
            # Score the FCM pair
            result = score_fcm(
                fcm1_path=path1,
                fcm2_path=path2,
                output_dir=pair_output_dir,
                output_format=output_format,
                threshold=threshold,
                model_name=model_name,
                tp_scale=tp_scale,
                pp_scale=pp_scale,
                batch_size=batch_size,
                seed=seed,
                verbose=verbose
            )
            
            # Add file pair information to result
            result.insert(0, 'file_pair', stem)
            result.insert(1, 'dir1_file', os.path.basename(path1))
            result.insert(2, 'dir2_file', os.path.basename(path2))
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        if verbose:
            print()
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        if output_format.lower() in ['csv', 'both']:
            output_file = os.path.join(output_dir, "combined_results.csv")
            combined_results.to_csv(output_file, index=False)
            if verbose:
                print(f"Combined results saved to: {output_file}")
        
        if output_format.lower() in ['json', 'both']:
            output_file = os.path.join(output_dir, "combined_results.json")
            combined_results.to_json(output_file, orient='records', indent=2)
            if verbose:
                print(f"Combined results saved to: {output_file}")
        
        return combined_results
    else:
        return pd.DataFrame()


def main():
    """CLI interface for comparing FCM directories."""
    parser = argparse.ArgumentParser(
        description='Compare FCMs from two directories by matching file names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all FCMs in two directories
  python compare_fcm_directories.py dir1/ dir2/
  
  # Save results to a specific directory
  python compare_fcm_directories.py dir1/ dir2/ --output-dir results/
  
  # Use custom threshold and batch size
  python compare_fcm_directories.py dir1/ dir2/ --threshold 0.7 --batch-size 4
        """
    )
    
    parser.add_argument('dir1', help='First directory containing FCM files')
    parser.add_argument('dir2', help='Second directory containing FCM files')
    parser.add_argument('--output-dir', help='Output directory (default: comparison_results)')
    parser.add_argument('--output-format', choices=['csv', 'json', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Similarity threshold (default: 0.6)')
    parser.add_argument('--model-name', default='Qwen/Qwen3-Embedding-0.6B',
                        help='Model name for scoring')
    parser.add_argument('--tp-scale', type=float, default=1.0,
                        help='True positive scale (default: 1.0)')
    parser.add_argument('--pp-scale', type=float, default=1.1,
                        help='Partial positive scale (default: 1.1)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    results = compare_directories(
        dir1=args.dir1,
        dir2=args.dir2,
        output_dir=args.output_dir,
        output_format=args.output_format,
        threshold=args.threshold,
        model_name=args.model_name,
        tp_scale=args.tp_scale,
        pp_scale=args.pp_scale,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    if not results.empty:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total pairs compared: {len(results)}")
        print(f"\nAverage scores:")
        print(f"  F1 Score:      {results['F1'].mean():.4f}")
        print(f"  Jaccard Score: {results['Jaccard'].mean():.4f}")
        print(f"\nSee output directory for detailed results: {args.output_dir or 'comparison_results'}")


if __name__ == "__main__":
    main()
