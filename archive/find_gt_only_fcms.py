"""
Find ground-truth FCMs that don't have matching AI-generated FCMs.
"""

import os
import glob
from pathlib import Path
import pandas as pd


def extract_ai_id(csv_file, dataset_dir):
    """
    Extract the ID from an AI file using the same logic as compare_ai_to_groundtruth.py
    """
    parent_dir = Path(csv_file).parent.name
    if parent_dir != Path(dataset_dir).name:
        # File is in a subdirectory, use directory name as ID
        return parent_dir
    else:
        # File is directly in dataset_dir, extract from filename
        file_id = Path(csv_file).stem
        # Try to extract GT-style ID from filename patterns
        if '_' in file_id:
            parts = file_id.split('_')
            # For Red Snapper pattern (number_CODE_date)
            if len(parts) >= 2 and parts[0].isdigit():
                return parts[1]
            # For OSW pattern (CODE_description)
            elif len(parts) >= 1:
                return parts[0]
        return file_id


def find_gt_only_files(ai_base_dir, gt_base_dir, datasets=None):
    """
    Find all GT files that don't have matching AI files.
    
    Returns:
        Dictionary mapping dataset names to lists of GT-only file info
    """
    if datasets is None:
        # Auto-detect datasets
        ai_datasets = [d for d in os.listdir(ai_base_dir) 
                      if os.path.isdir(os.path.join(ai_base_dir, d))]
        gt_datasets = [d for d in os.listdir(gt_base_dir)
                      if os.path.isdir(os.path.join(gt_base_dir, d))]
        
        ai_stems = {d.replace('_ai', ''): d for d in ai_datasets}
        gt_stems = {d.replace('_gt', ''): d for d in gt_datasets}
        
        datasets = sorted(set(ai_stems.keys()) & set(gt_stems.keys()))
    
    results = {}
    
    for dataset_name in datasets:
        ai_dir = os.path.join(ai_base_dir, f"{dataset_name}_ai")
        gt_dir = os.path.join(gt_base_dir, f"{dataset_name}_gt")
        
        if not os.path.exists(ai_dir) or not os.path.exists(gt_dir):
            continue
        
        # Find all AI files
        ai_files = {}
        for csv_file in glob.glob(os.path.join(ai_dir, '**', '*.csv'), recursive=True):
            file_id = extract_ai_id(csv_file, ai_dir)
            ai_files[file_id] = csv_file
        
        # Find all GT files
        gt_files = {}
        for csv_file in glob.glob(os.path.join(gt_dir, '**', '*.csv'), recursive=True):
            file_id = Path(csv_file).stem
            gt_files[file_id] = csv_file
        
        # Find GT-only files
        gt_only_ids = set(gt_files.keys()) - set(ai_files.keys())
        
        if gt_only_ids:
            gt_only_list = []
            for file_id in sorted(gt_only_ids):
                gt_only_list.append({
                    'file_id': file_id,
                    'file_path': gt_files[file_id],
                    'relative_path': os.path.relpath(gt_files[file_id], gt_dir)
                })
            results[dataset_name] = gt_only_list
    
    return results


def main():
    """Main function to find and report GT-only FCMs."""
    
    ai_base_dir = r'C:\Users\Nbrug\Desktop\fcm_ai'
    gt_base_dir = r'C:\Users\Nbrug\Desktop\fcm_gt'
    output_file = r'C:\Users\Nbrug\Desktop\fcm_comparison_results\gt_only_fcms.csv'
    
    print("=" * 70)
    print("FINDING GROUND-TRUTH FCMs WITHOUT AI MATCHES")
    print("=" * 70)
    print(f"\nAI Base Directory: {ai_base_dir}")
    print(f"GT Base Directory: {gt_base_dir}\n")
    
    # Find GT-only files
    gt_only_results = find_gt_only_files(ai_base_dir, gt_base_dir)
    
    # Display results by dataset
    all_gt_only = []
    total_gt_only = 0
    
    for dataset_name in sorted(gt_only_results.keys()):
        gt_only_list = gt_only_results[dataset_name]
        count = len(gt_only_list)
        total_gt_only += count
        
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"GT-only FCMs: {count}\n")
        
        if count > 0:
            print("Files:")
            for item in gt_only_list:
                print(f"  - {item['file_id']}")
                # Add dataset name to item for CSV export
                item['dataset'] = dataset_name
                all_gt_only.append(item)
    
    # Create summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total datasets checked: {len(gt_only_results)}")
    print(f"Total GT-only FCMs: {total_gt_only}")
    
    # Save to CSV
    if all_gt_only:
        df = pd.DataFrame(all_gt_only)
        # Reorder columns
        df = df[['dataset', 'file_id', 'relative_path', 'file_path']]
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo GT-only files found - all GT files have matching AI files!")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
