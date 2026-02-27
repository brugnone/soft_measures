"""
Re-score ALL FCM comparisons with CORRECT parameter order.

CRITICAL FIX: The scoring function treats fcm1 as ground truth (reference) and fcm2 as prediction (AI).
Previous scripts passed fcm1=AI, fcm2=GT (backwards!).

This script corrects the parameter order:
- fcm1_path = GT file (ground truth/reference)
- fcm2_path = AI file (prediction)

This matters because the embedding model uses asymmetric encoding:
- fcm2 edges (queries) get encoded WITH instruction prompt
- fcm1 edges (documents) get encoded WITHOUT prompt
Results in different embeddings → different matchings → different metrics.
"""

import pandas as pd
import os
from pathlib import Path
from score_fcms import score_fcm_with_scorer

def find_all_file_pairs():
    """
    Find all AI-GT file pairs by directly scanning directories.
    
    Directory structure:
    - AI: C:\\Users\\Nbrug\\Desktop\\fcm_ai_20260225\\{Biodiversity|FLPP|Gulf OSW|Red snapper}\\
    - GT: C:\\Users\\Nbrug\\Desktop\\fcm_gt\\{dataset}_gt\\
    
    Returns:
        dict: {dataset_name: [(ai_file, gt_file, file_id), ...]}
    """
    # Directory mappings
    ai_base_dir = r'C:\Users\Nbrug\Desktop\fcm_ai_20260225'
    gt_base_dir = r'C:\Users\Nbrug\Desktop\fcm_gt'
    
    dataset_dir_mapping = {
        'biodiversity': {'ai': 'Biodiversity', 'gt': 'biodiversity_gt'},
        'flpp': {'ai': 'FLPP', 'gt': 'flpp_gt'},
        'osw': {'ai': 'Gulf OSW', 'gt': 'osw_gt'},
        'red_snapper': {'ai': 'Red snapper', 'gt': 'red_snapper_gt'}
    }
    
    print("Scanning directories for AI-GT file pairs...")
    
    all_pairs = {}
    
    for dataset_name, dirs in dataset_dir_mapping.items():
        ai_dir = os.path.join(ai_base_dir, dirs['ai'])
        gt_dir = os.path.join(gt_base_dir, dirs['gt'])
        
        if not os.path.exists(ai_dir):
            print(f"WARNING: AI directory not found: {ai_dir}")
            continue
        if not os.path.exists(gt_dir):
            print(f"WARNING: GT directory not found: {gt_dir}")
            continue
        
        # Build dictionaries of all CSV files in each directory tree
        print(f"\n{dataset_name}:")
        print(f"  Scanning AI dir: {ai_dir}")
        ai_files = {}
        for root, dirs_list, files in os.walk(ai_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_id = Path(file).stem
                    
                    # Extract simpler file_id for matching with GT
                    # For Red Snapper: "357392_BeFa_7_26_21" -> "BeFa"
                    # For OSW: "AE_IEA-Wind CM - AE" -> "AE"
                    if dataset_name == 'red_snapper':
                        # Pattern: number_CODE_date -> extract CODE (2nd part)
                        parts = file_id.split('_')
                        if len(parts) >= 2 and parts[0].isdigit():
                            file_id = parts[1]
                    elif dataset_name == 'osw':
                        # Pattern: CODE_description -> extract CODE (1st part)
                        file_id = file_id.split('_')[0]
                    
                    # Skip scoring results files
                    if 'scoring_results' in file_id.lower():
                        continue
                    
                    ai_files[file_id] = os.path.join(root, file)
        print(f"    Found {len(ai_files)} AI CSV files")
        
        print(f"  Scanning GT dir: {gt_dir}")
        gt_files = {}
        for root, dirs_list, files in os.walk(gt_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_id = Path(file).stem
                    gt_files[file_id] = os.path.join(root, file)
        print(f"    Found {len(gt_files)} GT CSV files")
        
        # Find matching pairs (files with same base name in both AI and GT)
        matching_ids = set(ai_files.keys()) & set(gt_files.keys())
        
        pairs = []
        for file_id in sorted(matching_ids):
            ai_file = ai_files[file_id]
            gt_file = gt_files[file_id]
            pairs.append((ai_file, gt_file, file_id))
        
        all_pairs[dataset_name] = pairs
        print(f"  Matched {len(pairs)} file pairs")
    
    return all_pairs


def rescore_all_datasets(model_name='Qwen/Qwen-2.5-Coder-32B-Instruct',
                         threshold=0.6,
                         tp_scale=1.0,
                         pp_scale=0.6,
                         data='v2'):
    """
    Re-score all datasets with CORRECT parameter order.
    
    CRITICAL: Passes fcm1_path=GT (reference), fcm2_path=AI (prediction)
    """
    print("=" * 80)
    print("RE-SCORING ALL FCM COMPARISONS WITH CORRECT PARAMETER ORDER")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Parameters: threshold={threshold}, tp_scale={tp_scale}, pp_scale={pp_scale}")
    print(f"\nCRITICAL FIX:")
    print(f"  - fcm1_path = GT file (reference/ground truth)")
    print(f"  - fcm2_path = AI file (prediction)")
    print(f"  - fcm2 edges encoded WITH instruction prompt")
    print(f"  - fcm1 edges encoded WITHOUT prompt")
    print("=" * 80)
    
    # Find all file pairs
    print("\nFinding all file pairs...")
    all_pairs = find_all_file_pairs()
    
    total_pairs = sum(len(pairs) for pairs in all_pairs.values())
    print(f"\nTotal comparisons to process: {total_pairs}")
    
    # Create output directory
    output_base_dir = Path(r'C:\Users\Nbrug\Desktop\fcm_comparison_results_CORRECT')
    output_base_dir.mkdir(exist_ok=True)
    
    # Load model once and reuse for all scoring
    print("\n" + "=" * 80)
    print("LOADING MODEL (once, will be reused for all comparisons)")
    print("=" * 80)
    
    from score_fcms import ScoreCalculator
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create a scorer instance to load the model
    print(f"\nLoading {model_name}...")
    print("(This may take 10-20 seconds...)")
    
    # Initialize scorer - this loads the model and will be shared
    scorer = ScoreCalculator(
        threshold=threshold,
        model_name=model_name,
        data=data,
        tp_scale=tp_scale,
        pp_scale=pp_scale
    )
    
    print("[OK] Model loaded and ready")
    
    # Process each dataset
    all_results = []
    processed_count = 0
    
    for dataset_name, pairs in all_pairs.items():
        if not pairs:
            continue
        
        print("\n" + "=" * 80)
        print(f"PROCESSING: {dataset_name.upper()}")
        print("=" * 80)
        print(f"Files to process: {len(pairs)}")
        
        dataset_results = []
        
        for idx, (ai_file, gt_file, file_id) in enumerate(pairs, 1):
            processed_count += 1
            print(f"\n[{processed_count}/{total_pairs}] {dataset_name} - {file_id}")
            print(f"  GT: {Path(gt_file).name}")
            print(f"  AI: {Path(ai_file).name}")
            
            try:
                # CORRECT parameter order: fcm1=GT, fcm2=AI
                result_df = score_fcm_with_scorer(
                    fcm1_path=gt_file,  # CORRECT: GT is reference
                    fcm2_path=ai_file,  # CORRECT: AI is prediction
                    scorer=scorer,      # Reuse the loaded model
                    output_dir=None,    # Don't save individual files
                    verbose=False       # Reduce output noise
                )
                
                # Add metadata
                result_df['dataset'] = dataset_name
                result_df['file_id'] = file_id
                result_df['ai_file'] = ai_file
                result_df['gt_file'] = gt_file
                
                dataset_results.append(result_df)
                
                # Show key metrics
                f1 = result_df['F1'].values[0]
                tp = result_df['TP'].values[0]
                pp = result_df['PP'].values[0]
                fp = result_df['FP'].values[0]
                fn = result_df['FN'].values[0]
                print(f"  [OK] F1={f1:.3f}, TP={tp}, PP={pp}, FP={fp}, FN={fn}")
                
            except Exception as e:
                print(f"  [ERROR]: {e}")
                continue
        
        # Save dataset results
        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            
            # Reorder columns for clarity
            dataset_output_dir = output_base_dir / dataset_name
            dataset_output_dir.mkdir(exist_ok=True)
            
            output_file = dataset_output_dir / f'{dataset_name}_scoring_results_CORRECT.csv'
            dataset_df.to_csv(output_file, index=False)
            
            print(f"\n[OK] Saved {len(dataset_df)} results to: {output_file}")
            
            all_results.append(dataset_df)
    
    # Combine all results
    if all_results:
        print("\n" + "=" * 80)
        print("COMBINING ALL RESULTS")
        print("=" * 80)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        column_order = [
            'dataset', 'file_id', 'ai_file', 'gt_file', 'Model', 'data',
            'F1', 'Jaccard',
            'TP', 'PP', 'FP', 'FN',
            'threshold', 'tp_scale', 'pp_scale',
            'fcm1_nodes', 'fcm1_edges', 'fcm2_nodes', 'fcm2_edges'
        ]
        combined_df = combined_df[column_order]
        
        output_file = output_base_dir / 'all_fcm_comparisons_CORRECT.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n[OK] Combined results saved to: {output_file}")
        print(f"  Total comparisons: {len(combined_df)}")
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS (CORRECT RESULTS)")
        print("=" * 80)
        
        for dataset in combined_df['dataset'].unique():
            ds_df = combined_df[combined_df['dataset'] == dataset]
            print(f"\n{dataset.upper()}:")
            print(f"  Count: {len(ds_df)}")
            print(f"  F1:      Mean={ds_df['F1'].mean():.3f}, Median={ds_df['F1'].median():.3f}")
            print(f"  Jaccard: Mean={ds_df['Jaccard'].mean():.3f}, Median={ds_df['Jaccard'].median():.3f}")
            print(f"  TP:  Mean={ds_df['TP'].mean():.1f}, Median={ds_df['TP'].median():.1f}")
            print(f"  PP:  Mean={ds_df['PP'].mean():.1f}, Median={ds_df['PP'].median():.1f}")
            print(f"  FP:  Mean={ds_df['FP'].mean():.1f}, Median={ds_df['FP'].median():.1f}")
            print(f"  FN:  Mean={ds_df['FN'].mean():.1f}, Median={ds_df['FN'].median():.1f}")
            print(f"  GT edges (fcm1): Mean={ds_df['fcm1_edges'].mean():.1f}")
            print(f"  AI edges (fcm2): Mean={ds_df['fcm2_edges'].mean():.1f}")
        
        print("\n" + "=" * 80)
        print("RE-SCORING COMPLETE!")
        print("=" * 80)
        print(f"\nNOTE: Column names are fcm1_edges/fcm2_edges where:")
        print(f"  - fcm1_edges = GT edges (reference)")
        print(f"  - fcm2_edges = AI edges (prediction)")
        print(f"\nVerify identity constraints:")
        print(f"  - TP + PP + FN should equal fcm1_edges (GT)")
        print(f"  - TP + PP + FP should equal fcm2_edges (AI)")
        
        return combined_df
    
    return None


if __name__ == '__main__':
    # Use the correct model name
    model_name = 'Qwen/Qwen3-Embedding-0.6B'
    
    result_df = rescore_all_datasets(
        model_name=model_name,
        threshold=0.6,
        tp_scale=1.0,
        pp_scale=0.6,
        data='v2'
    )
    
    if result_df is not None:
        print(f"\n[OK] Successfully re-scored {len(result_df)} comparisons")
    else:
        print("\n[ERROR] No results generated")
