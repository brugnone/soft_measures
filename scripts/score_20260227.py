"""
Score FCM comparisons for a batch of AI-generated FCMs.

Expected directory structure (participant-per-subfolder):
  AI: {batch_dir}/{Dataset}/{participant_id}/{file}.csv
      e.g. Biodiversity/BD001/BD001.csv
           Gulf OSW/AE/AE_IEA-Wind CM - AE.csv
           Red snapper/357392_BeFa_7_26_21/357392_BeFa_7_26_21.csv

Usage:
  python score_20260227.py                          # defaults (Feb 27 batch 1)
  python score_20260227.py <ai_dir> <output_dir>    # custom paths
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Ensure the repo root (containing score_fcms.py) is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from score_fcms import score_fcm_with_scorer


def find_all_file_pairs(ai_base_dir):
    """
    Find all AI-GT file pairs.

    AI directory structure: each participant is a subfolder containing one CSV.
    GT directory is fixed.

    Returns:
        dict: {dataset_name: [(ai_file, gt_file, file_id), ...]}
    """
    gt_base_dir = r'C:\Users\Nbrug\Desktop\fcm_gt'

    dataset_dir_mapping = {
        'biodiversity': {'ai': 'Biodiversity',  'gt': 'biodiversity_gt'},
        'flpp':         {'ai': 'FLPP',           'gt': 'flpp_gt'},
        'osw':          {'ai': 'Gulf OSW',        'gt': 'osw_gt'},
        'red_snapper':  {'ai': 'Red snapper',     'gt': 'red_snapper_gt'},
    }

    print("Scanning directories for AI-GT file pairs...")
    all_pairs = {}

    for dataset_name, dirs in dataset_dir_mapping.items():
        ai_dataset_dir = os.path.join(ai_base_dir, dirs['ai'])
        gt_dataset_dir = os.path.join(gt_base_dir, dirs['gt'])

        if not os.path.exists(ai_dataset_dir):
            print(f"WARNING: AI directory not found: {ai_dataset_dir}")
            continue
        if not os.path.exists(gt_dataset_dir):
            print(f"WARNING: GT directory not found: {gt_dataset_dir}")
            continue

        print(f"\n{dataset_name}:")

        # --- Collect AI files: one CSV per participant subfolder ---
        ai_files = {}
        for participant_folder in sorted(os.listdir(ai_dataset_dir)):
            participant_path = os.path.join(ai_dataset_dir, participant_folder)
            if not os.path.isdir(participant_path):
                continue

            # Find the CSV inside this participant folder
            csvs = [f for f in os.listdir(participant_path) if f.endswith('.csv')]
            if not csvs:
                print(f"  WARNING: no CSV in {participant_path}")
                continue

            # Use the folder name as the raw ID
            raw_id = participant_folder

            # Dataset-specific ID extraction to match GT filenames
            if dataset_name == 'red_snapper':
                # "357392_BeFa_7_26_21" -> "BeFa"  (numeric prefix)
                # "BeRa_4_5_23_R2"      -> "BeRa"  (no numeric prefix)
                parts = raw_id.split('_')
                if parts[0].isdigit():
                    file_id = parts[1]
                else:
                    file_id = parts[0]
            elif dataset_name == 'osw':
                # Most subfolder names are already the clean GT code ("AE", "AM", …)
                # A few use longer names in the AI batch that differ from GT abbreviations.
                osw_name_map = {
                    'DougP':     'DoP',
                    'MarianaS':  'MaS',
                    'MichelleS': 'MiS',
                    'NREL2':     'NREL',
                }
                file_id = osw_name_map.get(raw_id, raw_id)
            else:
                # biodiversity ("BD001") and flpp ("100") are direct matches
                file_id = raw_id

            ai_file = os.path.join(participant_path, csvs[0])
            ai_files[file_id] = ai_file

        print(f"  Found {len(ai_files)} AI files")

        # --- Collect GT files ---
        gt_files = {}
        for root, _, files in os.walk(gt_dataset_dir):
            for file in files:
                if file.endswith('.csv'):
                    gt_files[Path(file).stem] = os.path.join(root, file)
        print(f"  Found {len(gt_files)} GT files")

        # --- Match pairs ---
        matched_ids = set(ai_files.keys()) & set(gt_files.keys())
        pairs = [(ai_files[fid], gt_files[fid], fid) for fid in sorted(matched_ids)]

        unmatched_ai = set(ai_files.keys()) - matched_ids
        if unmatched_ai:
            print(f"  No GT match for: {sorted(unmatched_ai)}")

        print(f"  Matched {len(pairs)} file pairs")
        all_pairs[dataset_name] = pairs

    return all_pairs


def score_all_datasets(ai_base_dir, output_base_dir,
                       model_name='Qwen/Qwen3-Embedding-0.6B',
                       threshold=0.6, tp_scale=1.0, pp_scale=0.6, data='v2'):
    """Score all datasets and save combined results."""
    batch_name = Path(ai_base_dir).name
    print("=" * 80)
    print(f"SCORING FCM COMPARISONS: {batch_name} vs Ground Truth")
    print("=" * 80)
    print(f"\nModel:      {model_name}")
    print(f"Parameters: threshold={threshold}, tp_scale={tp_scale}, pp_scale={pp_scale}")
    print(f"\nConvention:")
    print(f"  fcm1 = GT (reference) – encoded WITHOUT prompt")
    print(f"  fcm2 = AI (prediction) – encoded WITH instruction prompt")
    print("=" * 80)

    all_pairs = find_all_file_pairs(ai_base_dir)
    total_pairs = sum(len(v) for v in all_pairs.values())
    print(f"\nTotal comparisons: {total_pairs}")

    # Create output directory
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_base_dir}")

    # Load model once
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    from score_fcms import ScoreCalculator
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {model_name}...")

    scorer = ScoreCalculator(
        threshold=threshold,
        model_name=model_name,
        data=data,
        tp_scale=tp_scale,
        pp_scale=pp_scale,
    )
    print("[OK] Model ready")

    # Score all pairs
    all_results = []
    processed = 0

    for dataset_name, pairs in all_pairs.items():
        if not pairs:
            continue

        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset_name.upper()}  ({len(pairs)} pairs)")
        print("=" * 80)

        dataset_results = []

        for ai_file, gt_file, file_id in pairs:
            processed += 1
            print(f"\n[{processed}/{total_pairs}] {dataset_name} – {file_id}")
            print(f"  GT : {Path(gt_file).name}")
            print(f"  AI : {Path(ai_file).name}")

            try:
                result_df = score_fcm_with_scorer(
                    fcm1_path=gt_file,   # GT = reference
                    fcm2_path=ai_file,   # AI = prediction
                    scorer=scorer,
                    output_dir=None,
                    verbose=False,
                )

                result_df['dataset'] = dataset_name
                result_df['file_id'] = file_id
                result_df['ai_file'] = ai_file
                result_df['gt_file'] = gt_file
                dataset_results.append(result_df)

                r = result_df.iloc[0]
                print(f"  [OK] F1={r['F1']:.3f}  TP={r['TP']}  PP={r['PP']}  "
                      f"FP={r['FP']}  FN={r['FN']}")

            except Exception as e:
                print(f"  [ERROR] {e}")

        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            ds_out_dir = output_base_dir / dataset_name
            ds_out_dir.mkdir(exist_ok=True)
            out_path = ds_out_dir / f'{dataset_name}_scoring_results.csv'
            dataset_df.to_csv(out_path, index=False)
            print(f"\n[OK] Saved {len(dataset_df)} results → {out_path}")
            all_results.append(dataset_df)

    # Combine and save
    if not all_results:
        print("\n[ERROR] No results to combine.")
        return None

    print("\n" + "=" * 80)
    print("COMBINING RESULTS")
    print("=" * 80)
    combined = pd.concat(all_results, ignore_index=True)

    # Rename fcm1/fcm2 columns to gt/ai for clarity
    rename_map = {
        'fcm1_nodes': 'gt_nodes', 'fcm1_edges': 'gt_edges',
        'fcm2_nodes': 'ai_nodes', 'fcm2_edges': 'ai_edges',
    }
    combined.rename(columns=rename_map, inplace=True)

    col_order = [
        'dataset', 'file_id', 'ai_file', 'gt_file', 'Model', 'data',
        'F1', 'Jaccard',
        'TP', 'PP', 'FP', 'FN',
        'threshold', 'tp_scale', 'pp_scale',
        'gt_nodes', 'gt_edges', 'ai_nodes', 'ai_edges',
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]

    out_file = output_base_dir / f'all_fcm_comparisons_{Path(ai_base_dir).name}.csv'
    combined.to_csv(out_file, index=False)
    print(f"\n[OK] Combined CSV → {out_file}")
    print(f"     Total rows: {len(combined)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for ds in combined['dataset'].unique():
        d = combined[combined['dataset'] == ds]
        print(f"\n{ds.upper()} (n={len(d)})")
        print(f"  F1:      {d['F1'].mean():.3f} ± {d['F1'].std():.3f}")
        print(f"  Jaccard: {d['Jaccard'].mean():.3f} ± {d['Jaccard'].std():.3f}")
        print(f"  FP (mean): {d['FP'].mean():.1f}   FN (mean): {d['FN'].mean():.1f}")
        print(f"  GT edges: {d['gt_edges'].mean():.1f}   AI edges: {d['ai_edges'].mean():.1f}")

    print(f"\n[OK] Done – {len(combined)} comparisons scored")
    return combined


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Score FCM batch vs ground truth')
    parser.add_argument('ai_dir', nargs='?',
                        default=r'C:\Users\Nbrug\Desktop\fcm_ai_20260227_160145',
                        help='Path to AI batch directory')
    parser.add_argument('output_dir', nargs='?',
                        default=None,
                        help='Output directory (default: Desktop/fcm_comparison_results_<batch_name>)')
    args = parser.parse_args()

    out_dir = args.output_dir or str(
        Path(r'C:\Users\Nbrug\Desktop') / f'fcm_comparison_results_{Path(args.ai_dir).name}'
    )

    score_all_datasets(
        ai_base_dir=args.ai_dir,
        output_base_dir=out_dir,
        model_name='Qwen/Qwen3-Embedding-0.6B',
        threshold=0.6,
        tp_scale=1.0,
        pp_scale=0.6,
        data='v2',
    )
