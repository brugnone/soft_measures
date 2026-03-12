"""
Score FCM comparisons for the three no-reasoning AI batches generated on 2026-03-03.

  Batch 1 (Gemini 2.5 Flash – no reasoning):
    fcm_adjacency_matrices_fcm_interviews_gemini25flash_nr_20260303_192146
  Batch 2 (Gemini 3 Flash – no reasoning):
    fcm_adjacency_matrices_fcm_interviews_gemini3flash_nr_20260303_192210
  Batch 3 (GPT-4o-mini o3 – no reasoning):
    fcm_adjacency_matrices_fcm_interviews_gpt52_nr_20260303_192229

All batches share the same directory structure as previous batches:
  {batch_dir}/{Dataset}/{participant_id}/{file}.csv
      e.g.  Biodiversity/BD001/BD001.csv
            Gulf OSW/AE/...csv
            Red snapper/357392_BeFa_7_26_21/...csv

The model is loaded once and reused across all three batches.

Usage:
  python score_20260303.py                  # scores all three default batches
  python score_20260303.py <ai_dir> ...     # positional: one or more ai_dirs
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Ensure repo root (score_fcms.py) is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from score_fcms import score_fcm_with_scorer, ScoreCalculator
import torch


# ---------------------------------------------------------------------------
# Ground-truth directory and dataset mappings (same as previous scripts)
# ---------------------------------------------------------------------------
GT_BASE_DIR = r'C:\Users\Nbrug\Desktop\fcm_gt'

DATASET_DIR_MAPPING = {
    'biodiversity': {'ai': 'Biodiversity',  'gt': 'biodiversity_gt'},
    'flpp':         {'ai': 'FLPP',           'gt': 'flpp_gt'},
    'osw':          {'ai': 'Gulf OSW',        'gt': 'osw_gt'},
    'red_snapper':  {'ai': 'Red snapper',     'gt': 'red_snapper_gt'},
}

# OSW subfolder names that differ from the GT abbreviation
OSW_NAME_MAP = {
    'DougP':     'DoP',
    'MarianaS':  'MaS',
    'MichelleS': 'MiS',
    'NREL2':     'NREL',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_all_file_pairs(ai_base_dir):
    """
    Find AI-GT CSV file pairs for a single AI batch directory.

    Returns:
        dict: {dataset_name: [(ai_file, gt_file, file_id), ...]}
    """
    print(f"\nScanning AI batch: {ai_base_dir}")
    all_pairs = {}

    for dataset_name, dirs in DATASET_DIR_MAPPING.items():
        ai_dataset_dir = os.path.join(ai_base_dir, dirs['ai'])
        gt_dataset_dir = os.path.join(GT_BASE_DIR, dirs['gt'])

        if not os.path.exists(ai_dataset_dir):
            print(f"  WARNING: AI dir not found: {ai_dataset_dir}")
            continue
        if not os.path.exists(gt_dataset_dir):
            print(f"  WARNING: GT dir not found: {gt_dataset_dir}")
            continue

        print(f"\n  {dataset_name}:")

        # --- Collect AI files: one CSV per participant subfolder ---
        ai_files = {}
        for participant_folder in sorted(os.listdir(ai_dataset_dir)):
            participant_path = os.path.join(ai_dataset_dir, participant_folder)
            if not os.path.isdir(participant_path):
                continue

            csvs = [f for f in os.listdir(participant_path) if f.endswith('.csv')]
            if not csvs:
                print(f"    WARNING: no CSV in {participant_path}")
                continue

            raw_id = participant_folder

            if dataset_name == 'red_snapper':
                # "357392_BeFa_7_26_21" -> "BeFa"  /  "BeRa_4_5_23_R2" -> "BeRa"
                parts = raw_id.split('_')
                file_id = parts[1] if parts[0].isdigit() else parts[0]
            elif dataset_name == 'osw':
                file_id = OSW_NAME_MAP.get(raw_id, raw_id)
            else:
                file_id = raw_id  # biodiversity / flpp are direct matches

            ai_files[file_id] = os.path.join(participant_path, csvs[0])

        print(f"    Found {len(ai_files)} AI files")

        # --- Collect GT files ---
        gt_files = {}
        for root, _, files in os.walk(gt_dataset_dir):
            for file in files:
                if file.endswith('.csv'):
                    gt_files[Path(file).stem] = os.path.join(root, file)
        print(f"    Found {len(gt_files)} GT files")

        # --- Match ---
        matched_ids = set(ai_files.keys()) & set(gt_files.keys())
        unmatched = set(ai_files.keys()) - matched_ids
        if unmatched:
            print(f"    No GT match for: {sorted(unmatched)}")

        pairs = [(ai_files[fid], gt_files[fid], fid) for fid in sorted(matched_ids)]
        print(f"    Matched {len(pairs)} file pairs")
        all_pairs[dataset_name] = pairs

    return all_pairs


def score_batch(ai_base_dir, output_base_dir, scorer):
    """
    Score one AI batch against ground truth using a pre-loaded *scorer*.

    Returns combined DataFrame of all results (or None if nothing scored).
    """
    batch_name = Path(ai_base_dir).name
    print("\n" + "=" * 80)
    print(f"SCORING BATCH: {batch_name}")
    print("=" * 80)

    all_pairs = find_all_file_pairs(ai_base_dir)
    total_pairs = sum(len(v) for v in all_pairs.values())
    print(f"\nTotal comparisons: {total_pairs}")

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_base_dir}")

    all_results = []
    processed = 0

    for dataset_name, pairs in all_pairs.items():
        if not pairs:
            continue

        print(f"\n{'=' * 60}")
        print(f"DATASET: {dataset_name.upper()}  ({len(pairs)} pairs)")
        print("=" * 60)

        dataset_results = []

        for ai_file, gt_file, file_id in pairs:
            processed += 1
            print(f"\n  [{processed}/{total_pairs}] {dataset_name} – {file_id}")
            print(f"    GT : {Path(gt_file).name}")
            print(f"    AI : {Path(ai_file).name}")

            try:
                result_df = score_fcm_with_scorer(
                    fcm1_path=gt_file,   # GT = reference (encoded WITHOUT prompt)
                    fcm2_path=ai_file,   # AI = prediction (encoded WITH instruction prompt)
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
                print(f"    [OK] F1={r['F1']:.3f}  TP={r['TP']}  PP={r['PP']}  "
                      f"FP={r['FP']}  FN={r['FN']}")

            except Exception as e:
                print(f"    [ERROR] {e}")

        if dataset_results:
            dataset_df = pd.concat(dataset_results, ignore_index=True)
            ds_out_dir = output_base_dir / dataset_name
            ds_out_dir.mkdir(exist_ok=True)
            out_path = ds_out_dir / f'{dataset_name}_scoring_results.csv'
            dataset_df.to_csv(out_path, index=False)
            print(f"\n  [OK] Saved {len(dataset_df)} results -> {out_path}")
            all_results.append(dataset_df)

    if not all_results:
        print("\n[ERROR] No results to combine.")
        return None

    combined = pd.concat(all_results, ignore_index=True)

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

    out_file = output_base_dir / f'all_fcm_comparisons_{batch_name}.csv'
    combined.to_csv(out_file, index=False)
    print(f"\n[OK] Combined CSV -> {out_file}  ({len(combined)} rows)")

    # Per-dataset summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    for ds in combined['dataset'].unique():
        d = combined[combined['dataset'] == ds]
        print(f"\n  {ds.upper()} (n={len(d)})")
        print(f"    F1:      {d['F1'].mean():.3f} ± {d['F1'].std():.3f}")
        print(f"    Jaccard: {d['Jaccard'].mean():.3f} ± {d['Jaccard'].std():.3f}")
        print(f"    FP: {d['FP'].mean():.1f}   FN: {d['FN'].mean():.1f}")
        print(f"    GT edges: {d['gt_edges'].mean():.1f}   AI edges: {d['ai_edges'].mean():.1f}")

    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

NR_BASE = r'C:\Users\Nbrug\Desktop\no reasoning results'

DEFAULT_BATCHES = [
    rf'{NR_BASE}\fcm_adjacency_matrices_fcm_interviews_gemini25flash_nr_20260303_192146',
    rf'{NR_BASE}\fcm_adjacency_matrices_fcm_interviews_gemini3flash_nr_20260303_192210',
    rf'{NR_BASE}\fcm_adjacency_matrices_fcm_interviews_gpt52_nr_20260303_192229',
]

MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
THRESHOLD  = 0.6
TP_SCALE   = 1.0
PP_SCALE   = 0.6
DATA       = 'v2'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Score one or more FCM batch directories against ground truth (model loaded once).'
    )
    parser.add_argument(
        'ai_dirs', nargs='*',
        default=DEFAULT_BATCHES,
        help='One or more AI batch directories to score (default: the three 2026-03-03 no-reasoning batches)',
    )
    args = parser.parse_args()

    ai_dirs = args.ai_dirs

    print("=" * 80)
    print("FCM SCORING  –  2026-03-03 No-Reasoning batches")
    print("=" * 80)
    print(f"\nBatches to score ({len(ai_dirs)}):")
    for d in ai_dirs:
        print(f"  {d}")
    print(f"\nModel:      {MODEL_NAME}")
    print(f"Parameters: threshold={THRESHOLD}, tp_scale={TP_SCALE}, pp_scale={PP_SCALE}")
    print(f"\nConvention:")
    print(f"  fcm1 = GT (reference) – encoded WITHOUT prompt")
    print(f"  fcm2 = AI (prediction) – encoded WITH instruction prompt")
    print("=" * 80)

    # Load the model ONCE and reuse across all batches
    print("\nLoading model (once)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scorer = ScoreCalculator(
        threshold=THRESHOLD,
        model_name=MODEL_NAME,
        data=DATA,
        tp_scale=TP_SCALE,
        pp_scale=PP_SCALE,
    )
    print("[OK] Model ready\n")

    # Score each batch
    for ai_dir in ai_dirs:
        batch_name = Path(ai_dir).name
        out_dir = Path(r'C:\Users\Nbrug\Desktop') / f'fcm_comparison_results_{batch_name}'
        score_batch(
            ai_base_dir=ai_dir,
            output_base_dir=str(out_dir),
            scorer=scorer,
        )

    print("\n" + "=" * 80)
    print("ALL BATCHES COMPLETE")
    print("=" * 80)
