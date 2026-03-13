"""
score_missing_gpt5mini.py
=========================
Scores the 9 gpt5mini participants that have GT files but were missed in the
original scoring run. AI files are taken from fcm_adjacency_organised/gpt5mini/
rather than the original (now-empty) fcm_ai_20260225 source directory.

After scoring, the new rows are appended to all_non-aco_fcm_results_combined.csv.

Usage:
    python scripts/score_missing_gpt5mini.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from score_fcms import ScoreCalculator, score_fcm_with_scorer

# ── Paths ──────────────────────────────────────────────────────────────────────
DESKTOP   = Path(r"C:\Users\Nbrug\Desktop")
AI_BASE   = DESKTOP / "fcm_adjacency_organised" / "gpt5mini"
GT_BASE   = DESKTOP / "fcm_adjacency_organised" / "gt"
COMBINED  = DESKTOP / "all_non-aco_fcm_results_combined.csv"

# ── Scorer settings (must match existing runs) ────────────────────────────────
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
THRESHOLD   = 0.6
TP_SCALE    = 1.0
PP_SCALE    = 0.6
METHOD      = "gpt5mini"

# Participants to score: {dataset_folder: [file_ids]}
MISSING = {
    "biodiversity": ["BD042", "BD064"],
    "osw":          ["DoP", "MaS", "MiS", "NREL"],
    "red_snapper":  ["BeRa", "GuWa", "KeKn"],
}


def main():
    # ── Verify all files exist before loading model ────────────────────────────
    pairs = []
    print("Verifying file pairs …")
    for dataset, ids in MISSING.items():
        for fid in ids:
            ai_path = AI_BASE / dataset / f"{fid}.csv"
            gt_path = GT_BASE / dataset / f"{fid}.csv"
            if not ai_path.exists():
                raise FileNotFoundError(f"AI file missing: {ai_path}")
            if not gt_path.exists():
                raise FileNotFoundError(f"GT file missing: {gt_path}")
            pairs.append((dataset, fid, ai_path, gt_path))
            print(f"  OK {dataset}/{fid}")

    print(f"\n{len(pairs)} pairs ready.\n")

    # ── Load model once ────────────────────────────────────────────────────────
    print("Loading embedding model …")
    scorer = ScoreCalculator(
        threshold=THRESHOLD,
        model_name=EMBED_MODEL,
        data="init",
        tp_scale=TP_SCALE,
        pp_scale=PP_SCALE,
    )
    print("Model loaded.\n")

    # ── Score each pair ────────────────────────────────────────────────────────
    rows = []
    for dataset, fid, ai_path, gt_path in pairs:
        print(f"Scoring {fid} ({dataset}) …")
        # fcm1 = GT, fcm2 = AI  (matches convention in score.py)
        result = score_fcm_with_scorer(
            fcm1_path=str(gt_path),
            fcm2_path=str(ai_path),
            scorer=scorer,
            output_dir=None,   # suppress file output
            output_format="csv",
            verbose=False,
        )
        row = result.iloc[0].to_dict()
        # Map generic fcm1/fcm2 column names to gt/ai convention
        row["gt_nodes"] = int(row.pop("fcm1_nodes"))
        row["gt_edges"] = int(row.pop("fcm1_edges"))
        row["ai_nodes"] = int(row.pop("fcm2_nodes"))
        row["ai_edges"] = int(row.pop("fcm2_edges"))
        # Prepend identifying columns
        row["method"]   = METHOD
        row["dataset"]  = dataset
        row["file_id"]  = fid
        row["ai_file"]  = str(ai_path)
        row["gt_file"]  = str(gt_path)
        print(
            f"  F1={row['F1']:.4f}  TP={row['TP']}  PP={row['PP']}"
            f"  FP={row['FP']}  FN={row['FN']}"
        )
        rows.append(row)

    # ── Build new-rows DataFrame matching combined CSV column order ─────────────
    COLUMNS = [
        "method", "dataset", "file_id", "ai_file", "gt_file",
        "Model", "data", "F1", "Jaccard", "TP", "PP", "FP", "FN",
        "threshold", "tp_scale", "pp_scale",
        "gt_nodes", "gt_edges", "ai_nodes", "ai_edges",
    ]
    new_df = pd.DataFrame(rows)[COLUMNS]

    print(f"\nNew rows:\n{new_df[['file_id','dataset','F1','TP','PP','FP','FN']].to_string(index=False)}\n")

    # ── Check for duplicates before appending ─────────────────────────────────
    combined = pd.read_csv(COMBINED)
    existing_keys = set(zip(combined["method"], combined["file_id"]))
    dupes = [(r["method"], r["file_id"]) for r in rows if (r["method"], r["file_id"]) in existing_keys]
    if dupes:
        print(f"WARNING: {len(dupes)} rows already exist in combined CSV — skipping duplicates: {dupes}")
        new_df = new_df[~new_df.apply(lambda r: (r["method"], r["file_id"]) in set(dupes), axis=1)]

    if new_df.empty:
        print("Nothing new to append — all rows already present.")
        return

    # ── Append and save ────────────────────────────────────────────────────────
    updated = pd.concat([combined, new_df], ignore_index=True)
    updated.to_csv(COMBINED, index=False)
    print(f"Appended {len(new_df)} rows → {COMBINED}")
    print(f"Combined CSV now has {len(updated)} rows (was {len(combined)}).")


if __name__ == "__main__":
    main()
