"""
score.py  –  Unified FCM scoring script
========================================

Two subcommands:

  standard  –  Score AI-generated CSV FCMs against ground-truth CSVs.
               AI structure: {batch_dir}/{Dataset}/{participant_id}/{file}.csv
               GT structure:  C:\\Users\\Nbrug\\Desktop\\fcm_gt\\{dataset}_gt\\{id}.csv

  aco       –  Score ACO-generated JSON FCMs against ground-truth CSVs.
               ACO structure: Desktop/aco_adjacencies/{model}/{dataset}/{participant}/{id}_fcm.json
               GT structure:  Desktop/aco_adjacencies/gt/{dataset}/{id}.csv

Examples
--------
  # Score one or more standard batch directories (defaults to the 2026-03-03 batches):
  python score.py standard
  python score.py standard <ai_dir1> [<ai_dir2> ...]

  # Score ACO models (defaults to aco_mistral + aco_qwen):
  python score.py aco
  python score.py aco --models aco_mistral aco_qwen
  python score.py aco --output C:\\path\\to\\all_aco_results.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure repo root (score_fcms.py) is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score_fcms import ScoreCalculator, load_matrix_from_file, score_fcm_with_scorer


# ──────────────────────────────────────────────────────────────────────────────
# Shared scorer settings
# ──────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
THRESHOLD   = 0.6
TP_SCALE    = 1.0
PP_SCALE    = 0.6
DATA_TAG    = "v2"  # stored in 'data' column of standard results

DESKTOP = Path(r"C:\Users\Nbrug\Desktop")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STANDARD MODE  (CSV-based AI files against fcm_gt)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

GT_BASE_DIR = DESKTOP / "fcm_gt"

DATASET_DIR_MAPPING = {
    "biodiversity": {"ai": "Biodiversity",  "gt": "biodiversity_gt"},
    "flpp":         {"ai": "FLPP",           "gt": "flpp_gt"},
    "osw":          {"ai": "Gulf OSW",        "gt": "osw_gt"},
    "red_snapper":  {"ai": "Red snapper",     "gt": "red_snapper_gt"},
}

# OSW subfolder names that differ from the GT abbreviation
OSW_NAME_MAP = {
    "DougP":     "DoP",
    "MarianaS":  "MaS",
    "MichelleS": "MiS",
    "NREL2":     "NREL",
}


def find_all_file_pairs(ai_base_dir: str) -> dict:
    """
    Discover AI–GT CSV file pairs for one standard batch directory.

    Returns
    -------
    dict  {dataset_name: [(ai_file, gt_file, file_id), ...]}
    """
    print(f"\nScanning AI batch: {ai_base_dir}")
    all_pairs: dict = {}

    for dataset_name, dirs in DATASET_DIR_MAPPING.items():
        ai_dataset_dir = Path(ai_base_dir) / dirs["ai"]
        gt_dataset_dir = GT_BASE_DIR / dirs["gt"]

        if not ai_dataset_dir.exists():
            print(f"  WARNING: AI dir not found: {ai_dataset_dir}")
            continue
        if not gt_dataset_dir.exists():
            print(f"  WARNING: GT dir not found: {gt_dataset_dir}")
            continue

        print(f"\n  {dataset_name}:")

        # ── AI files: one CSV per participant subfolder ─────────────────────
        ai_files: dict = {}
        for part_folder in sorted(ai_dataset_dir.iterdir()):
            if not part_folder.is_dir():
                continue
            csvs = list(part_folder.glob("*.csv"))
            if not csvs:
                print(f"    WARNING: no CSV in {part_folder}")
                continue

            raw_id = part_folder.name

            if dataset_name == "red_snapper":
                # "357392_BeFa_7_26_21" → "BeFa"  /  "BeRa_4_5_23_R2" → "BeRa"
                parts = raw_id.split("_")
                file_id = parts[1] if parts[0].isdigit() else parts[0]
            elif dataset_name == "osw":
                file_id = OSW_NAME_MAP.get(raw_id, raw_id)
            else:
                file_id = raw_id  # biodiversity / flpp: direct match

            ai_files[file_id] = str(csvs[0])

        print(f"    Found {len(ai_files)} AI files")

        # ── GT files ────────────────────────────────────────────────────────
        gt_files = {p.stem: str(p) for p in gt_dataset_dir.rglob("*.csv")}
        print(f"    Found {len(gt_files)} GT files")

        # ── Match ───────────────────────────────────────────────────────────
        matched_ids = set(ai_files) & set(gt_files)
        unmatched   = set(ai_files) - matched_ids
        if unmatched:
            print(f"    No GT match for: {sorted(unmatched)}")

        pairs = [(ai_files[fid], gt_files[fid], fid) for fid in sorted(matched_ids)]
        print(f"    Matched {len(pairs)} file pairs")
        all_pairs[dataset_name] = pairs

    return all_pairs


def score_standard_batch(ai_base_dir: str, output_base_dir: str, scorer: ScoreCalculator):
    """Score one standard AI batch. Returns combined DataFrame (or None)."""
    batch_name = Path(ai_base_dir).name
    print("\n" + "=" * 80)
    print(f"SCORING BATCH: {batch_name}")
    print("=" * 80)

    all_pairs   = find_all_file_pairs(ai_base_dir)
    total_pairs = sum(len(v) for v in all_pairs.values())
    print(f"\nTotal comparisons: {total_pairs}")

    out_dir = Path(output_base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    all_results = []
    processed   = 0

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
                    fcm1_path=gt_file,   # GT = reference, encoded WITHOUT prompt
                    fcm2_path=ai_file,   # AI = prediction, encoded WITH prompt
                    scorer=scorer,
                    output_dir=None,
                    verbose=False,
                )
                result_df["dataset"] = dataset_name
                result_df["file_id"] = file_id
                result_df["ai_file"] = ai_file
                result_df["gt_file"] = gt_file
                dataset_results.append(result_df)

                r = result_df.iloc[0]
                print(f"    [OK] F1={r['F1']:.3f}  TP={r['TP']}  PP={r['PP']}  "
                      f"FP={r['FP']}  FN={r['FN']}")
            except Exception as e:
                print(f"    [ERROR] {e}")

        if dataset_results:
            ds_df = pd.concat(dataset_results, ignore_index=True)
            ds_out = out_dir / dataset_name
            ds_out.mkdir(exist_ok=True)
            ds_df.to_csv(ds_out / f"{dataset_name}_scoring_results.csv", index=False)
            print(f"\n  [OK] Saved {len(ds_df)} results -> {ds_out}")
            all_results.append(ds_df)

    if not all_results:
        print("\n[ERROR] No results to combine.")
        return None

    combined = pd.concat(all_results, ignore_index=True)

    combined.rename(columns={
        "fcm1_nodes": "gt_nodes", "fcm1_edges": "gt_edges",
        "fcm2_nodes": "ai_nodes", "fcm2_edges": "ai_edges",
    }, inplace=True)

    col_order = [
        "dataset", "file_id", "ai_file", "gt_file", "Model", "data",
        "F1", "Jaccard",
        "TP", "PP", "FP", "FN",
        "threshold", "tp_scale", "pp_scale",
        "gt_nodes", "gt_edges", "ai_nodes", "ai_edges",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]

    out_file = out_dir / f"all_fcm_comparisons_{batch_name}.csv"
    combined.to_csv(out_file, index=False)
    print(f"\n[OK] Combined CSV -> {out_file}  ({len(combined)} rows)")

    # Summary
    print("\n" + "-" * 60 + "\nSUMMARY\n" + "-" * 60)
    for ds in combined["dataset"].unique():
        d = combined[combined["dataset"] == ds]
        print(f"\n  {ds.upper()} (n={len(d)})")
        print(f"    F1:      {d['F1'].mean():.3f} ± {d['F1'].std():.3f}")
        print(f"    Jaccard: {d['Jaccard'].mean():.3f} ± {d['Jaccard'].std():.3f}")
        print(f"    FP: {d['FP'].mean():.1f}   FN: {d['FN'].mean():.1f}")
        print(f"    GT edges: {d['gt_edges'].mean():.1f}   AI edges: {d['ai_edges'].mean():.1f}")

    return combined


def cmd_standard(args):
    """Entry point for `python score.py standard ...`."""
    # Default batches (2026-03-03 no-reasoning)
    NR_BASE = DESKTOP / "no reasoning results"
    DEFAULT_BATCHES = [
        str(NR_BASE / "fcm_adjacency_matrices_fcm_interviews_gemini25flash_nr_20260303_192146"),
        str(NR_BASE / "fcm_adjacency_matrices_fcm_interviews_gemini3flash_nr_20260303_192210"),
        str(NR_BASE / "fcm_adjacency_matrices_fcm_interviews_gpt52_nr_20260303_192229"),
    ]

    ai_dirs = args.ai_dirs or DEFAULT_BATCHES

    print("=" * 80)
    print("FCM SCORING  –  Standard (CSV) mode")
    print("=" * 80)
    print(f"\nBatches to score ({len(ai_dirs)}):")
    for d in ai_dirs:
        print(f"  {d}")
    print(f"\nModel:      {EMBED_MODEL}")
    print(f"Parameters: threshold={THRESHOLD}, tp_scale={TP_SCALE}, pp_scale={PP_SCALE}")
    print(f"\nConvention: fcm1 = GT (no prompt),  fcm2 = AI (with prompt)")
    print("=" * 80)

    print("\nLoading embedding model …")
    scorer = ScoreCalculator(
        threshold=THRESHOLD,
        model_name=EMBED_MODEL,
        data=DATA_TAG,
        tp_scale=TP_SCALE,
        pp_scale=PP_SCALE,
    )
    print("[OK] Model ready\n")

    for ai_dir in ai_dirs:
        out_dir = DESKTOP / f"fcm_comparison_results_{Path(ai_dir).name}"
        score_standard_batch(
            ai_base_dir=ai_dir,
            output_base_dir=str(out_dir),
            scorer=scorer,
        )

    print("\n" + "=" * 80)
    print("ALL BATCHES COMPLETE")
    print("=" * 80)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ACO MODE  (JSON-based AI files against aco_adjacencies/gt)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ACO_ROOT   = DESKTOP / "aco_adjacencies"
ACO_OUT_CSV = DESKTOP / "all_aco_results.csv"

# Dataset folder → display label in the CSV
DS_LABEL = {
    "biodiversity": "biodiversity",
    "flpp":         "flpp",
    "osw":          "gulf-osw",
    "red_snapper":  "red-snapper",
}


# ── ID normalisation ──────────────────────────────────────────────────────────

def _normalize_bio(name: str) -> str:
    """BD042 (1) → BD042,  BD027(1) → BD027,  BD064_ → BD064"""
    name = re.sub(r"\s*\(\d+\)$", "", name.strip())
    name = re.sub(r"\(\d+\)$",    "", name)
    return name.rstrip("_").strip()


def _normalize_osw(name: str) -> str:
    """'IEA-Wind CM - AE' → 'AE',  'IEA-Wind CM -JMF' → 'JMF'"""
    if " - " in name:
        return name.split(" - ")[-1].strip()
    if " -" in name:
        return name.split(" -")[-1].strip()
    return name.strip()


def _normalize_rs(name: str) -> str:
    """357392_BeFa_7_26_21 → BeFa  /  BeRa_4_5_23_R2 → BeRa"""
    parts = name.split("_")
    return parts[1] if len(parts) > 1 and parts[0].isdigit() else parts[0]


def _aco_gt_id(raw_name: str, dataset: str) -> str:
    if dataset == "biodiversity":
        return _normalize_bio(raw_name)
    if dataset == "osw":
        return _normalize_osw(raw_name)
    if dataset == "red_snapper":
        return _normalize_rs(raw_name)
    return raw_name  # flpp: exact match


# ── ACO JSON → adjacency DataFrame ───────────────────────────────────────────

def aco_json_to_df(json_path: Path) -> pd.DataFrame:
    """Convert an ACO FCM JSON file to an adjacency DataFrame."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    id2concept = {n["id"]: n.get("concepts", n["id"]) for n in data.get("nodes", [])}
    edges = data.get("edges", [])

    concepts: set = set()
    for e in edges:
        concepts.add(id2concept.get(e["source"], e["source"]))
        concepts.add(id2concept.get(e["target"], e["target"]))
    concepts = sorted(concepts) or ["empty_graph"]

    mat = pd.DataFrame(0.0, index=concepts, columns=concepts)
    for e in edges:
        src = id2concept.get(e["source"], e["source"])
        tgt = id2concept.get(e["target"], e["target"])
        mat.loc[src, tgt] = float(e.get("weight", 0.0))
    return mat


# ── Score one (model, dataset) combination ───────────────────────────────────

def score_aco_model_dataset(
    model_dir_name: str,
    model_short: str,
    dataset: str,
    scorer: ScoreCalculator,
    gt_lookup: dict,
) -> list:
    """Score all participants for one model × dataset. Returns list of row dicts."""
    ai_ds_dir = ACO_ROOT / model_dir_name / dataset
    if not ai_ds_dir.exists():
        print(f"  SKIP {model_dir_name}/{dataset}: directory not found")
        return []

    results = []
    participants = sorted(p for p in ai_ds_dir.iterdir() if p.is_dir())
    print(f"  Scoring {model_dir_name}/{dataset}: {len(participants)} participants")

    for p_dir in participants:
        pid_raw = p_dir.name
        gt_key  = _aco_gt_id(pid_raw, dataset)

        if gt_key not in gt_lookup:
            print(f"    SKIP {pid_raw!r} -> GT key {gt_key!r} not found")
            continue

        json_files = list(p_dir.glob("*_fcm.json"))
        if not json_files:
            print(f"    SKIP {pid_raw!r}: no _fcm.json found")
            continue
        json_path = json_files[0]

        try:
            gt_df  = load_matrix_from_file(str(gt_lookup[gt_key]))
            aco_df = aco_json_to_df(json_path)
        except Exception as ex:
            print(f"    ERROR loading {pid_raw!r}: {ex}")
            continue

        if aco_df.shape[0] == 0 or gt_df.shape[0] == 0:
            print(f"    SKIP {pid_raw!r}: empty matrix")
            continue

        try:
            scorer.data = gt_key
            result = scorer.calculate_scores(gt_df, aco_df)
        except Exception as ex:
            print(f"    ERROR scoring {pid_raw!r}: {ex}")
            continue

        row_in = result.iloc[0]
        results.append({
            "dataset_name":        DS_LABEL[dataset],
            "interview_file_name": gt_key,
            "model_name":          model_short,
            "F1":                  row_in["F1"],
            "Jaccard":             row_in["Jaccard"],
            "TP":                  row_in["TP"],
            "PP":                  row_in["PP"],
            "FP":                  row_in["FP"],
            "FN":                  row_in["FN"],
            "threshold":           row_in["threshold"],
            "tp_scale":            row_in["tp_scale"],
            "pp_scale":            row_in["pp_scale"],
            "fcm1_nodes":          row_in["fcm1_nodes"],
            "fcm1_edges":          row_in["fcm1_edges"],
            "fcm2_nodes":          row_in["fcm2_nodes"],
            "fcm2_edges":          row_in["fcm2_edges"],
        })
        print(f"    OK {gt_key}: F1={row_in['F1']:.4f}")

    return results


def cmd_aco(args):
    """Entry point for `python score.py aco ...`."""
    out_csv = Path(args.output) if args.output else ACO_OUT_CSV

    # model_key → display label
    models = []
    for m in args.models:
        # Accept "aco_mistral" or "aco-mistral" as key; store clean dir name
        key   = m.replace("-", "_")    # "aco-mistral" → "aco_mistral"
        label = m.replace("_", "-")    # "aco_mistral" → "aco-mistral"
        models.append((key, label))

    print("=" * 80)
    print("FCM SCORING  –  ACO (JSON) mode")
    print("=" * 80)
    print(f"Models:  {[lb for _, lb in models]}")
    print(f"Output:  {out_csv}")
    print(f"Model:   {EMBED_MODEL}")
    print(f"Params:  threshold={THRESHOLD}, tp_scale={TP_SCALE}, pp_scale={PP_SCALE}")
    print("=" * 80)

    # Build GT lookup tables: {dataset → {id_stem → Path}}
    gt_lookups = {}
    for ds in DS_LABEL:
        gt_dir = ACO_ROOT / "gt" / ds
        gt_lookups[ds] = (
            {p.stem: p for p in gt_dir.glob("*.csv")} if gt_dir.exists() else {}
        )
        print(f"  GT {ds}: {len(gt_lookups[ds])} participants")

    print("\nLoading embedding model …")
    scorer = ScoreCalculator(
        threshold=THRESHOLD,
        model_name=EMBED_MODEL,
        data="aco",
        tp_scale=TP_SCALE,
        pp_scale=PP_SCALE,
    )
    print("Model loaded.\n")

    all_new_rows = []
    model_labels_scored = []
    for model_dir, model_label in models:
        print(f"\n=== {model_label} ===")
        model_labels_scored.append(model_label)
        for ds in DS_LABEL:
            rows = score_aco_model_dataset(
                model_dir, model_label, ds, scorer, gt_lookups[ds]
            )
            all_new_rows.extend(rows)
            print(f"  -> {len(rows)} rows scored for {ds}")

    if not all_new_rows:
        print("No results to append.")
        return

    new_df = pd.DataFrame(all_new_rows)
    print(f"\nNew rows total: {len(new_df)}")
    print(new_df.groupby(["model_name", "dataset_name"]).size().to_string())

    # Append to (or create) the output CSV, replacing prior rows for the same models
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        print(f"\nExisting rows: {len(existing)}")
        existing = existing[~existing["model_name"].isin(model_labels_scored)].copy()
    else:
        existing = pd.DataFrame(columns=new_df.columns)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"Saved {len(combined)} rows -> {out_csv}")
    print("Model names now:", sorted(combined["model_name"].unique()))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified FCM scoring: standard (CSV) or aco (JSON) mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ── standard ──────────────────────────────────────────────────────────────
    std = subparsers.add_parser(
        "standard",
        help="Score CSV-based AI batches against fcm_gt ground truth.",
    )
    std.add_argument(
        "ai_dirs",
        nargs="*",
        metavar="AI_DIR",
        help=(
            "One or more AI batch directories to score. "
            "Defaults to the three 2026-03-03 no-reasoning batches."
        ),
    )

    # ── aco ───────────────────────────────────────────────────────────────────
    aco = subparsers.add_parser(
        "aco",
        help="Score ACO JSON FCMs against aco_adjacencies/gt ground truth.",
    )
    aco.add_argument(
        "--models",
        nargs="+",
        default=["aco_mistral", "aco_qwen"],
        metavar="MODEL",
        help="ACO model directory name(s) under aco_adjacencies/ (default: aco_mistral aco_qwen).",
    )
    aco.add_argument(
        "--output",
        default=None,
        metavar="CSV_PATH",
        help=f"Output CSV path (default: {ACO_OUT_CSV}).",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.subcommand == "standard":
        cmd_standard(args)
    elif args.subcommand == "aco":
        cmd_aco(args)
