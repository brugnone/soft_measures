# score_aco_new_models.py
# Score ACO FCMs for: aco_gpt_5_mini, aco_gpt_52, aco_gemini_25_flash
# Runs in serial (one model at a time, one dataset at a time).
#
# At the end, the three new results are combined with:
#   - Desktop/aco_gemini_3_flash_results.csv   (aco-gemini-3-flash)
#   - Desktop/all_aco_results.csv              (only aco-mistral and aco-qwen rows)
# and saved to Desktop/all_aco_results_final.csv
#
# Participant folder naming (same for all three models):
#   biodiversity  : BD001, BD006, ...          → GT stem exact match
#   flpp          : 100, 101, ...              → GT stem exact match
#   osw           : IEA-Wind-CM-AE, GoM-IEA-Wind-CM-SM, ...
#                   → last dash-segment (AE, SM, ...); 3 extras skipped if no GT
#   red_snapper   : BeFa, BeRa, ...           → GT stem exact match

import re, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_fcms import ScoreCalculator, load_matrix_from_file

# ── Paths ────────────────────────────────────────────────────────────────────
ACO_ROOT = Path(r'C:\Users\Nbrug\Desktop\aco_adjacencies')
DESKTOP  = Path(r'C:\Users\Nbrug\Desktop')

# Models to score: (folder_name, model_label, output_csv)
MODELS = [
    ('aco_gpt_5_mini',      'aco-gpt-5-mini',        DESKTOP / 'aco_gpt_5_mini_results.csv'),
    ('aco_gpt_52',          'aco-gpt-5.2',            DESKTOP / 'aco_gpt_52_results.csv'),
    ('aco_gemini_25_flash',  'aco-gemini-2.5-flash',  DESKTOP / 'aco_gemini_25_flash_results.csv'),
]

# ── Dataset label mapping ─────────────────────────────────────────────────────
DS_LABEL = {
    'biodiversity': 'biodiversity',
    'flpp':         'flpp',
    'osw':          'gulf-osw',
    'red_snapper':  'red-snapper',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def gt_key_for(participant_name: str, dataset: str, gt_stems: set) -> str | None:
    """Return the GT stem matching this participant folder name, or None."""
    if participant_name in gt_stems:
        return participant_name
    if dataset == 'osw':
        last = participant_name.rsplit('-', 1)[-1].strip()
        if last in gt_stems:
            return last
    if dataset == 'biodiversity':
        cleaned = re.sub(r'\s*\(\d+\)$', '', participant_name.strip()).rstrip('_').strip()
        if cleaned in gt_stems:
            return cleaned
    return None


def aco_json_to_df(json_path: Path) -> pd.DataFrame:
    """Convert ACO FCM JSON to adjacency DataFrame using concept labels."""
    data = json.loads(json_path.read_text(encoding='utf-8'))
    id2concept = {n['id']: n.get('concepts', n['id']) for n in data.get('nodes', [])}
    edges = data.get('edges', [])
    concepts: set = set()
    for e in edges:
        concepts.add(id2concept.get(e['source'], e['source']))
        concepts.add(id2concept.get(e['target'], e['target']))
    concepts_list = sorted(concepts) or ['empty_graph']
    mat = pd.DataFrame(0.0, index=concepts_list, columns=concepts_list)
    for e in edges:
        src = id2concept.get(e['source'], e['source'])
        tgt = id2concept.get(e['target'], e['target'])
        mat.loc[src, tgt] = float(e.get('weight', 0.0))
    return mat


def score_dataset(dataset: str, model_dir: Path, model_label: str,
                  scorer: ScoreCalculator, gt_lookup: dict) -> list:
    ai_ds_dir = model_dir / dataset
    if not ai_ds_dir.exists():
        print(f'  SKIP {dataset}: directory not found')
        return []

    gt_stems = set(gt_lookup.keys())
    participants = sorted(p for p in ai_ds_dir.iterdir() if p.is_dir())
    print(f'\n  [{dataset}] {len(participants)} participants')

    results = []
    for p_dir in participants:
        pid_raw = p_dir.name
        key = gt_key_for(pid_raw, dataset, gt_stems)

        if key is None:
            print(f'    SKIP {pid_raw!r}: no GT match')
            continue

        json_files = list(p_dir.glob('*_fcm.json'))
        if not json_files:
            print(f'    SKIP {pid_raw!r}: no _fcm.json')
            continue
        json_path = json_files[0]

        try:
            gt_df  = load_matrix_from_file(str(gt_lookup[key]))
            aco_df = aco_json_to_df(json_path)
        except Exception as ex:
            print(f'    ERROR loading {pid_raw!r}: {ex}')
            continue

        if aco_df.shape[0] == 0 or gt_df.shape[0] == 0:
            print(f'    SKIP {pid_raw!r}: empty matrix')
            continue

        try:
            scorer.data = key
            result = scorer.calculate_scores(gt_df, aco_df)
        except Exception as ex:
            print(f'    ERROR scoring {pid_raw!r}: {ex}')
            continue

        r = result.iloc[0]
        results.append({
            'dataset_name':        DS_LABEL[dataset],
            'interview_file_name': key,
            'model_name':          model_label,
            'F1':                  r['F1'],
            'Jaccard':             r['Jaccard'],
            'TP':                  r['TP'],
            'PP':                  r['PP'],
            'FP':                  r['FP'],
            'FN':                  r['FN'],
            'threshold':           r['threshold'],
            'tp_scale':            r['tp_scale'],
            'pp_scale':            r['pp_scale'],
            'fcm1_nodes':          r['fcm1_nodes'],
            'fcm1_edges':          r['fcm1_edges'],
            'fcm2_nodes':          r['fcm2_nodes'],
            'fcm2_edges':          r['fcm2_edges'],
        })
        print(f'    OK {key}: F1={r["F1"]:.4f}')

    return results


def score_model(folder_name: str, model_label: str, out_csv: Path,
                scorer: ScoreCalculator, gt_lookups: dict) -> pd.DataFrame:
    model_dir = ACO_ROOT / folder_name
    print(f'\n{"="*60}')
    print(f'Scoring model: {model_label}  ({folder_name})')
    print(f'{"="*60}')

    all_rows = []
    for ds in DS_LABEL:
        rows = score_dataset(ds, model_dir, model_label, scorer, gt_lookups[ds])
        all_rows.extend(rows)
        print(f'  -> {len(rows)} rows scored for {ds}')

    if not all_rows:
        print(f'  WARNING: no results for {model_label}')
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f'\n  Saved {len(df)} rows -> {out_csv.name}')
    print(df.groupby('dataset_name')['F1'].agg(['count', 'mean']).rename(
        columns={'count': 'n', 'mean': 'mean_F1'}).to_string())
    print(f'  Overall mean F1: {df["F1"].mean():.4f}')
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Build GT lookup tables once
    gt_lookups = {}
    for ds in DS_LABEL:
        gt_dir = ACO_ROOT / 'gt' / ds
        gt_lookups[ds] = {p.stem: p for p in gt_dir.glob('*.csv')} if gt_dir.exists() else {}
        print(f'GT {ds}: {len(gt_lookups[ds])} participants')

    print('\nLoading embedding model...')
    scorer = ScoreCalculator(
        threshold=0.6,
        model_name='Qwen/Qwen3-Embedding-0.6B',
        data='aco',
        tp_scale=1.0,
        pp_scale=0.6,
    )
    print('Model loaded.\n')

    # Score all three models in serial
    new_dfs = []
    for folder_name, model_label, out_csv in MODELS:
        # Clear the scorer cache between models to avoid stale entries
        ScoreCalculator.cache.clear()
        df = score_model(folder_name, model_label, out_csv, scorer, gt_lookups)
        if not df.empty:
            new_dfs.append(df)

    if not new_dfs:
        print('\nNo results produced — aborting combine step.')
        return

    # ── Combine into all_aco_results_final.csv ────────────────────────────────
    print(f'\n{"="*60}')
    print('Combining into all_aco_results_final.csv ...')

    # 1. aco-mistral and aco-qwen from existing all_aco_results.csv
    existing_path = DESKTOP / 'all_aco_results.csv'
    existing = pd.read_csv(existing_path)
    base = existing[existing['model_name'].isin(['aco-mistral', 'aco-qwen'])].copy()
    print(f'  From all_aco_results.csv (aco-mistral + aco-qwen): {len(base)} rows')

    # 2. aco-gemini-3-flash from separate CSV
    g3f_path = DESKTOP / 'aco_gemini_3_flash_results.csv'
    g3f = pd.read_csv(g3f_path)
    print(f'  From aco_gemini_3_flash_results.csv: {len(g3f)} rows')

    # 3. Newly scored models
    new_combined = pd.concat(new_dfs, ignore_index=True)
    print(f'  Newly scored (gpt-5-mini, gpt-5.2, gemini-2.5-flash): {len(new_combined)} rows')

    # Align columns to base schema
    cols = list(base.columns)
    # Add any missing columns with NaN
    for df in [g3f, new_combined]:
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

    final = pd.concat([base, g3f[cols], new_combined[cols]], ignore_index=True)
    out_final = DESKTOP / 'all_aco_results_final.csv'
    final.to_csv(out_final, index=False)

    print(f'\n{"="*60}')
    print(f'FINAL: {len(final)} rows -> {out_final.name}')
    print(final.groupby('model_name')['F1'].agg(['count', 'mean']).rename(
        columns={'count': 'n', 'mean': 'mean_F1'}).sort_index().to_string())


if __name__ == '__main__':
    main()
