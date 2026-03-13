# score_aco_gemini_3_flash.py
# Score aco_gemini_3_flash FCMs against GT CSVs.
# Saves results to Desktop/aco_gemini_3_flash_results.csv (separate from all_aco_results.csv).
#
# Participant folder naming in aco_gemini_3_flash:
#   biodiversity  : BD001, BD006, ...       → GT stem exact match
#   flpp          : 100, 101, ...           → GT stem exact match
#   osw           : IEA-Wind-CM-AE, GoM-IEA-Wind-CM-SM, ...
#                   → try exact match first, then last dash-segment (AE, AM, ...)
#   red_snapper   : BeFa, BeRa, ...        → GT stem exact match (already short)

import re, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_fcms import ScoreCalculator, load_matrix_from_file

# ── Paths ────────────────────────────────────────────────────────────────────
ACO_ROOT    = Path(r'C:\Users\Nbrug\Desktop\aco_adjacencies')
MODEL_DIR   = ACO_ROOT / 'aco_gemini_3_flash'
MODEL_LABEL = 'aco-gemini-3-flash'
OUT_CSV     = Path(r'C:\Users\Nbrug\Desktop\aco_gemini_3_flash_results.csv')

# ── Dataset name mapping ──────────────────────────────────────────────────────
DS_LABEL = {
    'biodiversity': 'biodiversity',
    'flpp':         'flpp',
    'osw':          'gulf-osw',
    'red_snapper':  'red-snapper',
}

# ── ID normalisation ──────────────────────────────────────────────────────────

def gt_key_for(participant_name: str, dataset: str, gt_stems: set) -> str | None:
    """
    Return the GT stem that matches this participant folder name, or None.
    """
    # Exact match always wins
    if participant_name in gt_stems:
        return participant_name

    if dataset == 'osw':
        # 'IEA-Wind-CM-AE' → 'AE'  (last dash-segment)
        last = participant_name.rsplit('-', 1)[-1].strip()
        if last in gt_stems:
            return last

    if dataset == 'biodiversity':
        # strip trailing underscores / parentheses just in case
        cleaned = re.sub(r'\s*\(\d+\)$', '', participant_name.strip()).rstrip('_').strip()
        if cleaned in gt_stems:
            return cleaned

    return None  # no match found

# ── ACO JSON → adjacency DataFrame ───────────────────────────────────────────

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

# ── Score one dataset ─────────────────────────────────────────────────────────

def score_dataset(dataset: str, scorer: ScoreCalculator, gt_lookup: dict) -> list:
    ai_ds_dir = MODEL_DIR / dataset
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
            'model_name':          MODEL_LABEL,
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Build GT lookup tables
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

    all_rows = []
    for ds in DS_LABEL:
        rows = score_dataset(ds, scorer, gt_lookups[ds])
        all_rows.extend(rows)
        print(f'  -> {len(rows)} rows scored for {ds}')

    if not all_rows:
        print('No results produced.')
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    print(f'\n{"="*60}')
    print(f'Saved {len(df)} rows -> {OUT_CSV}')
    print(df.groupby('dataset_name')['F1'].agg(['count', 'mean']).rename(
        columns={'count': 'n', 'mean': 'mean_F1'}).to_string())
    print(f'\nOverall mean F1: {df["F1"].mean():.4f}')
    print('Done.')


if __name__ == '__main__':
    main()
