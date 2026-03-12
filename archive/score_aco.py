# score_aco.py
# Score aco_mistral and aco_qwen against GT CSVs, then append to all_aco_results.csv.
# ACO adjacencies root: Desktop/aco_adjacencies/
#   gt/{biodiversity,flpp,osw,red_snapper}/<participant>.csv
#   aco_mistral/{...}/<participant>/<participant>_fcm.json
#   aco_qwen/{...}/<participant>/<participant>_fcm.json
import re, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_fcms import ScoreCalculator, load_matrix_from_file

# ── Paths ───────────────────────────────────────────────────────────────────────
ACO_ROOT   = Path(r'C:\Users\Nbrug\Desktop\aco_adjacencies')
OUT_CSV    = Path(r'C:\Users\Nbrug\Desktop\all_aco_results.csv')

# ── ID normalisation ────────────────────────────────────────────────────────────

def normalize_bio(name: str) -> str:
    """BD042 (1) → BD042,  BD027(1) → BD027,  BD064_ → BD064"""
    name = re.sub(r'\s*\(\d+\)$', '', name.strip())
    name = re.sub(r'\(\d+\)$',    '', name)
    return name.rstrip('_').strip()

def normalize_osw(name: str) -> str:
    """'IEA-Wind CM - AE' → 'AE',  'IEA-Wind CM -JMF' → 'JMF'"""
    # try ' - ' first, then ' -'
    if ' - ' in name:
        return name.split(' - ')[-1].strip()
    if ' -' in name:
        return name.split(' -')[-1].strip()
    return name.strip()

def normalize_rs(name: str) -> str:
    """
    '357392_BeFa_7_26_21' -> 'BeFa'  (numeric prefix: take token[1])
    'BeRa_4_5_23_R2'      -> 'BeRa'  (alpha prefix:   take token[0])
    """
    parts = name.split('_')
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1]
    return parts[0]

def gt_id(raw_name: str, dataset: str) -> str:
    if dataset == 'biodiversity':
        return normalize_bio(raw_name)
    if dataset == 'osw':
        return normalize_osw(raw_name)
    if dataset == 'red_snapper':
        return normalize_rs(raw_name)
    return raw_name  # flpp: exact match

# ── ACO-JSON → adjacency DataFrame ─────────────────────────────────────────────

def aco_json_to_df(json_path: Path) -> pd.DataFrame:
    """Convert ACO FCM JSON to adjacency DataFrame using concept labels."""
    data = json.loads(json_path.read_text(encoding='utf-8'))
    # id → concept label map
    id2concept = {n['id']: n.get('concepts', n['id']) for n in data.get('nodes', [])}
    edges = data.get('edges', [])
    # collect all concept nodes
    concepts = set()
    for e in edges:
        concepts.add(id2concept.get(e['source'], e['source']))
        concepts.add(id2concept.get(e['target'], e['target']))
    concepts = sorted(concepts) or ['empty_graph']
    mat = pd.DataFrame(0.0, index=concepts, columns=concepts)
    for e in edges:
        src = id2concept.get(e['source'], e['source'])
        tgt = id2concept.get(e['target'], e['target'])
        w   = float(e.get('weight', 0.0))
        mat.loc[src, tgt] = w
    return mat

# ── Dataset label mapping ───────────────────────────────────────────────────────
DS_LABEL = {
    'biodiversity': 'biodiversity',
    'flpp':         'flpp',
    'osw':          'gulf-osw',
    'red_snapper':  'red-snapper',
}

# ── Score one (model, dataset) pair ────────────────────────────────────────────

def score_model_dataset(model_dir_name, model_short, dataset, scorer, gt_lookup):
    """Score all participants for one model × dataset. Returns list of result dicts."""
    ai_ds_dir = ACO_ROOT / model_dir_name / dataset
    if not ai_ds_dir.exists():
        print(f'  SKIP {model_dir_name}/{dataset}: directory not found')
        return []

    results = []
    participants = sorted(p for p in ai_ds_dir.iterdir() if p.is_dir())
    print(f'  Scoring {model_dir_name}/{dataset}: {len(participants)} participants')

    for p_dir in participants:
        pid_raw = p_dir.name
        gt_key  = gt_id(pid_raw, dataset)

        if gt_key not in gt_lookup:
            print(f'    SKIP {pid_raw!r} -> GT key {gt_key!r} not found')
            continue

        # Find ACO FCM json
        json_files = list(p_dir.glob('*_fcm.json'))
        if not json_files:
            print(f'    SKIP {pid_raw!r}: no _fcm.json found')
            continue
        json_path = json_files[0]

        # Load matrices
        try:
            gt_df  = load_matrix_from_file(str(gt_lookup[gt_key]))
            aco_df = aco_json_to_df(json_path)
        except Exception as ex:
            print(f'    ERROR loading {pid_raw!r}: {ex}')
            continue

        if aco_df.shape[0] == 0 or gt_df.shape[0] == 0:
            print(f'    SKIP {pid_raw!r}: empty matrix')
            continue

        # Score
        try:
            scorer.data = gt_key
            result = scorer.calculate_scores(gt_df, aco_df)
        except Exception as ex:
            print(f'    ERROR scoring {pid_raw!r}: {ex}')
            continue

        row_in = result.iloc[0]
        results.append({
            'dataset_name':       DS_LABEL[dataset],
            'interview_file_name': gt_key,
            'model_name':         model_short,
            'F1':                 row_in['F1'],
            'Jaccard':            row_in['Jaccard'],
            'TP':                 row_in['TP'],
            'PP':                 row_in['PP'],
            'FP':                 row_in['FP'],
            'FN':                 row_in['FN'],
            'threshold':          row_in['threshold'],
            'tp_scale':           row_in['tp_scale'],
            'pp_scale':           row_in['pp_scale'],
            'fcm1_nodes':         row_in['fcm1_nodes'],
            'fcm1_edges':         row_in['fcm1_edges'],
            'fcm2_nodes':         row_in['fcm2_nodes'],
            'fcm2_edges':         row_in['fcm2_edges'],
        })
        print(f'    OK {gt_key}: F1={row_in["F1"]:.4f}')

    return results

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    # Build GT lookup tables: {dataset → {id_stem → Path}}
    gt_lookups = {}
    for ds in DS_LABEL:
        gt_dir = ACO_ROOT / 'gt' / ds
        gt_lookups[ds] = {p.stem: p for p in gt_dir.glob('*.csv')} if gt_dir.exists() else {}
        print(f'  GT {ds}: {len(gt_lookups[ds])} participants')

    print('\nLoading embedding model...')
    scorer = ScoreCalculator(
        threshold=0.6,
        model_name='Qwen/Qwen3-Embedding-0.6B',
        data='aco',
        tp_scale=1.0,
        pp_scale=0.6,
    )
    print('Model loaded.\n')

    MODELS_TO_SCORE = [
        ('aco_mistral', 'aco-mistral'),
        ('aco_qwen',    'aco-qwen'),
    ]
    DATASETS = list(DS_LABEL.keys())

    all_new_rows = []
    for model_dir, model_short in MODELS_TO_SCORE:
        print(f'\n=== {model_short} ===')
        for ds in DATASETS:
            rows = score_model_dataset(model_dir, model_short, ds, scorer, gt_lookups[ds])
            all_new_rows.extend(rows)
            print(f'  -> {len(rows)} rows scored for {ds}')

    if not all_new_rows:
        print('No results to append.')
        return

    new_df = pd.DataFrame(all_new_rows)
    print(f'\nNew rows total: {len(new_df)}')
    print(new_df.groupby(['model_name', 'dataset_name']).size().to_string())

    # Load existing and append (drop any stale aco-mistral/aco-qwen rows first)
    existing = pd.read_csv(OUT_CSV)
    print(f'\nExisting rows: {len(existing)}')
    existing = existing[~existing['model_name'].isin(['aco-mistral', 'aco-qwen'])].copy()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f'Saved {len(combined)} rows -> {OUT_CSV}')
    print('Model names now:', sorted(combined['model_name'].unique()))

if __name__ == '__main__':
    main()
