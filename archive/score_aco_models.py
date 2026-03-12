"""
score_aco_models.py
Score aco_mistral and aco_qwen FCMs against GT, then append to all_aco_results.csv.

ACO JSON structure:
  aco_adjacencies/{model}/{dataset}/{participant_id}/{participant_id}_fcm.json
GT CSV structure:
  aco_adjacencies/gt/{dataset}/{short_id}.csv

Dataset ID extraction:
  biodiversity  : participant_id = 'BD001'         → gt short = 'BD001'
  flpp          : participant_id = '100'            → gt short = '100'
  osw           : participant_id = 'IEA-Wind CM - AE' → gt short = split(' - ')[-1]
  red_snapper   : participant_id = '357392_BeFa_'  → gt short = split('_')[1]
"""
import sys, os
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score_fcms import ScoreCalculator, score_fcm_with_scorer

# ── Paths ──────────────────────────────────────────────────────────────────────
ACO_ADJ_DIR = Path(r'C:\Users\Nbrug\Desktop\aco_adjacencies')
GT_DIR      = ACO_ADJ_DIR / 'gt'
ALL_ACO_CSV = Path(r'C:\Users\Nbrug\Desktop\all_aco_results.csv')
SCRATCH_DIR = Path(r'C:\Users\Nbrug\Desktop\aco_score_scratch')
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Scorer settings (same as other methods) ───────────────────────────────────
EMBED_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
THRESHOLD   = 0.6
TP_SCALE    = 1.0
PP_SCALE    = 0.6

# ── Dataset name mapping (folder → dataset_name in CSV) ──────────────────────
DS_NAME_MAP = {
    'biodiversity': 'biodiversity',
    'flpp':         'flpp',
    'osw':          'gulf-osw',
    'red_snapper':  'red-snapper',
}


def participant_to_gt_stem(dataset_folder: str, participant_id: str) -> str:
    """Return the GT CSV stem (no extension) for a given participant folder name."""
    if dataset_folder in ('biodiversity', 'flpp'):
        return participant_id                         # BD001 / 100 / etc.
    elif dataset_folder == 'osw':
        # 'IEA-Wind CM - AE'  →  'AE'
        # 'GoM-IEA Wind CM - SM' → 'SM'
        return participant_id.split(' - ')[-1].strip()
    elif dataset_folder == 'red_snapper':
        # '357392_BeFa_7_26_21' → 'BeFa'
        parts = participant_id.split('_')
        return parts[1] if len(parts) > 1 else participant_id
    return participant_id


def interview_file_name(dataset_folder: str, participant_id: str) -> str:
    """Return the interview_file_name value for the results CSV."""
    return participant_to_gt_stem(dataset_folder, participant_id)


def find_fcm_json(participant_dir: Path) -> Path | None:
    """Find the *_fcm.json in a participant directory."""
    jsons = [f for f in participant_dir.iterdir()
             if f.suffix == '.json' and f.stem.endswith('_fcm')]
    if not jsons:
        jsons = list(participant_dir.glob('*.json'))
    return jsons[0] if jsons else None


def collect_pairs(model_key: str):
    """
    Returns list of dicts:
      {dataset_folder, dataset_name, participant_id, interview_id, gt_csv, ai_json}
    """
    model_dir = ACO_ADJ_DIR / model_key
    pairs = []
    for ds_dir in sorted(model_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_folder = ds_dir.name           # 'biodiversity', 'flpp', ...
        ds_name   = DS_NAME_MAP.get(ds_folder, ds_folder)
        gt_ds_dir = GT_DIR / ds_folder

        for part_dir in sorted(ds_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            pid        = part_dir.name
            gt_stem    = participant_to_gt_stem(ds_folder, pid)
            interview  = interview_file_name(ds_folder, pid)
            gt_csv     = gt_ds_dir / f'{gt_stem}.csv'
            ai_json    = find_fcm_json(part_dir)

            if not gt_csv.exists():
                print(f'  WARN: GT not found → {gt_csv}')
                continue
            if ai_json is None:
                print(f'  WARN: No FCM JSON in {part_dir}')
                continue

            pairs.append(dict(
                dataset_folder=ds_folder,
                dataset_name=ds_name,
                participant_id=pid,
                interview_id=interview,
                gt_csv=gt_csv,
                ai_json=ai_json,
            ))
    return pairs


def score_model(model_key: str, model_label: str, scorer: ScoreCalculator) -> pd.DataFrame:
    """Score all participants for one ACO model, return DataFrame of results."""
    print(f'\n{"="*60}')
    print(f'Scoring: {model_label}  ({model_key})')
    print(f'{"="*60}')
    pairs = collect_pairs(model_key)
    print(f'Found {len(pairs)} AI–GT pairs')

    rows = []
    for i, p in enumerate(pairs):
        pid  = p['participant_id']
        ds   = p['dataset_name']
        print(f'  [{i+1}/{len(pairs)}] {ds} / {pid}', end=' ... ', flush=True)
        try:
            result = score_fcm_with_scorer(
                fcm1_path=str(p['gt_csv']),
                fcm2_path=str(p['ai_json']),
                scorer=scorer,
                output_dir=str(SCRATCH_DIR),
                output_format='csv',
                verbose=False,
            )
            r = result.iloc[0]
            rows.append({
                'dataset_name':        ds,
                'interview_file_name': p['interview_id'],
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
            print(f'F1={r["F1"]:.3f}')
        except Exception as e:
            print(f'ERROR: {e}')

    return pd.DataFrame(rows)


def main():
    # Init a single scorer (reused across all pairs)
    print('Loading embedding model …')
    scorer = ScoreCalculator(
        threshold=THRESHOLD,
        model_name=EMBED_MODEL,
        data='aco',
        tp_scale=TP_SCALE,
        pp_scale=PP_SCALE,
    )
    print('Model loaded.\n')

    all_new = []
    for model_key, model_label in [('aco_mistral', 'aco-mistral'),
                                    ('aco_qwen',    'aco-qwen')]:
        df = score_model(model_key, model_label, scorer)
        all_new.append(df)
        print(f'\n  {model_label}: {len(df)} rows scored')
        print(f'  Mean F1 = {df["F1"].mean():.4f}')

    new_df = pd.concat(all_new, ignore_index=True)

    # Load existing and append (drop any prior aco-mistral / aco-qwen to be safe)
    existing = pd.read_csv(ALL_ACO_CSV)
    existing = existing[~existing['model_name'].isin(['aco-mistral', 'aco-qwen'])].copy()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(ALL_ACO_CSV, index=False)

    print(f'\n{"="*60}')
    print(f'Updated all_aco_results.csv  →  {combined.shape[0]} rows total')
    print(f'Models now in file: {sorted(combined["model_name"].unique())}')
    print('Done.')


if __name__ == '__main__':
    main()
