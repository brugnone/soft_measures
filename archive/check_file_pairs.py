"""
Quick diagnostic to check what file pairs are being found.
"""
import os
from pathlib import Path

# Directory mappings
ai_base_dir = r'C:\Users\Nbrug\Desktop\fcm_ai_20260225'
gt_base_dir = r'C:\Users\Nbrug\Desktop\fcm_gt'

dataset_dir_mapping = {
    'biodiversity': {'ai': 'Biodiversity', 'gt': 'biodiversity_gt'},
    'flpp': {'ai': 'FLPP', 'gt': 'flpp_gt'},
    'osw': {'ai': 'Gulf OSW', 'gt': 'osw_gt'},
    'red_snapper': {'ai': 'Red snapper', 'gt': 'red_snapper_gt'}
}

print("=" * 80)
print("CHECKING FILE PAIRS")
print("=" * 80)

for dataset_name, dirs in dataset_dir_mapping.items():
    ai_dir = os.path.join(ai_base_dir, dirs['ai'])
    gt_dir = os.path.join(gt_base_dir, dirs['gt'])
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  AI dir: {ai_dir}")
    print(f"  Exists: {os.path.exists(ai_dir)}")
    
    if os.path.exists(ai_dir):
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
                    if 'scoring_results' in file_id.lower() or 'scoring_results' in file.lower():
                        continue
                    
                    ai_files[file_id] = os.path.join(root, file)
        print(f"  AI CSV files: {len(ai_files)}")
        if ai_files:
            print(f"  Sample AI files: {list(ai_files.keys())[:3]}")
    
    print(f"  GT dir: {gt_dir}")
    print(f"  Exists: {os.path.exists(gt_dir)}")
    
    if os.path.exists(gt_dir):
        gt_files = {}
        for root, dirs_list, files in os.walk(gt_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_id = Path(file).stem
                    gt_files[file_id] = os.path.join(root, file)
        print(f"  GT CSV files: {len(gt_files)}")
        if gt_files:
            print(f"  Sample GT files: {list(gt_files.keys())[:3]}")
    
    if os.path.exists(ai_dir) and os.path.exists(gt_dir):
        matching_ids = set(ai_files.keys()) & set(gt_files.keys())
        print(f"  Matching pairs: {len(matching_ids)}")
        if matching_ids:
            print(f"  Sample matches: {list(sorted(matching_ids))[:3]}")

print("\n" + "=" * 80)
