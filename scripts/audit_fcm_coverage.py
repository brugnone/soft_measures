"""
audit_fcm_coverage.py
=====================
Counts AI-generated FCMs per method/dataset and compares against the unified GT
to report how many are present and how many are missing.
"""
from pathlib import Path

NON_ACO_ROOT = Path(r"C:\Users\Nbrug\Desktop\fcm_adjacency_organised")
ACO_ROOT     = Path(r"C:\Users\Nbrug\Desktop\aco_adjacencies")
UNIFIED_GT   = Path(r"C:\Users\Nbrug\Desktop\gt_unified")

DATASETS = ["biodiversity", "flpp", "osw", "red_snapper"]

# ── GT stems per dataset ──────────────────────────────────────────────────────
gt = {ds: {f.stem for f in (UNIFIED_GT / ds).glob("*.csv")} for ds in DATASETS}
gt_total = sum(len(v) for v in gt.values())

print(f"Unified GT totals: {gt_total}")
for ds in DATASETS:
    print(f"  {ds}: {len(gt[ds])}")
print()

# ── Collect method folders ────────────────────────────────────────────────────
methods = []
for d in sorted(NON_ACO_ROOT.iterdir()):
    if d.name != "gt":
        methods.append(("non-aco", d.name, d))
for d in sorted(ACO_ROOT.iterdir()):
    if d.name != "gt":
        methods.append(("aco", d.name, d))

# ── Header ────────────────────────────────────────────────────────────────────
HDR = f"{'Method':<26} {'Type':<8}  {'bio':>4} {'flpp':>4} {'osw':>4} {'rs':>4}  {'matched':>8} {'missing':>8} {'extra':>6}"
print(HDR)
print("-" * len(HDR))

all_rows = []
for mtype, mname, mdir in methods:
    matched  = {}   # AI stems that exist in GT
    missing  = {}   # GT stems not found in AI
    extra    = {}   # AI stems not in GT

    for ds in DATASETS:
        ds_dir = mdir / ds
        if not ds_dir.exists():
            matched[ds] = set()
            missing[ds] = set(gt[ds])
            extra[ds]   = set()
            continue

        if mtype == "aco":
            participant_dirs = [p for p in ds_dir.iterdir() if p.is_dir()]
            found_stems = set()
            for p in participant_dirs:
                name = p.name
                if ds == "osw":
                    if " - " in name:
                        stem = name.rsplit(" - ", 1)[-1].strip()
                    else:
                        stem = name.rsplit("-", 1)[-1].strip()
                elif ds == "red_snapper" and "_" in name:
                    parts = name.split("_")
                    stem = parts[1] if len(parts) > 1 else name
                else:
                    stem = name
                found_stems.add(stem)
        else:
            found_stems = {f.stem for f in ds_dir.glob("*.csv")}

        matched[ds] = gt[ds] & found_stems        # in both GT and AI
        missing[ds] = gt[ds] - found_stems        # in GT but not AI
        extra[ds]   = found_stems - gt[ds]        # in AI but not GT

    total_matched = sum(len(v) for v in matched.values())
    total_missing = sum(len(v) for v in missing.values())
    total_extra   = sum(len(v) for v in extra.values())

    bio  = len(matched["biodiversity"])
    flpp = len(matched["flpp"])
    osw  = len(matched["osw"])
    rs   = len(matched["red_snapper"])
    print(f"{mname:<26} {mtype:<8}  {bio:>4} {flpp:>4} {osw:>4} {rs:>4}  {total_matched:>7} {total_missing:>8} {total_extra:>6}")
    all_rows.append((mname, mtype, matched, missing, extra, total_matched, total_missing, total_extra))

# ── Detail: missing IDs per method ───────────────────────────────────────────
print()
print("=" * 60)
print("DETAIL (only methods with gaps)")
print("=" * 60)
for mname, mtype, matched, missing, extra, total_matched, total_missing, total_extra in all_rows:
    if total_missing == 0 and total_extra == 0:
        continue
    print(f"\n{mname}  (matched={total_matched}/{gt_total}  missing={total_missing}  extra={total_extra}):")
    for ds in DATASETS:
        m = sorted(missing[ds])
        if m:
            print(f"  MISSING {ds} ({len(m)}): {m}")
    for ds in DATASETS:
        e = sorted(extra[ds])
        if e:
            print(f"  EXTRA   {ds} ({len(e)}): {e}")
