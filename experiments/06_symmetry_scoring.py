"""
Experiment 06 — Symmetry scoring of study collections.

Tests different subsets of the available study databases and scores each
collection by all three symmetry definitions from TECHNICAL_REFERENCE §4:

  1. Engine-permutation symmetric (structural): are all Venn regions represented?
  2. Coverage-uniform (statistical): is density proportional to region volume?
  3. KS-balanced (semantic): equal K_H / K_E / K_T representation?

Also asks: how many studies does the collection need to achieve each symmetry type?
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)
_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PKG)

import json, glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

_OUT = Path(__file__).parent.parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

# All study databases with their metadata
ALL_STUDIES = {
    # Set-op studies
    "study_a": {
        "path": "output/db_study_a_artist_consensus",
        "ks_type": "K_H_OVERLAP",
        "set_op": "intersection",
        "engine_a": "mixbox",
        "engine_b": "coloraide_ryb",
        "label": "mixbox+RYB",
    },
    "study_b": {
        "path": "output/db_study_b_physics_vs_artist",
        "ks_type": "K_T",
        "set_op": "difference",
        "engine_a": "spectral",
        "engine_b": "coloraide_ryb",
        "label": "spectral-RYB",
    },
    "study_c": {
        "path": "output/db_study_c_oilpaint_vs_fooddye",
        "ks_type": "K_E",
        "set_op": "difference",
        "engine_a": "kubelka_munk",
        "engine_b": "mixbox",
        "label": "KM-mixbox",
    },
    # Single-engine databases
    "spectral": {"path": "output/db_spectral", "ks_type": "K_H",
                 "set_op": "single", "engine_a": "spectral", "engine_b": None, "label": "spectral"},
    "mixbox":   {"path": "output/db_mixbox",   "ks_type": "K_E",
                 "set_op": "single", "engine_a": "mixbox",   "engine_b": None, "label": "mixbox"},
    "km":       {"path": "output/db_km",        "ks_type": "K_E",
                 "set_op": "single", "engine_a": "kubelka_munk", "engine_b": None, "label": "KM"},
    "ryb":      {"path": "output/db_ryb",       "ks_type": "K_T",
                 "set_op": "single", "engine_a": "coloraide_ryb", "engine_b": None, "label": "RYB"},
}

KS_TYPES_NEEDED = {"K_H", "K_E", "K_T", "K_H_OVERLAP"}  # for KS-balance check


def load_targets(db_path: str, max_targets: int = 2000) -> np.ndarray | None:
    db = Path(_ROOT) / db_path
    if not db.exists():
        return None
    targets = []
    for path in sorted(glob.glob(str(db / "targets" / "*.json")))[:50]:
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, dict) and "targets" in d:
            targets.extend(d["targets"])
        if len(targets) >= max_targets:
            break
    if not targets:
        return None
    arr = np.clip(np.array(targets[:max_targets], dtype=float) / 255.0, 0, 1)
    return arr


def voxelise(pts: np.ndarray, res: int = 28) -> np.ndarray:
    idx = np.clip((pts * res).astype(int), 0, res - 1)
    g = np.zeros((res, res, res), dtype=bool)
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return g


# ── Symmetry scores ──────────────────────────────────────────────────────────

def score_ks_balance(study_keys: list[str]) -> dict:
    """Definition 4.3: are K_H, K_E, K_T all present?"""
    ks_present = {ALL_STUDIES[k]["ks_type"] for k in study_keys}
    required = {"K_H_OVERLAP", "K_E", "K_T"}
    covered = required & ks_present
    score = len(covered) / len(required)
    return {
        "score": score,
        "covered": sorted(covered),
        "missing": sorted(required - ks_present),
        "perfect": score >= 1.0,
    }


def score_engine_permutation(study_keys: list[str], voxels: dict[str, np.ndarray]) -> dict:
    """
    Definition 4.1: are paired asymmetric studies (A→B and B→A) both present?
    Proxy: for each difference study D = engine_a - engine_b, check whether
    the complement (engine_b − engine_a) is also in the collection.

    Also measures volume imbalance between A-only and B-only regions.
    """
    # Find all difference pairs in collection
    diff_studies = [(k, ALL_STUDIES[k]) for k in study_keys
                    if ALL_STUDIES[k]["set_op"] == "difference"]

    paired = 0
    unpaired = 0
    imbalances = []

    for key, meta in diff_studies:
        ea, eb = meta["engine_a"], meta["engine_b"]
        # Look for the complement (eb − ea) in collection
        mirror = next((k for k in study_keys
                       if ALL_STUDIES[k]["set_op"] == "difference"
                       and ALL_STUDIES[k]["engine_a"] == eb
                       and ALL_STUDIES[k]["engine_b"] == ea), None)
        if mirror:
            paired += 1
            # Volume imbalance
            if key in voxels and mirror in voxels:
                va = voxels[key].sum()
                vb = voxels[mirror].sum()
                imb = abs(va - vb) / (va + vb) if (va + vb) else 1.0
                imbalances.append(imb)
        else:
            unpaired += 1

    total_diffs = len(diff_studies)
    pair_score = (paired / total_diffs) if total_diffs else 1.0
    imbalance_score = 1.0 - float(np.mean(imbalances)) if imbalances else (1.0 if paired > 0 else 0.5)

    return {
        "score": (pair_score + imbalance_score) / 2.0,
        "paired_diffs": paired,
        "unpaired_diffs": unpaired,
        "mean_volume_imbalance": float(np.mean(imbalances)) if imbalances else None,
        "perfect": unpaired == 0 and (not imbalances or max(imbalances) < 0.05),
    }


def score_coverage_uniform(study_keys: list[str], voxels: dict[str, np.ndarray]) -> dict:
    """
    Definition 4.2: is the combined density approximately uniform?
    Measure: coefficient of variation (CoV) of per-study voxel volume.
    Lower CoV = more uniform.
    """
    vols = [voxels[k].sum() for k in study_keys if k in voxels]
    if not vols:
        return {"score": 0.0, "cov": float("nan")}
    cov = float(np.std(vols) / np.mean(vols)) if np.mean(vols) > 0 else 1.0
    score = max(0.0, 1.0 - cov)
    return {
        "score": score,
        "cov": cov,
        "volumes": {k: int(voxels[k].sum()) for k in study_keys if k in voxels},
        "perfect": cov < 0.1,
    }


def combined_symmetry_score(ks, ep, cu, weights=(0.4, 0.35, 0.25)) -> float:
    return weights[0] * ks["score"] + weights[1] * ep["score"] + weights[2] * cu["score"]


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # Load available databases
    available = {}
    voxels = {}
    for key, meta in ALL_STUDIES.items():
        pts = load_targets(meta["path"])
        if pts is not None:
            available[key] = pts
            voxels[key] = voxelise(pts)
            print(f"  Loaded {key:12s}: {len(pts)} pts  vol={voxels[key].sum()}")
        else:
            print(f"  NOT FOUND: {key} at {meta['path']}")

    if not available:
        print("No databases found.")
        return

    avail_keys = list(available.keys())
    print(f"\n{len(avail_keys)} databases available: {avail_keys}")

    # --- Score every possible subset ---
    results = []
    for k in range(1, len(avail_keys) + 1):
        for subset in combinations(avail_keys, k):
            subset = list(subset)
            ks  = score_ks_balance(subset)
            ep  = score_engine_permutation(subset, voxels)
            cu  = score_coverage_uniform(subset, voxels)
            total = combined_symmetry_score(ks, ep, cu)
            results.append({
                "k": k,
                "studies": subset,
                "label": "+".join(ALL_STUDIES[s]["label"] for s in subset),
                "ks_balance": ks,
                "engine_permutation": ep,
                "coverage_uniform": cu,
                "combined_score": total,
            })

    results.sort(key=lambda r: (-r["combined_score"], r["k"]))

    # --- Report top-10 subsets ---
    print("\n=== Top 10 Study Collections by Combined Symmetry Score ===")
    print(f"{'k':>3}  {'Combined':>9}  {'KS-bal':>8}  {'Eng-perm':>9}  {'Cov-unif':>9}  Studies")
    for r in results[:10]:
        print(f"{r['k']:>3}  {r['combined_score']:>9.4f}  "
              f"{r['ks_balance']['score']:>8.4f}  "
              f"{r['engine_permutation']['score']:>9.4f}  "
              f"{r['coverage_uniform']['score']:>9.4f}  "
              f"{r['label']}")

    # --- Report specifically for {A, B, C} ---
    abc = [r for r in results if set(r["studies"]) == {"study_a", "study_b", "study_c"}]
    if abc:
        r = abc[0]
        print(f"\n=== The 3 Set-Op Studies {{A, B, C}} ===")
        print(f"  Combined score   : {r['combined_score']:.4f}")
        print(f"  KS-balance       : {r['ks_balance']['score']:.4f}  "
              f"(covered: {r['ks_balance']['covered']}, missing: {r['ks_balance']['missing']})")
        print(f"  Engine-perm sym  : {r['engine_permutation']['score']:.4f}  "
              f"(paired={r['engine_permutation']['paired_diffs']}, "
              f"unpaired={r['engine_permutation']['unpaired_diffs']})")
        print(f"  Coverage-uniform : {r['coverage_uniform']['score']:.4f}  "
              f"(CoV={r['coverage_uniform']['cov']:.4f})")

    # --- How many studies needed to cross threshold? ---
    print("\n=== Minimum k to cross symmetry thresholds ===")
    for thresh in [0.5, 0.7, 0.85, 0.95]:
        min_k = min((r["k"] for r in results if r["combined_score"] >= thresh), default=None)
        if min_k is not None:
            best_at_k = next(r for r in results
                             if r["k"] == min_k and r["combined_score"] >= thresh)
            print(f"  >= {thresh:.2f}: k={min_k}  best={best_at_k['label'][:60]}")
        else:
            print(f"  >= {thresh:.2f}: not achievable with current databases")

    # --- Plot: combined score vs k ---
    by_k = {}
    for r in results:
        by_k.setdefault(r["k"], []).append(r["combined_score"])
    ks = sorted(by_k.keys())
    max_scores = [max(by_k[k]) for k in ks]
    mean_scores = [np.mean(by_k[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, max_scores, "b-o", label="Best subset of size k")
    ax.plot(ks, mean_scores, "g--s", label="Mean over all subsets of size k")
    ax.axhline(0.85, color="red", linestyle=":", label="0.85 threshold")
    ax.set_xlabel("Number of studies k")
    ax.set_ylabel("Combined symmetry score")
    ax.set_title("How many asymmetric studies make a symmetric whole?")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_OUT / "symmetry_score_vs_k.png", dpi=150)
    plt.close()
    print("\n[06] Saved symmetry_score_vs_k.png")

    # --- Save JSON ---
    out_path = _OUT / "symmetry_scoring.json"
    with open(out_path, "w") as f:
        json.dump(results[:50], f, indent=2)
    print(f"[06] Saved {out_path}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Symmetry scoring (Exp 06)")
    _p.add_argument("--studies", nargs="+", metavar="NAME=PATH",
                    help="Override paths for known study keys (key=path pairs)")
    _p.add_argument("--output-dir", default=None, metavar="DIR")
    _args = _p.parse_args()
    if _args.output_dir:
        _OUT = Path(_args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    if _args.studies:
        for _s in _args.studies:
            _name, _, _path = _s.partition("=")
            _name = _name.strip()
            if _name in ALL_STUDIES:
                ALL_STUDIES[_name]["path"] = _path.strip()
            else:
                ALL_STUDIES[_name] = {"path": _path.strip(), "ks_type": "K_H",
                                      "set_op": "single", "engine_a": _name, "engine_b": None, "label": _name}
    run()
