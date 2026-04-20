"""
Experiment 03 — Tiling number T(G).

For each engine, compute the minimum number of O_h symmetry transforms
needed to cover ≥99% of the unit cube (or a symmetric target).

Uses a greedy set-cover algorithm (gives an upper bound; the true minimum
can be lower if an ILP is solved, but greedy is within ln(N) factor).

Also tests:
  - Different coverage thresholds (90%, 95%, 99%, 100%)
  - Different target spaces (full cube, gamut intersection of 2 engines)
  - Sub-groups only (S3, Z2^3)

Saves:
  results/tiling_number_summary.json
  results/tiling_summary_chart.png
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _ROOT)
_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PKG)

import json
import numpy as np
from pathlib import Path

from gamut_sampler import sample_gamut, voxelise
from coverage_checker import tiling_number_greedy, orbit_coverage
from symmetry_group import OH_GROUP, S3_GROUP, Z2_GROUP, SUBGROUPS
from visualizer import plot_tiling_summary

_OUT = Path(__file__).parent.parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

ENGINES = ["spectral", "mixbox", "kubelka_munk", "coloraide_ryb"]
STEPS = 16
VOXEL_RES = 36      # keep manageable; higher = slower but more accurate
THRESHOLDS = [0.90, 0.95, 0.99]


def run(engines=None, voxel_res=VOXEL_RES, thresholds=None):
    if engines is None:
        engines = ENGINES
    if thresholds is None:
        thresholds = THRESHOLDS

    all_results = []
    engine_voxels = {}

    # Load all available engines first
    for eng in engines:
        try:
            pts = sample_gamut(eng, steps=STEPS, use_cache=True)
            engine_voxels[eng] = voxelise(pts, resolution=voxel_res)
            print(f"[03] {eng}: {engine_voxels[eng].sum():,} voxels loaded")
        except Exception as e:
            print(f"[03] {eng}: SKIP — {e}")

    available = list(engine_voxels.keys())

    # --- Per-engine, full O_h group ---
    print(f"\n{'='*60}")
    print(" Tiling numbers under full O_h group")
    print(f"{'='*60}")

    for eng in available:
        vox = engine_voxels[eng]
        eng_result = {"engine": eng, "by_threshold": {}, "by_subgroup": {}}

        # Full O_h at each threshold
        for thresh in thresholds:
            res = tiling_number_greedy(vox, OH_GROUP, coverage_threshold=thresh)
            eng_result["by_threshold"][f"{thresh:.2f}"] = res
            print(f"  {eng} @ {thresh*100:.0f}%: T(G) = {res['tiling_number']:2d}  "
                  f"coverage = {res['final_coverage']*100:.2f}%  "
                  f"feasible={res['is_feasible']}")

        # Per subgroup at 99%
        for sg_name, sg in SUBGROUPS.items():
            if sg_name in ("identity", "Oh"):
                continue
            res = tiling_number_greedy(vox, sg, coverage_threshold=0.99)
            eng_result["by_subgroup"][sg_name] = res
            print(f"  {eng} subgroup={sg_name}: T(G)={res['tiling_number']}  "
                  f"cov={res['final_coverage']*100:.1f}%")

        all_results.append(eng_result)

    # --- Cross-engine: can engine A tile engine B's gamut? ---
    print(f"\n{'='*60}")
    print(" Cross-engine tiling (can A tile B's gamut as target?)")
    print(f"{'='*60}")

    cross_results = {}
    for eng_a in available:
        for eng_b in available:
            if eng_a == eng_b:
                continue
            source = engine_voxels[eng_a]
            target = engine_voxels[eng_b]
            res = tiling_number_greedy(source, OH_GROUP, target_voxels=target, coverage_threshold=0.95)
            key = f"{eng_a}→{eng_b}"
            cross_results[key] = res
            print(f"  {key}: T={res['tiling_number']:2d}  cov={res['final_coverage']*100:.1f}%")

    # --- Intersection target ---
    if len(available) >= 2:
        print(f"\n{'='*60}")
        print(" Tiling number for intersection target")
        print(f"{'='*60}")
        intersection = engine_voxels[available[0]].copy()
        for eng in available[1:]:
            intersection &= engine_voxels[eng]
        intersection_vol = int(intersection.sum())
        print(f"  Intersection voxels: {intersection_vol} "
              f"({intersection_vol/voxel_res**3*100:.2f}% of cube)")

        for eng in available:
            source = engine_voxels[eng]
            if intersection_vol == 0:
                print(f"  {eng}: intersection is empty — skip")
                continue
            res = tiling_number_greedy(source, OH_GROUP,
                                       target_voxels=intersection,
                                       coverage_threshold=0.99)
            print(f"  {eng} → intersection: T={res['tiling_number']}  "
                  f"cov={res['final_coverage']*100:.1f}%")

    # Save JSON
    full_output = {
        "per_engine": all_results,
        "cross_engine": cross_results,
    }
    out_path = _OUT / "tiling_number_summary.json"
    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[03] Results saved to {out_path}")

    # Chart
    chart_data = []
    for er in all_results:
        t99_key = "0.99"
        if t99_key in er["by_threshold"]:
            chart_data.append({
                "engine": er["engine"],
                "tiling_number": er["by_threshold"][t99_key]["tiling_number"],
                "final_coverage": er["by_threshold"][t99_key]["final_coverage"],
            })
    if chart_data:
        plot_tiling_summary(chart_data, save_path=_OUT / "tiling_summary_chart.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--engines", nargs="+", default=None)
    p.add_argument("--resolution", type=int, default=VOXEL_RES)
    p.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    args = p.parse_args()
    run(engines=args.engines, voxel_res=args.resolution, thresholds=args.thresholds)
