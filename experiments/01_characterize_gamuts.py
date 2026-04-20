"""
Experiment 01 — Characterise each engine's gamut.

For each available engine:
  - Sample a dense point cloud
  - Compute AABB, convex hull volume, voxel fill fraction
  - Measure symmetry score under Oh, S3, Z2^3

Saves results to  extended/gamut_symmetry/results/gamut_characterization.json
and several PNG plots.
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _ROOT)
_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PKG)

import json
import numpy as np
from pathlib import Path

from gamut_sampler import sample_gamut, gamut_bounds, gamut_convex_hull_volume, voxelise
from coverage_checker import symmetry_score
from symmetry_group import OH_GROUP, S3_GROUP, Z2_GROUP, CYCLIC_GROUP
from visualizer import plot_gamut_3d, plot_gamut_projections, plot_symmetry_scores

_OUT = Path(__file__).parent.parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

ENGINES = ["spectral", "mixbox", "kubelka_munk", "coloraide_ryb"]
STEPS = 16          # grid steps per pigment axis  (16^3 ≈ 4k valid combos)
VOXEL_RES = 48      # voxel grid resolution for symmetry scoring

SUBGROUPS = {
    "Oh": OH_GROUP,
    "S3": S3_GROUP,
    "Z2^3": Z2_GROUP,
    "Cyclic-3": CYCLIC_GROUP,
}


def run():
    all_results = {}
    sym_scores: dict[str, list[float]] = {sg: [] for sg in SUBGROUPS}
    available_engines = []

    for eng in ENGINES:
        print(f"\n{'='*55}")
        print(f" Engine: {eng}")
        print(f"{'='*55}")

        try:
            pts = sample_gamut(eng, steps=STEPS, use_cache=True)
        except Exception as e:
            print(f"  [SKIP] Could not load engine '{eng}': {e}")
            for sg in SUBGROUPS:
                sym_scores[sg].append(None)
            all_results[eng] = {"error": str(e)}
            continue

        available_engines.append(eng)

        # Basic stats
        bounds = gamut_bounds(pts)
        print(f"  Points sampled : {bounds['n_points']:,}")
        print(f"  AABB min       : {[f'{v:.3f}' for v in bounds['min']]}")
        print(f"  AABB max       : {[f'{v:.3f}' for v in bounds['max']]}")
        print(f"  AABB volume    : {bounds['bbox_volume']:.4f}")

        # Convex hull volume
        try:
            ch_vol = gamut_convex_hull_volume(pts)
            print(f"  Convex hull    : {ch_vol:.4f}  ({ch_vol*100:.1f}% of unit cube)")
        except ImportError:
            ch_vol = float("nan")
            print("  Convex hull    : scipy not available")

        # Voxelise
        vox = voxelise(pts, resolution=VOXEL_RES)
        vox_frac = vox.sum() / VOXEL_RES**3
        print(f"  Voxel fill     : {vox.sum():,} / {VOXEL_RES**3:,}  ({vox_frac*100:.1f}%)")

        # Symmetry scores
        eng_sym = {}
        for sg_name, sg in SUBGROUPS.items():
            score = symmetry_score(vox, sg)
            eng_sym[sg_name] = score
            sym_scores[sg_name].append(score)
            print(f"  Symmetry ({sg_name:8s}): {score:.4f}")

        all_results[eng] = {
            "n_points": bounds["n_points"],
            "aabb_min": bounds["min"],
            "aabb_max": bounds["max"],
            "aabb_volume": bounds["bbox_volume"],
            "convex_hull_volume": ch_vol,
            "voxel_fill_fraction": float(vox_frac),
            "symmetry_scores": eng_sym,
        }

        # Plots
        plot_gamut_3d(
            pts,
            title=f"{eng} gamut",
            save_path=_OUT / f"gamut_3d_{eng}.png",
        )
        plot_gamut_projections(
            pts,
            title=f"{eng} gamut — 2D projections",
            save_path=_OUT / f"gamut_proj_{eng}.png",
        )

    # Summary bar chart
    valid_engines = [e for e in available_engines]
    valid_scores = {sg: [sym_scores[sg][ENGINES.index(e)] for e in valid_engines]
                    for sg in SUBGROUPS}
    if valid_engines:
        plot_symmetry_scores(
            valid_engines,
            valid_scores,
            save_path=_OUT / "symmetry_scores_all_engines.png",
        )

    # Save JSON
    out_path = _OUT / "gamut_characterization.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[01] Results saved to {out_path}")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    header = f"{'Engine':<20} {'Hull%':>7} {'Vox%':>7} {'Sym-Oh':>8} {'Sym-S3':>8}"
    print(header)
    print("-" * len(header))
    for eng, res in all_results.items():
        if "error" in res:
            print(f"{eng:<20}  (unavailable)")
            continue
        ch = res.get("convex_hull_volume", float("nan"))
        vf = res.get("voxel_fill_fraction", float("nan"))
        so = res["symmetry_scores"].get("Oh", float("nan"))
        ss = res["symmetry_scores"].get("S3", float("nan"))
        print(f"{eng:<20} {ch*100:>6.1f}% {vf*100:>6.1f}%  {so:>7.4f}  {ss:>7.4f}")


if __name__ == "__main__":
    run()
