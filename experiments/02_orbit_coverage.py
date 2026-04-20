"""
Experiment 02 — Orbit coverage.

For each engine's gamut, apply every element of O_h in sequence and track
how the coverage of the full unit cube grows.

Key questions:
  1. Does the full orbit cover 100% of [0,1]^3?
  2. At what number of transforms does coverage plateau?
  3. How does coverage grow under S3 vs Z2^3 vs full O_h?

Saves:
  results/orbit_coverage_<engine>.json
  results/orbit_coverage_curve_<engine>.png
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
from coverage_checker import orbit_coverage, incremental_coverage
from symmetry_group import OH_GROUP, S3_GROUP, Z2_GROUP, SUBGROUPS
from visualizer import plot_coverage_curve

_OUT = Path(__file__).parent.parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

ENGINES = ["spectral", "mixbox", "kubelka_munk", "coloraide_ryb"]
STEPS = 16
VOXEL_RES = 40      # coarser for speed on the coverage loop


def run(engines=None, voxel_res=VOXEL_RES):
    if engines is None:
        engines = ENGINES

    all_results = {}

    for eng in engines:
        print(f"\n{'='*55}")
        print(f" Engine: {eng}")
        print(f"{'='*55}")

        try:
            pts = sample_gamut(eng, steps=STEPS, use_cache=True)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        vox = voxelise(pts, resolution=voxel_res)
        print(f"  Gamut voxels: {vox.sum():,} / {voxel_res**3:,}  "
              f"({vox.sum()/voxel_res**3*100:.1f}%)")

        eng_result = {"engine": eng, "subgroups": {}}

        for sg_name, sg in SUBGROUPS.items():
            if sg_name == "identity":
                continue

            print(f"\n  Subgroup: {sg_name} ({len(sg)} elements)")

            # Full orbit coverage
            cov = orbit_coverage(vox, sg)
            print(f"    Full orbit coverage : {cov['covered_fraction']*100:.2f}%")
            print(f"    Orbit volume        : {cov['orbit_volume_fraction']*100:.2f}% of cube")
            print(f"    Overlap ratio       : {cov['overlap_ratio']:.2f}x")

            # Incremental coverage curve
            records = incremental_coverage(vox, sg)

            eng_result["subgroups"][sg_name] = {
                "full_coverage": cov,
                "incremental": records,
            }

            # Plot
            plot_coverage_curve(
                records,
                title=f"{eng} — {sg_name} orbit coverage",
                save_path=_OUT / f"orbit_coverage_{eng}_{sg_name}.png",
                threshold=0.99,
            )

        all_results[eng] = eng_result

    # Save JSON (compact records)
    out_path = _OUT / "orbit_coverage.json"
    with open(out_path, "w") as f:
        # incremental records can be large; keep them
        json.dump(all_results, f, indent=2)
    print(f"\n[02] Results saved to {out_path}")

    # Summary
    print("\n=== ORBIT COVERAGE SUMMARY ===")
    for eng, res in all_results.items():
        print(f"\n  {eng}:")
        for sg_name, sg_res in res.get("subgroups", {}).items():
            fc = sg_res["full_coverage"]["covered_fraction"]
            # Find where coverage first crosses 99%
            records = sg_res["incremental"]
            n99 = next((r["n_transforms"] for r in records
                        if r["covered_fraction"] >= 0.99), None)
            print(f"    {sg_name:<8} full={fc*100:.1f}%  "
                  f"n@99%={n99 if n99 else '>'+str(len(records))}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--engines", nargs="+", default=None)
    p.add_argument("--resolution", type=int, default=VOXEL_RES)
    args = p.parse_args()
    run(engines=args.engines, voxel_res=args.resolution)
