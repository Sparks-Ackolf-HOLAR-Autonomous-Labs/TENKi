"""
Experiment 04 — Which subgroup of O_h suffices for tiling?

For each engine, find the SMALLEST subgroup of O_h under which the gamut's
orbit covers ≥99% of the target.

Lattice of subgroups tested (from smallest to full):
  identity (1) → cyclic-3 (3) → S3 (6) → Z2^3 (8) → D4 (8) → S3×Z2 (12)
  → ... → O_h (48)

This tells us how much symmetry we actually *need* to tile — and whether
simpler operations (just swapping channels, or just negating) are sufficient.

Also computes:
  - The "symmetry defect" = 1 − symmetry_score(gamut, subgroup)
    (how far the gamut is from being invariant under each subgroup)
  - The "tiling efficiency" = T(G, subgroup) / |subgroup|
    (fraction of the subgroup you actually need)
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _ROOT)
_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PKG)

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from gamut_sampler import sample_gamut, voxelise
from coverage_checker import tiling_number_greedy, symmetry_score
from symmetry_group import (
    OH_GROUP, S3_GROUP, Z2_GROUP, CYCLIC_GROUP,
    SymmetryTransform, build_Oh_group, SUBGROUPS,
)

_OUT = Path(__file__).parent.parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

ENGINES = ["spectral", "mixbox", "kubelka_munk", "coloraide_ryb"]
STEPS = 16
VOXEL_RES = 36


def build_subgroup_ladder() -> dict[str, list[SymmetryTransform]]:
    """
    Return an ordered dict of subgroups from smallest to O_h, keyed by name.
    """
    oh = build_Oh_group()

    # Identity
    identity = [g for g in oh if g.perm == (0,1,2) and sum(g.flip) == 0]

    # Cyclic-3 permutations: (RGB), (GBR), (BRG)
    cyc3 = [g for g in oh if g.perm in {(0,1,2),(1,2,0),(2,0,1)} and sum(g.flip)==0]

    # S3: all 6 permutations, no flips
    s3 = [g for g in oh if sum(g.flip) == 0]

    # Z2^3: identity permutation, all flip combos
    z2 = [g for g in oh if g.perm == (0,1,2)]

    # S3 × Z2 (first channel negation composed with all perms): order 12
    # = all elements where only one channel might be flipped but all flips must
    # have even parity (even number of negated channels) + permutations
    s3z2 = [g for g in oh if sum(g.flip) % 2 == 0]

    # Full O_h
    return {
        "identity": identity,
        "cyclic-3": cyc3,
        "S3": s3,
        "Z2^3": z2,
        "S3×Z2(even)": s3z2,
        "Oh": oh,
    }


def run(engines=None, voxel_res=VOXEL_RES):
    if engines is None:
        engines = ENGINES

    subgroup_ladder = build_subgroup_ladder()
    sg_names = list(subgroup_ladder.keys())
    sg_sizes = {k: len(v) for k, v in subgroup_ladder.items()}
    print("Subgroup ladder:")
    for name, sg in subgroup_ladder.items():
        print(f"  {name:<20}: {len(sg):>3} elements")

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
        eng_results = {}

        T_values = []
        sym_defects = []

        for sg_name, sg in subgroup_ladder.items():
            if sg_name == "identity":
                # trivial: tiling number = number of cubes needed to fill, just skip
                T_values.append(None)
                sym_defects.append(1.0 - symmetry_score(vox, sg))
                eng_results[sg_name] = {"tiling_number": None, "symmetry_defect": 1.0}
                continue

            # Tiling number at 99%
            res = tiling_number_greedy(vox, sg, coverage_threshold=0.99)
            T = res["tiling_number"]
            eff = T / sg_sizes[sg_name] if sg_sizes[sg_name] else float("inf")
            # Symmetry defect
            defect = 1.0 - symmetry_score(vox, sg)

            T_values.append(T)
            sym_defects.append(defect)

            eng_results[sg_name] = {
                "tiling_number": T,
                "coverage": res["final_coverage"],
                "subgroup_size": sg_sizes[sg_name],
                "tiling_efficiency": eff,
                "symmetry_defect": defect,
                "feasible": res["is_feasible"],
            }
            print(f"  {sg_name:<20}: T(G)={T:>3}  |sg|={sg_sizes[sg_name]:>3}  "
                  f"eff={eff:.2f}  sym_defect={defect:.4f}  "
                  f"cov={res['final_coverage']*100:.1f}%")

        all_results[eng] = eng_results

        # Plot: T(G) and symmetry defect vs subgroup
        labels = [n for n in sg_names if n != "identity"]
        T_vals = [eng_results[n]["tiling_number"] for n in labels if n in eng_results]
        d_vals = [eng_results[n]["symmetry_defect"] for n in labels if n in eng_results]
        sizes  = [sg_sizes[n] for n in labels if n in eng_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        x = range(len(labels))

        ax1.bar(x, T_vals, color="steelblue")
        ax1.plot(x, sizes, "r--o", label="|subgroup|", linewidth=1.5)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax1.set_ylabel("Tiling number T(G)")
        ax1.set_title(f"{eng} — T(G) vs subgroup")
        ax1.legend()

        ax2.bar(x, d_vals, color="coral")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=20, ha="right")
        ax2.set_ylabel("Symmetry defect  (1 − IoU-mean)")
        ax2.set_title(f"{eng} — Asymmetry vs subgroup")

        plt.tight_layout()
        plt.savefig(_OUT / f"subgroup_analysis_{eng}.png", dpi=150)
        plt.close()
        print(f"  [04] Saved subgroup_analysis_{eng}.png")

    out_path = _OUT / "subgroup_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[04] Results saved to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--engines", nargs="+", default=None)
    p.add_argument("--resolution", type=int, default=VOXEL_RES)
    args = p.parse_args()
    run(engines=args.engines, voxel_res=args.resolution)
