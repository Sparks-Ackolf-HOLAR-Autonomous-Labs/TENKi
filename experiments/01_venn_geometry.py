"""
Experiment 01 — Venn Region Geometry.

Maps the actual set-theoretic overlap structure between engine gamuts using
voxel grids.  For every pair of engines (A, B) partitions the RGB cube into:

  intersection  G_A ∩ G_B   — colors both engines can produce
  A-only        G_A \\ G_B  — colors A produces that B cannot
  B-only        G_B \\ G_A  — colors B produces that A cannot

Replaces the O_h crystallographic tiling experiments (v0.1 experiments 01-04),
which measured geometric symmetry of single gamuts under channel permutations/
negations — a valid but orthogonal question to the set-theoretic overlap that
the study-comparison framework (experiments 05-08) actually relies on.

Key outputs
-----------
venn_geometry.json
    volumes (as fraction of RGB cube) for every pair and multi-engine intersection

venn_pair_<A>_<B>.png
    2D projection (RG, RB, GB planes) coloring intersection/A-only/B-only

venn_summary.png
    bar chart of region volumes + signed asymmetry index per pair

Asymmetry index = (|A-only| - |B-only|) / (|A-only| + |B-only|)
    > 0  A has more exclusive colors (A is the larger unique contributor)
    < 0  B has more exclusive colors
    = 0  perfectly symmetric pair
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_PKG  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _PKG)

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

from gamut_sampler import sample_gamut, voxelise, gamut_bounds

_OUT = Path(__file__).parent.parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

ENGINES   = ["spectral", "mixbox", "kubelka_munk", "coloraide_ryb"]
STEPS     = 16   # grid steps per pigment axis — same density as v0.1 experiments
VOXEL_RES = 40   # voxel grid resolution (40^3 = 64 k cells, fast + accurate enough)


# ---------------------------------------------------------------------------
# Core Venn computation
# ---------------------------------------------------------------------------

def venn_pair(vox_a: np.ndarray, vox_b: np.ndarray) -> dict:
    """Compute Venn partition volumes for two gamut voxel grids."""
    total    = vox_a.shape[0] ** 3
    inter    = vox_a & vox_b
    a_only   = vox_a & ~vox_b
    b_only   = vox_b & ~vox_a
    union_ab = vox_a | vox_b

    n_inter = int(inter.sum())
    n_a     = int(a_only.sum())
    n_b     = int(b_only.sum())
    n_union = int(union_ab.sum())

    asym    = (n_a - n_b) / (n_a + n_b + 1e-10)
    jaccard = n_inter / n_union if n_union else 0.0

    return {
        "intersection_frac": n_inter / total,
        "a_only_frac":       n_a     / total,
        "b_only_frac":       n_b     / total,
        "union_frac":        n_union / total,
        "asymmetry_index":   float(asym),
        "jaccard":           float(jaccard),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_venn_projections(vox_a, vox_b, name_a, name_b, save_path):
    """Three 2D projections coloured by Venn region membership."""
    inter  = vox_a & vox_b
    a_only = vox_a & ~vox_b
    b_only = vox_b & ~vox_a
    R = vox_a.shape[0]

    # projection axis indices: (x_channel, y_channel, collapse_axis)
    projections = [("R", "G", 2), ("R", "B", 1), ("G", "B", 0)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (xl, yl, collapse) in zip(axes, projections):
        p_inter  = inter.any(axis=collapse)
        p_a      = a_only.any(axis=collapse)
        p_b      = b_only.any(axis=collapse)

        # Build index arrays for scatter
        def _pts(mask):
            idx = np.argwhere(mask)
            return idx[:, 0] / R, idx[:, 1] / R

        xs_i, ys_i = _pts(p_inter)
        xs_a, ys_a = _pts(p_a & ~p_inter)
        xs_b, ys_b = _pts(p_b & ~p_inter)

        if len(xs_i): ax.scatter(xs_i, ys_i, s=1, c="purple", alpha=0.35, label="intersection")
        if len(xs_a): ax.scatter(xs_a, ys_a, s=1, c="crimson", alpha=0.35, label=f"{name_a}-only")
        if len(xs_b): ax.scatter(xs_b, ys_b, s=1, c="steelblue", alpha=0.35, label=f"{name_b}-only")

        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{xl}-{yl} plane", fontsize=9)
        ax.legend(markerscale=6, fontsize=7)

    fig.suptitle(f"Venn regions: {name_a}  vs  {name_b}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("Loading engine gamuts...")
    engine_voxels = {}
    engine_stats  = {}
    for eng in ENGINES:
        try:
            pts = sample_gamut(eng, steps=STEPS, use_cache=True)
            vox = voxelise(pts, resolution=VOXEL_RES)
            engine_voxels[eng] = vox
            b = gamut_bounds(pts)
            fill = float(vox.sum()) / VOXEL_RES ** 3
            engine_stats[eng] = {
                "n_points": len(pts),
                "fill_fraction": fill,
                "aabb_min_rgb": [int(v * 255) for v in b["min"]],
                "aabb_max_rgb": [int(v * 255) for v in b["max"]],
                "mean_rgb":     [int(v * 255) for v in b["mean"]],
            }
            print(f"  {eng}: {len(pts):,} pts  fill={fill*100:.1f}%  "
                  f"R=[{engine_stats[eng]['aabb_min_rgb'][0]},{engine_stats[eng]['aabb_max_rgb'][0]}] "
                  f"G=[{engine_stats[eng]['aabb_min_rgb'][1]},{engine_stats[eng]['aabb_max_rgb'][1]}] "
                  f"B=[{engine_stats[eng]['aabb_min_rgb'][2]},{engine_stats[eng]['aabb_max_rgb'][2]}]")
        except Exception as e:
            print(f"  [SKIP] {eng}: {e}")

    available  = list(engine_voxels.keys())
    total_vox  = VOXEL_RES ** 3

    # ------------------------------------------------------------------
    # Pairwise Venn
    # ------------------------------------------------------------------
    print("\n=== Pairwise Venn regions (fraction of RGB cube) ===")
    pair_results = {}
    for eng_a, eng_b in combinations(available, 2):
        key = f"{eng_a}_vs_{eng_b}"
        res = venn_pair(engine_voxels[eng_a], engine_voxels[eng_b])
        pair_results[key] = {"engine_a": eng_a, "engine_b": eng_b, **res}
        print(f"\n  {eng_a}  vs  {eng_b}")
        print(f"    intersection : {res['intersection_frac']*100:5.2f}%")
        print(f"    {eng_a}-only  : {res['a_only_frac']*100:5.2f}%")
        print(f"    {eng_b}-only  : {res['b_only_frac']*100:5.2f}%")
        print(f"    asymmetry    : {res['asymmetry_index']:+.4f}  "
              f"({'A larger' if res['asymmetry_index']>0.01 else 'B larger' if res['asymmetry_index']<-0.01 else 'symmetric'})")
        print(f"    Jaccard      : {res['jaccard']:.4f}")
        plot_venn_projections(
            engine_voxels[eng_a], engine_voxels[eng_b],
            eng_a, eng_b,
            _OUT / f"venn_pair_{eng_a}_{eng_b}.png",
        )

    # ------------------------------------------------------------------
    # Multi-engine intersections
    # ------------------------------------------------------------------
    print("\n=== Multi-engine intersections ===")
    multi_results = {}
    for k in range(2, len(available) + 1):
        for subset in combinations(available, k):
            vox = engine_voxels[subset[0]].copy()
            for eng in subset[1:]:
                vox &= engine_voxels[eng]
            frac = float(vox.sum()) / total_vox
            key  = "_inter_".join(subset)
            multi_results[key] = {
                "engines": list(subset),
                "fraction": frac,
                "count": int(vox.sum()),
                "non_empty": vox.any(),
            }
            label = " ∩ ".join(subset)
            print(f"  {label}: {frac*100:.3f}%  "
                  f"({'non-empty' if vox.any() else 'EMPTY'})")

    # Check whether 4-engine intersection is non-empty
    if len(available) == 4:
        four_key = "_inter_".join(available)
        status   = multi_results.get(four_key, {})
        print(f"\n  4-engine intersection non-empty: {status.get('non_empty', 'N/A')}")
        print(f"  (This is the 'fully symmetric core' — required for engine-permutation symmetry)")

    # ------------------------------------------------------------------
    # Summary plot
    # ------------------------------------------------------------------
    pairs      = list(pair_results.keys())
    pair_labels = [p.replace("_vs_", "\nvs\n") for p in pairs]
    x  = np.arange(len(pairs))
    w  = 0.25

    fracs_i = [pair_results[p]["intersection_frac"] * 100 for p in pairs]
    fracs_a = [pair_results[p]["a_only_frac"]        * 100 for p in pairs]
    fracs_b = [pair_results[p]["b_only_frac"]        * 100 for p in pairs]
    asym_v  = [pair_results[p]["asymmetry_index"]         for p in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - w, fracs_i, w, label="intersection", color="purple",    alpha=0.7)
    axes[0].bar(x,     fracs_a, w, label="A-only",       color="crimson",   alpha=0.7)
    axes[0].bar(x + w, fracs_b, w, label="B-only",       color="steelblue", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pair_labels, fontsize=7)
    axes[0].set_ylabel("% of RGB cube")
    axes[0].set_title("Venn region volumes per engine pair")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    bar_colors = ["crimson" if a > 0.01 else "steelblue" if a < -0.01 else "gray"
                  for a in asym_v]
    axes[1].bar(x, asym_v, color=bar_colors, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_labels, fontsize=7)
    axes[1].set_ylabel("Asymmetry index  =  (|A-only| − |B-only|) / (|A-only| + |B-only|)")
    axes[1].set_title("Directional asymmetry per pair\n(+: A has more unique colors; −: B does)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(_OUT / "venn_summary.png", dpi=150)
    plt.close()
    print("\n[01] Saved venn_summary.png")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out_data = {
        "voxel_resolution": VOXEL_RES,
        "total_voxels": total_vox,
        "engine_stats": engine_stats,
        "pairwise_venn": pair_results,
        "multi_engine_intersections": multi_results,
    }
    out_path = _OUT / "venn_geometry.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"[01] Saved {out_path}")


if __name__ == "__main__":
    run()
