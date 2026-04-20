"""
Experiment 05 — Compare target color distributions across studies A, B, C.

For each study database (and single-engine databases if available), load the
target color distributions and measure:
  - Centroid (mean RGB)
  - Spread (std dev, range)
  - Pairwise Wasserstein-1 distance
  - Venn overlap fraction between pairs
  - Symmetry score of the full collection

This directly answers: do the 3 set-op studies form a balanced (symmetric) whole?
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

_OUT = Path(__file__).parent.parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

# Databases to compare (path relative to color_mixing_lab root)
STUDY_DATABASES = {
    "study_a (mixbox+RYB, K_H_OVERLAP)": "output/db_study_a_artist_consensus",
    "study_b (spectral-RYB, K_T)":       "output/db_study_b_physics_vs_artist",
    "study_c (KM-mixbox, K_E)":          "output/db_study_c_oilpaint_vs_fooddye",
}
SINGLE_ENGINE_DATABASES = {
    "spectral (K_H)": "output/db_spectral",
    "mixbox (K_E)":   "output/db_mixbox",
    "km (K_E)":       "output/db_km",
    "ryb (K_T)":      "output/db_ryb",
}


def load_targets(db_path: str, max_targets: int = 5000) -> np.ndarray | None:
    """Load all target RGB values from a database directory."""
    db = Path(_ROOT) / db_path
    if not db.exists():
        return None

    targets = []
    for path in sorted(glob.glob(str(db / "targets" / "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, dict) and "targets" in d:
            targets.extend(d["targets"])
        if len(targets) >= max_targets:
            break

    if not targets:
        return None

    arr = np.array(targets[:max_targets], dtype=float)
    # Normalise to [0, 1]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0, 1)


def wasserstein1_approx(a: np.ndarray, b: np.ndarray, n_proj: int = 100) -> float:
    """
    Approximate Wasserstein-1 distance using random projections (sliced Wasserstein).
    Works in 3D (RGB).
    """
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(n_proj, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    distances = []
    for d in directions:
        pa = np.sort(a @ d)
        pb = np.sort(b @ d)
        # Interpolate to common size
        if len(pa) != len(pb):
            t = np.linspace(0, 1, min(len(pa), len(pb)))
            pa = np.interp(t, np.linspace(0, 1, len(pa)), pa)
            pb = np.interp(t, np.linspace(0, 1, len(pb)), pb)
        distances.append(np.mean(np.abs(pa - pb)))

    return float(np.mean(distances))


def voxel_overlap(a: np.ndarray, b: np.ndarray, res: int = 32) -> dict:
    """
    Compute Venn overlap statistics between two point clouds using voxels.
    Returns intersection/union/a-only/b-only fractions.
    """
    def voxelise(pts):
        idx = np.clip((pts * res).astype(int), 0, res - 1)
        g = np.zeros((res, res, res), dtype=bool)
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return g

    va = voxelise(a)
    vb = voxelise(b)

    inter = (va & vb).sum()
    union = (va | vb).sum()
    a_only = (va & ~vb).sum()
    b_only = (~va & vb).sum()

    return {
        "intersection_fraction": int(inter) / int(union) if union else 0.0,
        "a_only_fraction":       int(a_only) / int(va.sum()) if va.sum() else 0.0,
        "b_only_fraction":       int(b_only) / int(vb.sum()) if vb.sum() else 0.0,
        "jaccard":               int(inter) / int(union) if union else 0.0,
    }


def symmetry_score_collection(point_clouds: list[np.ndarray]) -> float:
    """
    Symmetry score of a collection: 1 - CoV of per-study voxel volumes.
    1.0 = perfectly balanced.  0.0 = one study dominates.
    """
    res = 32
    volumes = []
    for pts in point_clouds:
        idx = np.clip((pts * res).astype(int), 0, res - 1)
        g = np.zeros((res, res, res), dtype=bool)
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        volumes.append(g.sum())

    if not volumes or np.mean(volumes) == 0:
        return 0.0
    cov = np.std(volumes) / np.mean(volumes)
    return float(max(0.0, 1.0 - cov))


def run():
    all_dbs = {**STUDY_DATABASES, **SINGLE_ENGINE_DATABASES}
    loaded = {}

    print("=== Loading target distributions ===")
    for name, path in all_dbs.items():
        pts = load_targets(path)
        if pts is not None:
            loaded[name] = pts
            print(f"  {name}: {len(pts)} targets  "
                  f"R=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
                  f"G=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
                  f"B=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        else:
            print(f"  {name}: NOT FOUND at {path}")

    if not loaded:
        print("No databases found. Run generate_policy_data.py first.")
        return

    names = list(loaded.keys())
    n = len(names)

    # --- Per-study statistics ---
    print("\n=== Per-Study Statistics (mean RGB in [0,1]) ===")
    stats = {}
    for name, pts in loaded.items():
        stats[name] = {
            "mean": pts.mean(axis=0).tolist(),
            "std":  pts.std(axis=0).tolist(),
            "n":    len(pts),
        }
        m = stats[name]["mean"]
        s = stats[name]["std"]
        print(f"  {name[:45]:<45}  mean=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f})  "
              f"std=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f})")

    # --- Symmetry score of set-op studies ---
    set_op_names = [k for k in names if k.startswith("study_")]
    if set_op_names:
        sym_score = symmetry_score_collection([loaded[k] for k in set_op_names])
        print(f"\nSymmetry score of {{A,B,C}}: {sym_score:.4f}  "
              f"(1.0 = perfectly balanced by voxel volume)")

    # --- Pairwise Wasserstein distances ---
    print("\n=== Pairwise Wasserstein-1 Distances (sliced, lower=more similar) ===")
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = wasserstein1_approx(loaded[names[i]], loaded[names[j]])
            W[i, j] = W[j, i] = d

    header = f"{'':40}" + "".join(f"{k[:8]:>10}" for k in names)
    print(header)
    for i, ni in enumerate(names):
        row = f"{ni[:40]:<40}" + "".join(f"{W[i,j]:>10.4f}" for j in range(n))
        print(row)

    # --- Venn overlap for set-op studies vs each other ---
    print("\n=== Venn Overlap (set-op studies) ===")
    set_op_loaded = {k: v for k, v in loaded.items() if k.startswith("study_")}
    so_names = list(set_op_loaded.keys())
    for i, na in enumerate(so_names):
        for j, nb in enumerate(so_names):
            if j <= i:
                continue
            ov = voxel_overlap(set_op_loaded[na], set_op_loaded[nb])
            print(f"  {na[:30]} vs {nb[:30]}:")
            print(f"    Jaccard={ov['jaccard']:.4f}  "
                  f"A-only={ov['a_only_fraction']:.4f}  "
                  f"B-only={ov['b_only_fraction']:.4f}")

    # --- Visualise: 2D projections of all studies ---
    fig, axes = plt.subplots(len(loaded), 3, figsize=(12, 3.5 * len(loaded)))
    if len(loaded) == 1:
        axes = [axes]

    channel_pairs = [(0, 1, "R", "G"), (0, 2, "R", "B"), (1, 2, "G", "B")]
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(loaded)))

    for row, (name, pts) in enumerate(loaded.items()):
        for col, (i, j, xi, xj) in enumerate(channel_pairs):
            ax = axes[row][col]
            ax.scatter(pts[:, i], pts[:, j], s=2, alpha=0.3, color=colors_list[row])
            ax.set_xlabel(xi)
            ax.set_ylabel(xj)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel(f"{name[:25]}\n{xj}")

    fig.suptitle("Target Color Distributions — All Studies (2D projections)")
    plt.tight_layout()
    plt.savefig(_OUT / "study_comparison_projections.png", dpi=120)
    plt.close()
    print(f"\n[05] Saved study_comparison_projections.png")

    # --- Visualise: Centroids in 3D ---
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for (name, pts), c in zip(loaded.items(), colors_list):
        mu = pts.mean(axis=0)
        ax.scatter(*mu, s=120, color=c, label=name[:30], zorder=5)
        # Draw uncertainty ellipse (std dev)
        std = pts.std(axis=0)
        for dim in range(3):
            offset = np.zeros(3)
            offset[dim] = std[dim]
            ax.plot([mu[0]-offset[0], mu[0]+offset[0]],
                    [mu[1]-offset[1], mu[1]+offset[1]],
                    [mu[2]-offset[2], mu[2]+offset[2]],
                    color=c, alpha=0.5, linewidth=2)

    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_title("Study centroids ± std (3D RGB)")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(_OUT / "study_centroids_3d.png", dpi=120)
    plt.close()
    print("[05] Saved study_centroids_3d.png")

    # Save JSON
    out_path = _OUT / "study_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_study_stats": stats,
            "wasserstein_matrix": {
                names[i]: {names[j]: float(W[i, j]) for j in range(n)}
                for i in range(n)
            },
            "set_op_symmetry_score": float(sym_score) if set_op_names else None,
        }, f, indent=2)
    print(f"[05] Results saved to {out_path}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Study comparison (Exp 05)")
    _p.add_argument("--studies", nargs="+", metavar="NAME=PATH",
                    help="All study databases as name=path pairs. "
                         "Names containing 'study_' go into set-op group; others into single-engine group.")
    _p.add_argument("--output-dir", default=None, metavar="DIR")
    _args = _p.parse_args()
    if _args.output_dir:
        _OUT = Path(_args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    if _args.studies:
        STUDY_DATABASES.clear()
        SINGLE_ENGINE_DATABASES.clear()
        for _s in _args.studies:
            _name, _, _path = _s.partition("=")
            _name = _name.strip()
            if "study" in _name:
                STUDY_DATABASES[_name] = _path.strip()
            else:
                SINGLE_ENGINE_DATABASES[_name] = _path.strip()
    run()
