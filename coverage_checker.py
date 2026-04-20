"""
Coverage checker — given a gamut point cloud and a symmetry group, compute
how well the orbit of the gamut covers a target space.

All computations are done on discretised voxel grids for tractability.
The unit cube [0,1]^3 is divided into `resolution`^3 voxels.

Key outputs
-----------
orbit_coverage(gamut_voxels, group, target_voxels)
    → fraction of target voxels covered by the full orbit of gamut_voxels

minimum_cover(gamut_voxels, group, target_voxels)
    → smallest subset of transforms whose union covers (≥ threshold of) target

tiling_number(gamut_voxels, group, target_voxels)
    → minimum number of transforms needed for exact (or ε-approximate) coverage
"""

from __future__ import annotations
import numpy as np
from typing import Sequence

try:
    from .symmetry_group import SymmetryTransform, OH_GROUP, SUBGROUPS
except ImportError:
    from symmetry_group import SymmetryTransform, OH_GROUP, SUBGROUPS


# ---------------------------------------------------------------------------
# Low-level voxel helpers
# ---------------------------------------------------------------------------

def apply_transform_to_voxels(
    voxels: np.ndarray,
    transform: SymmetryTransform,
) -> np.ndarray:
    """
    Apply a symmetry transform to a boolean voxel grid.

    `voxels` is shape (R, R, R) with dtype bool.
    Returns a new boolean grid of the same shape.
    """
    R = voxels.shape[0]
    # Build index arrays for the occupied voxels
    idx = np.argwhere(voxels)  # (N, 3)
    # Convert to [0, 1] coordinates (centre of each voxel)
    pts = (idx + 0.5) / R
    # Apply transform
    pts_t = transform(pts)
    # Convert back to voxel indices
    idx_t = np.clip((pts_t * R).astype(int), 0, R - 1)
    # Build output grid
    out = np.zeros_like(voxels)
    out[idx_t[:, 0], idx_t[:, 1], idx_t[:, 2]] = True
    return out


def orbit_union(
    voxels: np.ndarray,
    group: list[SymmetryTransform],
) -> np.ndarray:
    """Union of all transforms in group applied to voxels."""
    union = voxels.copy()
    for t in group:
        union |= apply_transform_to_voxels(voxels, t)
    return union


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------

def orbit_coverage(
    gamut_voxels: np.ndarray,
    group: list[SymmetryTransform],
    target_voxels: np.ndarray | None = None,
) -> dict:
    """
    Compute how much of `target_voxels` is covered by the orbit of `gamut_voxels`.

    Parameters
    ----------
    gamut_voxels : boolean voxel grid (R, R, R)
    group : list of symmetry transforms to apply
    target_voxels : boolean voxel grid to measure coverage against.
                    If None, the full [0,1]^3 cube is used.

    Returns
    -------
    dict with keys:
        covered_fraction   fraction of target covered
        covered_count      number of target voxels covered
        target_count       total number of target voxels
        orbit_volume       fraction of cube occupied by the orbit union
        overlap_ratio      (sum of individual volumes) / orbit_volume  ≥ 1
    """
    R = gamut_voxels.shape[0]
    if target_voxels is None:
        target_voxels = np.ones((R, R, R), dtype=bool)

    # Compute per-transform coverages for overlap ratio
    total_individual = gamut_voxels.sum()
    union = gamut_voxels.copy()
    for t in group:
        t_vox = apply_transform_to_voxels(gamut_voxels, t)
        union |= t_vox
        total_individual += t_vox.sum()

    covered = union & target_voxels
    target_count = int(target_voxels.sum())
    covered_count = int(covered.sum())
    orbit_volume = int(union.sum())
    cube_volume = R ** 3

    return {
        "covered_fraction": covered_count / target_count if target_count else 0.0,
        "covered_count": covered_count,
        "target_count": target_count,
        "orbit_volume_fraction": orbit_volume / cube_volume,
        "gamut_volume_fraction": int(gamut_voxels.sum()) / cube_volume,
        "overlap_ratio": float(total_individual) / orbit_volume if orbit_volume else float("inf"),
        "n_transforms": len(group) + 1,  # includes identity
    }


def incremental_coverage(
    gamut_voxels: np.ndarray,
    group: list[SymmetryTransform],
    target_voxels: np.ndarray | None = None,
) -> list[dict]:
    """
    Compute coverage as transforms are added one-by-one (in group order).

    Returns list of dicts {n_transforms, covered_fraction} for n = 0..len(group).
    """
    R = gamut_voxels.shape[0]
    if target_voxels is None:
        target_voxels = np.ones((R, R, R), dtype=bool)

    target_count = int(target_voxels.sum())
    union = gamut_voxels.copy()
    records = [{
        "n_transforms": 1,
        "covered_fraction": float((union & target_voxels).sum()) / target_count,
    }]

    for i, t in enumerate(group):
        union |= apply_transform_to_voxels(gamut_voxels, t)
        records.append({
            "n_transforms": i + 2,
            "covered_fraction": float((union & target_voxels).sum()) / target_count,
        })

    return records


# ---------------------------------------------------------------------------
# Tiling number (greedy upper bound)
# ---------------------------------------------------------------------------

def tiling_number_greedy(
    gamut_voxels: np.ndarray,
    group: list[SymmetryTransform],
    target_voxels: np.ndarray | None = None,
    coverage_threshold: float = 0.99,
) -> dict:
    """
    Greedy set-cover: find the smallest subset of the group orbit that covers
    ≥ `coverage_threshold` fraction of `target_voxels`.

    Returns
    -------
    dict with:
        tiling_number         number of transforms selected
        selected_transforms   list of transform names
        final_coverage        actual coverage fraction achieved
        is_exact              True if coverage == 1.0
        is_feasible           True if coverage >= threshold
    """
    R = gamut_voxels.shape[0]
    if target_voxels is None:
        target_voxels = np.ones((R, R, R), dtype=bool)

    target_count = int(target_voxels.sum())
    remaining = target_voxels.copy()  # uncovered target voxels

    # Pre-compute all transform images
    all_images = [gamut_voxels.copy()]  # identity
    all_names = ["identity"]
    for t in group:
        all_images.append(apply_transform_to_voxels(gamut_voxels, t))
        all_names.append(t.name)

    selected_indices = []
    covered = np.zeros((R, R, R), dtype=bool)

    while True:
        # Current coverage
        cov_frac = float((covered & target_voxels).sum()) / target_count
        if cov_frac >= coverage_threshold:
            break
        if not remaining.any():
            break

        # Pick the transform that covers the most uncovered target voxels
        best_idx = -1
        best_gain = -1
        for i, img in enumerate(all_images):
            if i in selected_indices:
                continue
            gain = int((img & remaining).sum())
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx < 0 or best_gain == 0:
            break  # no progress possible

        selected_indices.append(best_idx)
        covered |= all_images[best_idx]
        remaining = target_voxels & ~covered

    final_cov = float((covered & target_voxels).sum()) / target_count

    return {
        "tiling_number": len(selected_indices),
        "selected_transforms": [all_names[i] for i in selected_indices],
        "final_coverage": final_cov,
        "is_exact": final_cov >= 1.0 - 1e-9,
        "is_feasible": final_cov >= coverage_threshold,
        "coverage_threshold": coverage_threshold,
    }


# ---------------------------------------------------------------------------
# Symmetry score of a voxel grid
# ---------------------------------------------------------------------------

def symmetry_score(
    voxels: np.ndarray,
    group: list[SymmetryTransform],
) -> float:
    """
    Measure how symmetric a voxel grid is under a group.

    Score = mean over all T ∈ group of IoU(voxels, T(voxels)).
    Score = 1.0 → perfectly symmetric.  Score ≈ 0 → fully asymmetric.
    """
    scores = []
    v_sum = voxels.sum()
    if v_sum == 0:
        return 0.0
    for t in group:
        tv = apply_transform_to_voxels(voxels, t)
        intersection = int((voxels & tv).sum())
        union = int((voxels | tv).sum())
        scores.append(intersection / union if union else 0.0)
    return float(np.mean(scores))


if __name__ == "__main__":
    # Quick smoke test with a synthetic asymmetric gamut
    R = 32
    # Asymmetric gamut: a corner of the cube
    g = np.zeros((R, R, R), dtype=bool)
    g[:R//2, :R//3, :R//4] = True  # very asymmetric

    from .symmetry_group import OH_GROUP, S3_GROUP

    print("Gamut volume fraction:", g.sum() / R**3)
    print("Symmetry score (Oh):", symmetry_score(g, OH_GROUP))

    result = orbit_coverage(g, OH_GROUP)
    print("Full Oh orbit coverage:", result)

    greedy = tiling_number_greedy(g, OH_GROUP, coverage_threshold=0.99)
    print("Greedy tiling number:", greedy["tiling_number"],
          "coverage:", greedy["final_coverage"])
