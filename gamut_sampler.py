"""
Gamut sampler — densely samples each engine's reachable color gamut.

Strategy: enumerate a grid of (red%, yellow%, blue%, water%) actions, pass each
through the engine, record the resulting RGB.  The resulting point cloud is the
empirical gamut.

All RGB values are normalised to [0, 1] for downstream analysis.
"""

from __future__ import annotations
import sys, os
# Allow running from any working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Literal

EngineType = Literal["spectral", "mixbox", "kubelka_munk", "coloraide_ryb"]

# Cache directory next to this file
_CACHE_DIR = Path(__file__).parent / "results" / "gamut_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _engine_instance(engine_type: EngineType):
    """Lazy-import and instantiate an engine."""
    if engine_type == "spectral":
        from src.color_mixing.spectral_engine import SpectralColorMixingEngine
        return SpectralColorMixingEngine()
    elif engine_type == "mixbox":
        from src.color_mixing.mixbox_engine import MixboxColorMixingEngine
        return MixboxColorMixingEngine()
    elif engine_type == "kubelka_munk":
        from src.color_mixing.kubelka_munk_engine import KubelkaMunkEngine
        return KubelkaMunkEngine()
    elif engine_type == "coloraide_ryb":
        from src.color_mixing.coloraide_ryb_engine import ColoraideRYBEngine
        return ColoraideRYBEngine()
    else:
        raise ValueError(f"Unknown engine: {engine_type}")


def sample_gamut(
    engine_type: EngineType,
    steps: int = 21,
    noise_level: float = 0.0,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Return a (N, 3) float32 array of RGB points in [0, 1] that are reachable
    by the given engine.

    Parameters
    ----------
    engine_type : one of the four supported engine names
    steps : number of grid steps per axis (steps^3 action combinations tried,
            but water is determined as 100 - sum(others), so effectively steps^3
            valid points; invalid combos are skipped)
    noise_level : noise added by the engine (0.0 = deterministic)
    use_cache : if True, load/save to a .npy cache file
    """
    cache_key = f"{engine_type}_s{steps}_n{noise_level:.3f}"
    cache_path = _CACHE_DIR / f"{cache_key}.npy"

    if use_cache and cache_path.exists():
        pts = np.load(cache_path)
        print(f"[sampler] Loaded {len(pts):,} points from cache: {cache_path.name}")
        return pts

    engine = _engine_instance(engine_type)

    from src.simulation.core.actions import ColorMixingAction

    vals = np.linspace(0, 100, steps)
    results = []
    seed = 0

    for r in vals:
        for y in vals:
            for b in vals:
                w = 100.0 - r - y - b
                if w < 0.0:
                    continue
                action = ColorMixingAction(
                    red_percent=float(r),
                    yellow_percent=float(y),
                    blue_percent=float(b),
                    water_percent=float(w),
                )
                try:
                    rgb = engine.mix_colors(action, noise_level=noise_level, seed=seed)
                    # Normalise: engine may return 0-1 or 0-255
                    rgb = np.array(rgb, dtype=float)
                    if rgb.max() > 1.5:
                        rgb = rgb / 255.0
                    rgb = np.clip(rgb, 0.0, 1.0)
                    results.append(rgb)
                except Exception:
                    pass
                seed += 1

    pts = np.array(results, dtype=np.float32)
    print(f"[sampler] Sampled {len(pts):,} points for engine '{engine_type}'")

    if use_cache:
        np.save(cache_path, pts)
        print(f"[sampler] Saved to {cache_path}")

    return pts


def gamut_bounds(pts: np.ndarray) -> dict:
    """
    Return axis-aligned bounding box and basic statistics of a gamut point cloud.
    All values in [0, 1].
    """
    return {
        "min": pts.min(axis=0).tolist(),
        "max": pts.max(axis=0).tolist(),
        "mean": pts.mean(axis=0).tolist(),
        "std": pts.std(axis=0).tolist(),
        "n_points": len(pts),
        "bbox_volume": float(np.prod(pts.max(axis=0) - pts.min(axis=0))),
    }


def gamut_convex_hull_volume(pts: np.ndarray) -> float:
    """
    Compute the volume of the convex hull of the gamut (fraction of unit cube).
    Requires scipy.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # absolute volume in [0,1]^3
    except Exception as e:
        print(f"[sampler] ConvexHull failed: {e}")
        return float("nan")


def voxelise(pts: np.ndarray, resolution: int = 64) -> np.ndarray:
    """
    Discretise a point cloud to a boolean voxel grid of shape (res, res, res).
    Values in [0,1] are mapped to voxel indices [0, res-1].
    """
    idx = np.clip((pts * resolution).astype(int), 0, resolution - 1)
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


if __name__ == "__main__":
    for eng in ["spectral", "mixbox"]:
        pts = sample_gamut(eng, steps=16)
        info = gamut_bounds(pts)
        print(f"\n=== {eng} ===")
        for k, v in info.items():
            print(f"  {k}: {v}")
        vol = gamut_convex_hull_volume(pts)
        print(f"  convex_hull_volume (fraction of unit cube): {vol:.4f}")
