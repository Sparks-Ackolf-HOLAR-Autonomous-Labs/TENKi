"""
Visualiser for gamut symmetry analysis.

Plots:
  - 3D scatter of a gamut point cloud
  - 3D voxel grid of gamut + orbit
  - 2D projections (RG, RB, GB planes)
  - Coverage-vs-N-transforms curve
  - Symmetry score bar chart per subgroup
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from typing import Sequence

_OUT = Path(__file__).parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)


def plot_gamut_3d(
    pts: np.ndarray,
    title: str = "Gamut",
    save_path: Path | None = None,
    color: str | None = None,
    alpha: float = 0.15,
    subsample: int = 4000,
):
    """3D scatter plot of a gamut point cloud (pts: (N,3) in [0,1])."""
    if len(pts) > subsample:
        idx = np.random.choice(len(pts), subsample, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    if color is None:
        colors = pts  # use the RGB values as colours
    else:
        colors = color

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, alpha=alpha, s=4)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved {save_path}")
    plt.show()
    plt.close()


def plot_gamut_projections(
    pts: np.ndarray,
    title: str = "Gamut projections",
    save_path: Path | None = None,
):
    """2D projection plots on RG, RB, GB planes."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pairs = [(0, 1, "R", "G"), (0, 2, "R", "B"), (1, 2, "G", "B")]

    for ax, (i, j, xi, yj) in zip(axes, pairs):
        colors = np.zeros((len(pts), 4))
        colors[:, i] = pts[:, i]
        colors[:, j] = pts[:, j]
        colors[:, 3] = 0.3
        ax.scatter(pts[:, i], pts[:, j], c=pts[:, [i, j, max(0, 3-i-j)]], alpha=0.15, s=3)
        ax.set_xlabel(xi)
        ax.set_ylabel(yj)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{xi}–{yj} plane")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved {save_path}")
    plt.show()
    plt.close()


def plot_coverage_curve(
    records: list[dict],
    title: str = "Coverage vs transforms",
    save_path: Path | None = None,
    threshold: float | None = 0.99,
):
    """
    Plot coverage fraction vs number of transforms used.

    `records` is a list of dicts with keys 'n_transforms' and 'covered_fraction'.
    """
    ns = [r["n_transforms"] for r in records]
    covs = [r["covered_fraction"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, covs, marker=".", linewidth=1.5)
    ax.set_xlabel("Number of symmetry transforms used")
    ax.set_ylabel("Fraction of target covered")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", label=f"{threshold:.0%} threshold")
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved {save_path}")
    plt.show()
    plt.close()


def plot_symmetry_scores(
    engine_names: list[str],
    scores_by_subgroup: dict[str, list[float]],
    save_path: Path | None = None,
):
    """
    Bar chart: symmetry score per engine × subgroup.

    scores_by_subgroup: {subgroup_name: [score_engine_0, score_engine_1, ...]}
    """
    n_engines = len(engine_names)
    subgroups = list(scores_by_subgroup.keys())
    x = np.arange(n_engines)
    width = 0.8 / len(subgroups)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, sg in enumerate(subgroups):
        ax.bar(x + k * width, scores_by_subgroup[sg], width, label=sg)

    ax.set_xticks(x + width * (len(subgroups) - 1) / 2)
    ax.set_xticklabels(engine_names)
    ax.set_ylabel("Symmetry score (IoU-mean)")
    ax.set_title("Gamut symmetry per engine & subgroup")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved {save_path}")
    plt.show()
    plt.close()


def plot_tiling_summary(
    results: list[dict],
    save_path: Path | None = None,
):
    """
    Horizontal bar chart showing T(G) (tiling number) for each engine.

    `results`: list of dicts with 'engine', 'tiling_number', 'final_coverage'.
    """
    engines = [r["engine"] for r in results]
    tnums = [r["tiling_number"] for r in results]
    covs = [r["final_coverage"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].barh(engines, tnums, color="steelblue")
    axes[0].set_xlabel("Tiling number T(G)")
    axes[0].set_title("Minimum transforms to reach 99% coverage")
    axes[0].axvline(48, color="red", linestyle="--", label="|O_h| = 48")
    axes[0].legend()

    axes[1].barh(engines, [c * 100 for c in covs], color="seagreen")
    axes[1].set_xlabel("Final coverage (%)")
    axes[1].set_title("Coverage achieved with T(G) transforms")
    axes[1].axvline(99, color="red", linestyle="--", label="99% threshold")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved {save_path}")
    plt.show()
    plt.close()
