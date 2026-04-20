"""
analysis/flip_reports.py — Plots and summary artifacts for the donor-flip pipeline.

Responsibilities
----------------
- Heatmaps for tau_ij(N) matrices.
- Heatmap for directional asymmetry A[i,j] = tau_ij(1) - tau_ji(1).
- External flip curves vs the HF reference.
- Mutual N* heatmap.
- Summary tables in Markdown and JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from analysis.flip_data import StudyScores
    from analysis.flip_models import FlipResult


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def plot_tau_matrix(
    mat: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
) -> None:
    """Generic Kendall-tau heatmap with cell annotations."""
    K = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, K * 1.1), max(5, K * 0.9)))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Reference  (full N)", fontsize=9)
    ax.set_ylabel("Source", fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(K):
        for j in range(K):
            ax.text(
                j, i, f"{mat[i, j]:.2f}",
                ha="center", va="center", fontsize=7, color="black",
            )
    plt.colorbar(im, ax=ax, label="Kendall tau")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_asymmetry_matrix(
    asym_mat: np.ndarray,
    labels: list[str],
    save_path: Path,
) -> None:
    """
    Red-Blue heatmap of A[i,j] = tau_ij(1) - tau_ji(1).
    Red = row is donor; Blue = row is receiver.
    """
    lim = max(0.01, float(np.abs(asym_mat).max()))
    plot_tau_matrix(
        asym_mat, labels,
        title=(
            "Directional asymmetry at N=1:  tau_ij(1) − tau_ji(1)\n"
            "Red = row is donor; Blue = row is receiver"
        ),
        save_path=save_path,
        vmin=-lim, vmax=lim, cmap="RdBu",
    )


# ---------------------------------------------------------------------------
# External flip curves
# ---------------------------------------------------------------------------

def plot_external_curves(
    frugal_curves: dict[str, tuple[list[int], list[float], list[float]]],
    ceilings: dict[str, float],
    hifi_name: str,
    save_path: Path,
) -> None:
    """
    All frugal-source curves on one axis, with dotted ceiling lines.

    Parameters
    ----------
    frugal_curves:
        {name: (n_values, means, stds)}
    ceilings:
        {name: ceiling_vs_hifi}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(frugal_curves)))

    for (src, (n_vals, means, stds)), color in zip(frugal_curves.items(), colors):
        ax.plot(n_vals, means, "-o", color=color, linewidth=2, label=src)
        ax.fill_between(
            n_vals,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.12, color=color,
        )
        ceiling = ceilings.get(src)
        if ceiling is not None:
            ax.axhline(ceiling, color=color, linestyle=":", linewidth=0.9)

    ax.set_xlabel("N experiments sub-sampled from source")
    ax.set_ylabel(f"Kendall tau vs {hifi_name} (full)")
    ax.set_title(
        f"Scenario 1 — External reference ({hifi_name})\n"
        "Dotted lines = ceilings.  Curves that cross = flippable pairs."
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Mutual crossover heatmap
# ---------------------------------------------------------------------------

def plot_crossover_heatmap(
    studies: list[str],
    gap_n1_mat: np.ndarray,
    n_star_mat: np.ndarray,
    save_path: Path,
) -> None:
    """
    Two-panel heatmap:
      Left  — mutual donor gap at N=1
      Right — N* (robots needed for row to flip over column)
    """
    K = len(studies)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel
    lim = max(0.01, float(np.abs(gap_n1_mat).max()))
    im1 = axes[0].imshow(gap_n1_mat, vmin=-lim, vmax=lim, cmap="RdBu", aspect="auto")
    axes[0].set_xticks(range(K))
    axes[0].set_yticks(range(K))
    axes[0].set_xticklabels(studies, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(studies, fontsize=8)
    axes[0].set_xlabel("Source B")
    axes[0].set_ylabel("Source A")
    axes[0].set_title(
        "Mutual donor gap at N=1:  tau_AB(1) − tau_BA(1)\n"
        "Red = A is donor; Blue = A is receiver"
    )
    for i in range(K):
        for j in range(K):
            if i != j:
                axes[0].text(
                    j, i, f"{gap_n1_mat[i, j]:+.2f}",
                    ha="center", va="center", fontsize=6,
                )
    plt.colorbar(im1, ax=axes[0])

    # Right panel
    cmap2 = plt.cm.YlGn.copy()
    cmap2.set_under("lightgray")
    vmax2 = max(
        1.0,
        float(np.nanmax(n_star_mat)) if not np.all(np.isnan(n_star_mat)) else 1.0,
    )
    disp = np.where(np.isnan(n_star_mat), -1.0, n_star_mat)
    im2 = axes[1].imshow(disp, vmin=0, vmax=vmax2, cmap=cmap2, aspect="auto")
    axes[1].set_xticks(range(K))
    axes[1].set_yticks(range(K))
    axes[1].set_xticklabels(studies, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(studies, fontsize=8)
    axes[1].set_xlabel("Source B")
    axes[1].set_ylabel("Source A")
    axes[1].set_title(
        "N* robots from A to surpass B@N=1\n"
        "Gray = not achieved within tested N range"
    )
    for i in range(K):
        for j in range(K):
            if i != j:
                fn = n_star_mat[i, j]
                lbl = str(int(fn)) if not np.isnan(fn) else ">"
                axes[1].text(j, i, lbl, ha="center", va="center", fontsize=7)
    plt.colorbar(im2, ax=axes[1], label="N* (experiments to flip)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Markdown and JSON summaries
# ---------------------------------------------------------------------------

def flip_result_to_dict(r: "FlipResult") -> dict:
    d = asdict(r)
    # Convert nan floats to None for JSON serialisability
    for k, v in d.items():
        if isinstance(v, float) and np.isnan(v):
            d[k] = None
    return d


def write_flip_summary_json(
    results: list["FlipResult"],
    meta: dict,
    save_path: Path,
) -> None:
    """Write a machine-readable JSON summary of all FlipResult objects."""
    out = {
        "meta": meta,
        "results": [flip_result_to_dict(r) for r in results],
    }
    with open(save_path, "w") as fh:
        json.dump(out, fh, indent=2)


def write_flip_summary_markdown(
    results: list["FlipResult"],
    studies: list[str],
    hifi_name: str,
    save_path: Path,
    meta: dict | None = None,
) -> None:
    """
    Write a human-readable Markdown summary.

    Sections:
    1. Donor/receiver ordering at N=1
    2. External permanent-gap pairs
    3. External flippable pairs with N*
    4. Mutual flip matrix
    5. Top candidates for cheap-source replacement
    """
    lines: list[str] = []

    ts = (meta or {}).get("timestamp", datetime.now(timezone.utc).isoformat())
    lines.append(f"# Donor-Flip Summary\n")
    lines.append(f"Generated: {ts}  \n")
    if meta:
        cfg = meta.get("config", {})
        if cfg:
            lines.append(f"n_bootstrap={cfg.get('n_bootstrap')}  "
                         f"flip_eps={cfg.get('flip_eps')}  "
                         f"hifi={cfg.get('hifi')}  \n")
    lines.append("")

    # ── 1. Donor / receiver ordering at N=1 ─────────────────────────────────
    lines.append("## 1. Donor/receiver ordering at N=1 (external reference)\n")
    ext_at_1: dict[str, float] = {}
    for r in results:
        if r.mode == "external" and not np.isnan(r.tau_source_at_1 or float("nan")):
            ext_at_1.setdefault(r.source, r.tau_source_at_1)
    if ext_at_1:
        lines.append("| Source | tau@N=1 vs HF | Ceiling | Role |")
        lines.append("|--------|--------------|---------|------|")
        for src, t in sorted(ext_at_1.items(), key=lambda x: -(x[1] or 0)):
            ceiling = next(
                (r.ceiling_source for r in results
                 if r.mode == "external" and r.source == src),
                None,
            )
            c_str = f"{ceiling:.3f}" if ceiling is not None else "—"
            role = "DONOR" if (t or 0) > 0.5 else "RECEIVER"
            lines.append(f"| {src} | {t:.3f} | {c_str} | {role} |")
    lines.append("")

    # ── 2. External permanent-gap pairs ─────────────────────────────────────
    perm = [r for r in results if r.mode == "external" and r.verdict == "PERMANENT_GAP"]
    lines.append("## 2. External permanent-gap pairs (physics/calibration limits)\n")
    if perm:
        lines.append("| Source | Competitor | gap_ceiling | Note |")
        lines.append("|--------|-----------|-------------|------|")
        for r in sorted(perm, key=lambda x: x.gap_ceiling):
            lines.append(
                f"| {r.source} | {r.competitor} | "
                f"{r.gap_ceiling:+.4f} | "
                f"physics-limited (irreducible) |"
            )
    else:
        lines.append("_None detected within tested range._")
    lines.append("")

    # ── 3. External flippable pairs with N* ──────────────────────────────────
    flip_ext = [r for r in results if r.mode == "external" and r.verdict == "FLIPPABLE"]
    lines.append("## 3. External flippable pairs with N\\*\n")
    if flip_ext:
        lines.append("| Source | Competitor | N* | gap_now | gap_ceiling |")
        lines.append("|--------|-----------|-----|---------|-------------|")
        for r in sorted(flip_ext, key=lambda x: x.flip_n or 999999):
            lines.append(
                f"| {r.source} | {r.competitor} | "
                f"{r.flip_n} | "
                f"{r.gap_now:+.4f} | "
                f"{r.gap_ceiling:+.4f} |"
            )
    else:
        lines.append("_None detected within tested N range._")
    lines.append("")

    # ── 4. Mutual flip matrix ────────────────────────────────────────────────
    lines.append("## 4. Mutual flip matrix (N* for A to surpass B@N=1)\n")
    mutual = [r for r in results if r.mode == "mutual"]
    if mutual and studies:
        header = "| Source \\ Competitor | " + " | ".join(studies) + " |"
        sep    = "|" + "---|" * (len(studies) + 1)
        lines.append(header)
        lines.append(sep)
        # Index lookup
        n_star: dict[tuple[str, str], str] = {}
        for r in mutual:
            val = str(r.flip_n) if r.flip_n is not None else ">"
            n_star[(r.source, r.competitor)] = val
        for src in studies:
            row_parts = [src]
            for comp in studies:
                if src == comp:
                    row_parts.append("—")
                else:
                    row_parts.append(n_star.get((src, comp), "—"))
            lines.append("| " + " | ".join(row_parts) + " |")
    lines.append("")

    # ── 5. Top candidates for cheap-source replacement ───────────────────────
    lines.append("## 5. Top candidates for cheap-source replacement\n")
    cheap = [
        r for r in results
        if r.mode == "mutual" and r.verdict in ("FLIPPABLE", "ALREADY_DONOR")
        and r.flip_n is not None
    ]
    if cheap:
        lines.append(
            "Sources that can replace a stronger donor at minimal experiment cost:\n"
        )
        lines.append("| Source | vs Competitor | N* | Verdict |")
        lines.append("|--------|-------------|-----|---------|")
        for r in sorted(cheap, key=lambda x: x.flip_n or 999999):
            lines.append(
                f"| {r.source} | {r.competitor} | {r.flip_n} | {r.verdict} |"
            )
    else:
        lines.append("_No clear cheap-source replacement candidates in tested range._")
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")


def build_meta(
    config: dict,
    studies: list[str],
    common_policies: list[str],
    max_n_per_study: dict[str, int],
    missing_studies: list[str] | None = None,
) -> dict:
    """Build a standard metadata block for every JSON artifact."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "common_policy_count": len(common_policies),
        "common_policies": common_policies,
        "studies_used": studies,
        "missing_studies": missing_studies or [],
        "max_n_per_study": max_n_per_study,
    }
