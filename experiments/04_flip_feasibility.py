"""
Experiment 04 -- Flip Feasibility Map (taxonomy only).

Consumes results from experiments 02 and 03 and computes:
  - donor centrality (external and mutual)
  - directed donor topology with cycle detection
  - PEGKi-aligned interpretation fields

This script is a POST-PROCESSING step.  Run experiments 02 and 03 first.

Definitions
-----------
external_ceiling(X) = tau(X full -> HF full)
    Different sources have different physics gaps relative to the HF source,
    so ceiling(study_c -> spectral) != ceiling(study_b -> spectral).

external_gap(A, B) = external_ceiling(A) - external_ceiling(B)
    > 0  A is already a better proxy for spectral -> FLIPPABLE_NOW
    < 0  A has a PERMANENT physics gap; N->inf cannot overcome it -> PERMANENT_GAP

mutual_gap_at_N1(A, B) = tau_AB(N=1) - tau_BA(N=1)
    > 0  A is currently the donor at N=1

Outputs
-------
results/flip_feasibility.json
results/flip_feasibility_summary.png  -- centrality bar chart + external gap matrix
results/flip_topology.png             -- directed donor graph (requires networkx)
"""

from __future__ import annotations

import argparse
import json
import sys
import os

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from analysis.flip_data import load_many_studies, common_policy_subset
from analysis.flip_metrics import full_data_ceiling

_OUT = Path(_HERE).parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_STUDIES = {
    "spectral": "output/db_spectral",
    "mixbox":   "output/db_mixbox",
    "km":       "output/db_km",
    "ryb":      "output/db_ryb",
    "study_a":  "output/db_study_a_artist_consensus",
    "study_b":  "output/db_study_b_physics_vs_artist",
    "study_c":  "output/db_study_c_oilpaint_vs_fooddye",
}

DEFAULT_HI_FI = "spectral"
DEFAULT_EPS   = 0.02


def run(
    studies: dict[str, str] | None = None,
    hifi: str = DEFAULT_HI_FI,
    flip_eps: float = DEFAULT_EPS,
    score_key: str = "best_color_distance_mean",
) -> None:
    study_map = studies or DEFAULT_STUDIES

    print("[04] Loading study databases...")
    all_studies = load_many_studies(study_map, score_key=score_key)
    if not all_studies:
        print("[04] No studies found.")
        return

    study_names = list(all_studies.keys())
    common = common_policy_subset(all_studies)
    print(f"  Studies loaded: {study_names}")
    print(f"  Common policies: {len(common)}")
    if len(common) < 2:
        print("[04] Need at least 2 common policies -- aborting.")
        return

    # ------------------------------------------------------------------
    # External ceilings: tau(X full -> HF full)
    # ------------------------------------------------------------------
    ext_ceiling: dict[str, float] = {}
    if hifi in all_studies:
        hifi_study = all_studies[hifi]
        for name, study in all_studies.items():
            ext_ceiling[name] = (
                1.0 if name == hifi
                else full_data_ceiling(study, hifi_study, common)
            )
    else:
        print(f"  [warn] HF study '{hifi}' not loaded -- external ceilings unavailable.")

    K = len(study_names)

    # External gap matrix
    ext_gap_mat = np.zeros((K, K))
    for i, sa in enumerate(study_names):
        for j, sb in enumerate(study_names):
            if sa != sb and sa in ext_ceiling and sb in ext_ceiling:
                ext_gap_mat[i, j] = ext_ceiling[sa] - ext_ceiling[sb]

    print("\n=== External gap: external_ceiling(A) - external_ceiling(B) ===")
    print("  > 0  A is a better spectral proxy (FLIPPABLE_NOW)")
    print("  < 0  PERMANENT physics gap (A can never catch B vs spectral)\n")
    for i, sa in enumerate(study_names):
        for j, sb in enumerate(study_names):
            if sa == sb:
                continue
            g = ext_gap_mat[i, j]
            verdict = (
                "FLIPPABLE_NOW"  if g > flip_eps else
                "PERMANENT_GAP" if g < -flip_eps else
                "SYMMETRIC"
            )
            print(f"  {sa:>10} vs {sb:>10}: ext_gap={g:+.4f}  {verdict}")

    # ------------------------------------------------------------------
    # External donor centrality
    # ------------------------------------------------------------------
    print("\n=== External donor centrality (ext_ceiling vs HF) ===")
    ext_centrality = {s: float(ext_ceiling.get(s, float("nan"))) for s in study_names}
    for name, score in sorted(ext_centrality.items(), key=lambda x: -(x[1] or 0)):
        if np.isnan(score):
            print(f"  {name:>12}: ext_ceiling=N/A")
            continue
        role = (
            "NET DONOR"  if score > 0.85 else
            "RECEIVER"   if score < 0.70 else
            "MIDDLE"
        )
        print(f"  {name:>12}: ext_ceiling={score:.4f}  {role}")

    # ------------------------------------------------------------------
    # Mutual N=1 asymmetry from exp 02 output (if available)
    # ------------------------------------------------------------------
    asym_N1: dict[tuple[str, str], float] = {}
    transfer_path = _OUT / "transfer_matrix.json"
    if transfer_path.exists():
        with open(transfer_path) as fh:
            td = json.load(fh)
        mats    = td["matrices"]
        n1_key  = "1" if "1" in mats else str(min(int(k) for k in mats if k != "full"))
        mat_N1  = np.array(mats[n1_key])
        s_list  = td["studies"]
        for i, si in enumerate(s_list):
            for j, sj in enumerate(s_list):
                if i != j:
                    asym_N1[(si, sj)] = float(mat_N1[i, j] - mat_N1[j, i])
        print(f"\n  Loaded N={n1_key} asymmetry from transfer_matrix.json")
    else:
        print("\n  [warn] transfer_matrix.json not found -- run exp 02 first.")

    mutual_centrality: dict[str, float] = {}
    if asym_N1:
        print("\n=== Mutual donor centrality at N=1 ===")
        for s in study_names:
            vals = [asym_N1.get((s, r), 0.0) for r in study_names if r != s]
            mutual_centrality[s] = float(np.mean(vals)) if vals else 0.0
        for name, score in sorted(mutual_centrality.items(), key=lambda x: -x[1]):
            role = (
                "NET DONOR"  if score > flip_eps else
                "RECEIVER"   if score < -flip_eps else
                "NEUTRAL"
            )
            print(f"  {name:>12}: mutual_centrality={score:+.4f}  {role}")

    # ------------------------------------------------------------------
    # Mutual flip N* from exp 03 output (if available)
    # ------------------------------------------------------------------
    flip_n_mutual: dict[tuple[str, str], int | None] = {}
    flip_path = _OUT / "flip_test_summary.json"
    if flip_path.exists():
        with open(flip_path) as fh:
            fd = json.load(fh)
        for rdict in fd.get("results", []):
            if rdict.get("mode") == "mutual":
                a, b = rdict["source"], rdict["competitor"]
                flip_n_mutual[(a, b)] = rdict.get("flip_n")
        print("  Loaded mutual N* from flip_test_summary.json")
    else:
        print("  [warn] flip_test_summary.json not found -- run exp 03 first.")

    if flip_n_mutual:
        print("\n=== Mutual flip N* -- experiments from A to surpass B@N=1 ===")
        n_star_mat = np.full((K, K), np.nan)
        for i, a in enumerate(study_names):
            for j, b in enumerate(study_names):
                if a != b and (a, b) in flip_n_mutual:
                    fn = flip_n_mutual[(a, b)]
                    n_star_mat[i, j] = fn if fn is not None else np.nan
        w = max(len(s) for s in study_names)
        header = f"{'':>{w}}" + "".join(f"  {s:>9}" for s in study_names)
        print("  " + header)
        for i, s in enumerate(study_names):
            row = f"  {s:>{w}}"
            for j in range(K):
                fn  = n_star_mat[i, j]
                lbl = f"{int(fn):>9}" if not np.isnan(fn) else f"{'?':>9}"
                row += f"  {lbl}"
            print(row)

    # ------------------------------------------------------------------
    # 3-cycle detection on external donor graph
    # ------------------------------------------------------------------
    print("\n=== 3-cycles in external donor graph (intransitive donors) ===")
    adj: dict[str, set[str]] = {s: set() for s in study_names}
    for i, sa in enumerate(study_names):
        for j, sb in enumerate(study_names):
            if sa != sb and ext_gap_mat[i, j] > flip_eps:
                adj[sa].add(sb)

    n_cycles = 0
    seen: set[tuple] = set()
    for a in study_names:
        for b in adj[a]:
            for c in adj[b]:
                if a in adj[c]:
                    key = tuple(sorted([a, b, c]))
                    if key not in seen:
                        seen.add(key)
                        print(f"  CYCLE: {a} -> {b} -> {c} -> {a}  "
                              f"(Blade-Chest intransitivity candidate)")
                        n_cycles += 1
    if n_cycles == 0:
        print("  None -- donor hierarchy is fully transitive")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    lim = max(0.01, float(np.abs(ext_gap_mat).max())) if ext_ceiling else 0.01
    im1 = axes[0].imshow(ext_gap_mat, vmin=-lim, vmax=lim, cmap="RdBu", aspect="auto")
    axes[0].set_xticks(range(K))
    axes[0].set_yticks(range(K))
    axes[0].set_xticklabels(study_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(study_names, fontsize=8)
    axes[0].set_xlabel("Competitor B")
    axes[0].set_ylabel("Source A")
    axes[0].set_title(
        f"External gap: ext_ceiling(A) - ext_ceiling(B)\n"
        f"Red = A already better proxy for {hifi}; "
        f"Blue = PERMANENT GAP"
    )
    for i in range(K):
        for j in range(K):
            if i != j:
                axes[0].text(
                    j, i, f"{ext_gap_mat[i, j]:+.2f}",
                    ha="center", va="center", fontsize=6,
                )
    plt.colorbar(im1, ax=axes[0])

    ext_names  = sorted(ext_centrality, key=lambda x: -(ext_centrality.get(x) or 0))
    ext_scores = [ext_centrality.get(n, 0.0) for n in ext_names]

    def _bar_color(score: float) -> str:
        if score > 0.85: return "steelblue"
        if score < 0.70: return "salmon"
        return "lightgray"

    bar_colors = [_bar_color(s) for s in ext_scores]
    axes[1].barh(range(len(ext_names)), ext_scores, color=bar_colors, alpha=0.85)
    axes[1].set_yticks(range(len(ext_names)))
    axes[1].set_yticklabels(ext_names, fontsize=9)
    axes[1].set_xlabel(f"External ceiling  (tau vs {hifi} at full N)")
    axes[1].set_title(
        f"External donor centrality\n"
        f"(higher = better donor for {hifi})"
    )
    axes[1].axvline(0.85, color="steelblue", linestyle="--", linewidth=1)
    axes[1].axvline(0.70, color="salmon",    linestyle="--", linewidth=1)
    legend_patches = [
        mpatches.Patch(color="steelblue", label="NET DONOR   (ceiling > 0.85)"),
        mpatches.Patch(color="salmon",    label="NET RECEIVER (ceiling < 0.70)"),
        mpatches.Patch(color="lightgray", label="MIDDLE"),
    ]
    axes[1].legend(handles=legend_patches, fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="x")
    axes[1].set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(_OUT / "flip_feasibility_summary.png", dpi=150)
    plt.close()
    print("\n[04] Saved flip_feasibility_summary.png")

    # Optional networkx topology
    try:
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(study_names)
        for sa in study_names:
            for sb in adj[sa]:
                G.add_edge(sa, sb, weight=float(ext_ceiling.get(sa, 0)) - float(ext_ceiling.get(sb, 0)))

        fig, ax = plt.subplots(figsize=(8, 7))
        pos = nx.spring_layout(G, seed=42, k=2)
        node_colors = [_bar_color(ext_centrality.get(n, 0.0)) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        nx.draw_networkx_edges(
            G, pos, arrows=True, arrowsize=20,
            edge_color="gray", ax=ax, connectionstyle="arc3,rad=0.1",
        )
        edge_labels = {(u, v): f"{d['weight']:+.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
        ax.set_title(
            f"External donor topology  "
            f"(A->B iff ext_ceiling_A > ext_ceiling_B + {flip_eps})\n"
            f"Blue = NET DONOR, Red = NET RECEIVER."
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(_OUT / "flip_topology.png", dpi=150)
        plt.close()
        print("[04] Saved flip_topology.png")
    except ImportError:
        print("[04] networkx not installed -- skipping topology plot")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    out_data = {
        "studies":              study_names,
        "hifi":                 hifi,
        "flip_eps":             flip_eps,
        "external_ceilings":    ext_ceiling,
        "external_gap_matrix":  ext_gap_mat.tolist(),
        "external_centrality":  ext_centrality,
        "mutual_centrality_N1": mutual_centrality,
        "mutual_flip_n": {
            f"{a}_{b}": flip_n_mutual.get((a, b))
            for a in study_names for b in study_names if a != b
        },
        "n_cycles": n_cycles,
        # PEGKi-aligned fields
        "pegki_interpretation": {
            "directionality_note": (
                "Asymmetry is explained by coverage mismatch, calibration basis, "
                "and rank-geometry mismatch (PEGKi Conjecture 4.2.1)."
            ),
            "impossible_flips_note": (
                "PERMANENT_GAP pairs represent irreducible knowledge across "
                "fidelities (PEGKi Conjecture 4.2.2). "
                "These are physics-limited, not sampling-limited."
            ),
            "cycles_note": (
                f"{n_cycles} 3-cycles detected. "
                "Cycles are Blade-Chest intransitivity candidates "
                "and should be analysed with the intransitivity pipeline."
            ),
        },
    }
    out_path = _OUT / "flip_feasibility.json"
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"[04] Saved {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flip feasibility taxonomy (Exp 04)")
    p.add_argument("--hifi",      default=DEFAULT_HI_FI)
    p.add_argument("--flip-eps",  type=float, default=DEFAULT_EPS)
    p.add_argument("--score-key", default="best_color_distance_mean")
    p.add_argument("--studies",   nargs="+", metavar="NAME=PATH",
                   help="Studies as name=db_path pairs")
    p.add_argument("--output-dir", default=None, metavar="DIR")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    study_map = None
    if args.studies:
        study_map = {}
        for item in args.studies:
            name, _, path = item.partition("=")
            study_map[name.strip()] = path.strip()
    run(
        studies=study_map,
        hifi=args.hifi,
        flip_eps=args.flip_eps,
        score_key=args.score_key,
    )
