"""
Experiment 02 -- Directed Pairwise Transfer Matrix.

For every ordered pair of study databases (source, reference) computes:

    tau_ij(N) = Kendall tau between
                  policies ranked by N sub-sampled experiments from source i
                  policies ranked by ALL experiments from reference j   (fixed)

This is inherently directional: tau_ij(N) != tau_ji(N) at small N.

At N = 1  (single experiment):
    - A "donor" source produces an accurate ranking signal immediately.
    - A "receiver" source needs many experiments before its ranking stabilises.
    - The asymmetry matrix  A[i,j] = tau_ij(1) - tau_ji(1)  identifies
      which study is the donor in each pair.

At N = full (all experiments):
    - tau_ij(full) ~= tau_ji(full)  (Kendall tau converges to the same shared
      ceiling when both parties are at maximum information).

Outputs
-------
results/transfer_matrix_N<n>.png   -- heatmap of tau matrix at each N value
results/transfer_asymmetry_N1.png  -- heatmap of A[i,j] = tau_ij(1) - tau_ji(1)
results/transfer_matrix.json       -- matrices + bootstrap spreads + donor scores
"""

from __future__ import annotations

import argparse
import json
import sys
import os

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
from pathlib import Path

from analysis.flip_data import load_many_studies, common_policy_subset
from analysis.flip_metrics import bootstrap_tau_curve, kendall_tau
from analysis.flip_reports import plot_tau_matrix, plot_asymmetry_matrix, build_meta

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


def run(
    studies: dict[str, str] | None = None,
    n_values: list[int] | None = None,
    n_bootstrap: int = 300,
    score_key: str = "best_color_distance_mean",
    seed: int = 42,
) -> None:
    study_map = studies or DEFAULT_STUDIES
    n_vals    = n_values or [1, 5, 10]

    print("[02] Loading study databases...")
    all_studies = load_many_studies(study_map, score_key=score_key)
    if not all_studies:
        print("[02] No studies found -- check database paths.")
        return

    study_names = list(all_studies.keys())
    missing     = [k for k in study_map if k not in all_studies]
    if missing:
        print(f"[02] Missing studies (databases not found): {missing}")

    for name, study in all_studies.items():
        print(f"  {name}: {study.n_policies} policies, up to {study.max_n} experiments each")

    common = common_policy_subset(all_studies)
    print(f"\n  Common policies across all loaded studies: {len(common)}")
    if len(common) < 2:
        print("[02] Need at least 2 common policies -- aborting.")
        return

    K = len(study_names)
    all_matrices: dict[str, list] = {}
    all_matrices_std: dict[str, list] = {}

    for N in n_vals + ["full"]:
        print(f"\n  Building transfer matrix at N={N}...")
        mat     = np.zeros((K, K))
        mat_std = np.zeros((K, K))

        for i, src_name in enumerate(study_names):
            src = all_studies[src_name]
            for j, ref_name in enumerate(study_names):
                ref = all_studies[ref_name]
                if i == j:
                    mat[i, j] = 1.0
                    continue

                ref_rank = [p for p in ref.full_rank if p in common]

                if N == "full":
                    src_rank = [p for p in src.full_rank if p in common]
                    mat[i, j] = kendall_tau(src_rank, ref_rank)
                else:
                    curve = bootstrap_tau_curve(
                        src, ref_rank, common, [N], n_bootstrap,
                        rng_seed=seed + i * 1000 + j,
                    )
                    stats = curve.get(N, {})
                    mat[i, j]     = stats.get("mean_tau", float("nan"))
                    mat_std[i, j] = stats.get("std_tau",  0.0)

        all_matrices[str(N)] = mat.tolist()
        if N != "full":
            all_matrices_std[str(N)] = mat_std.tolist()

        # Print
        w = max(len(s) for s in study_names)
        header = f"{'':>{w}}" + "".join(f"  {s:>9}" for s in study_names)
        print(f"\n  N={N}:")
        print("  " + header)
        for i, src in enumerate(study_names):
            donor = np.nanmean([mat[i, j] for j in range(K) if j != i])
            row = (
                f"  {src:>{w}}"
                + "".join(f"  {mat[i, j]:>9.3f}" for j in range(K))
                + f"    donor_score={donor:.3f}"
            )
            print(row)

    # Asymmetry at minimum N (labelled as N=1 when available, else first N value)
    n_min_key = "1" if "1" in all_matrices else str(min(int(k) for k in all_matrices if k != "full"))
    mat_N1   = np.array(all_matrices[n_min_key])
    asym_mat = mat_N1 - mat_N1.T

    print(f"\n=== Directional asymmetry A[i,j] = tau_ij(N={n_min_key}) - tau_ji(N={n_min_key}) ===")
    print(f"  Positive = row is donor at N={n_min_key} (more info per experiment)")
    w = max(len(s) for s in study_names)
    for i, src in enumerate(study_names):
        row = f"  {src:>{w}}" + "".join(f"  {asym_mat[i, j]:>+9.3f}" for j in range(K))
        print(row)

    donor_N1 = {
        study_names[i]: float(
            np.nanmean([asym_mat[i, j] for j in range(K) if j != i])
        )
        for i in range(K)
    }
    print("\n  Net donor score at N=1 (mean asymmetry as source):")
    for name, score in sorted(donor_N1.items(), key=lambda x: -x[1]):
        tag = "DONOR" if score > 0.02 else "RECEIVER" if score < -0.02 else "neutral"
        print(f"    {name:>12}: {score:+.4f}  {tag}")

    # Plots
    for N_key, mat_list in all_matrices.items():
        mat = np.array(mat_list)
        plot_tau_matrix(
            mat, study_names,
            title=(
                f"Transfer matrix: tau(source at N={N_key} -> reference at full N)\n"
                f"Row mean = donor quality   ({n_bootstrap} bootstrap samples)"
            ),
            save_path=_OUT / f"transfer_matrix_N{N_key}.png",
        )
    plot_asymmetry_matrix(asym_mat, study_names, _OUT / f"transfer_asymmetry_N{n_min_key}.png")
    print("\n[02] Saved transfer_asymmetry_N1.png")

    # JSON
    meta = build_meta(
        config=dict(n_bootstrap=n_bootstrap, score_key=score_key, seed=seed,
                    n_values=n_vals),
        studies=study_names,
        common_policies=common,
        max_n_per_study={n: all_studies[n].max_n for n in study_names},
        missing_studies=missing,
    )
    out_data = {
        "meta":             meta,
        "studies":          study_names,
        "common_policies":  common,
        "n_bootstrap":      n_bootstrap,
        "n_values":         [str(v) for v in n_vals] + ["full"],
        "matrices":         all_matrices,
        "matrix_std":       all_matrices_std,
        "asymmetry_at_N1":  asym_mat.tolist(),
        "donor_score_N1":   donor_N1,
    }
    out_path = _OUT / "transfer_matrix.json"
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"[02] Saved {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Directed pairwise transfer matrix (Exp 02)")
    p.add_argument(
        "--studies", nargs="+", metavar="NAME=PATH",
        help="Studies as name=db_path pairs (default: hard-coded STUDIES dict)",
    )
    p.add_argument(
        "--n-values", nargs="+", type=int, default=[1, 5, 10],
        metavar="N", help="N values to evaluate (plus full)",
    )
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument(
        "--score-key", default="best_color_distance_mean",
        help="JSON field inside policy_stats to use as score",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Write results here instead of default results/")
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
        n_values=args.n_values,
        n_bootstrap=args.n_bootstrap,
        score_key=args.score_key,
        seed=args.seed,
    )
