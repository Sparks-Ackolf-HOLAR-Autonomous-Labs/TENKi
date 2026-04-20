"""
Experiment 03 -- Swarm Scaling & Directional Flip Test.

PRIMARY SCRIPT for the donor-flip question.

Two complementary scenarios are tested for every ordered pair (A, B):

  Scenario 1 -- External reference (high-fidelity source as ground truth)
    tau_AH(N) = Kendall tau of A's ranking from N experiments vs HF full
    tau_BH    = Kendall tau of B's full-data ranking vs HF full

    N*_ext = min N such that tau_AH(N) > ceiling(B->H) + eps
    Impossible when ceiling(A->H) <= ceiling(B->H) + eps  (PERMANENT_GAP).

  Scenario 2 -- Mutual reference (each source used as the other's ground truth)
    tau_AB(N) = tau of A's N-experiment ranking vs B's full ranking
    tau_BA(1) = tau of B's single-experiment ranking vs A's full ranking

    N*_mut = min N such that tau_AB(N) > tau_BA(1) + eps
    Because the shared ceiling is symmetric, a finite N* always exists for
    non-degenerate pairs if the scan range is large enough.

Outputs
-------
results/flip_external_all.png       -- Scenario 1 curves for all frugal sources
results/flip_crossover_heatmap.png  -- mutual gap@N=1 + N* heatmap
results/flip_test_summary.json      -- all FlipResult objects + curve data
results/flip_summary.md             -- human-readable Markdown summary
"""

from __future__ import annotations

import argparse
import sys
import os

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
from pathlib import Path

from analysis.flip_data import (
    load_many_studies,
    common_policy_subset,
    restrict_to_common_policies,
)
from analysis.flip_metrics import bootstrap_tau_curve, full_data_ceiling
from analysis.flip_models import (
    external_flip_result,
    mutual_flip_result,
    FlipResult,
)
from analysis.flip_reports import (
    plot_external_curves,
    plot_crossover_heatmap,
    write_flip_summary_json,
    write_flip_summary_markdown,
    build_meta,
)

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

DEFAULT_N_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
DEFAULT_HI_FI    = "spectral"


def run(
    studies: dict[str, str] | None = None,
    hifi: str = DEFAULT_HI_FI,
    n_values: list[int] | None = None,
    n_bootstrap: int = 300,
    flip_eps: float = 0.01,
    score_key: str = "best_color_distance_mean",
    seed: int = 42,
) -> None:
    study_map = studies or DEFAULT_STUDIES
    n_vals    = sorted(set(n_values or DEFAULT_N_VALUES))

    print("[03] Loading study databases...")
    all_studies = load_many_studies(study_map, score_key=score_key)
    if not all_studies:
        print("[03] No studies found -- check database paths.")
        return

    missing = [k for k in study_map if k not in all_studies]
    if missing:
        print(f"[03] Missing studies: {missing}")

    for name, study in all_studies.items():
        print(f"  {name}: {study.n_policies} policies, up to {study.max_n} experiments each")

    common = common_policy_subset(all_studies)
    print(f"  Common policies: {len(common)}")
    if len(common) < 2:
        print("[03] Need at least 2 common policies -- aborting.")
        return

    # Restrict every study to the common policy subset
    studies_common = {
        name: restrict_to_common_policies(study, common)
        for name, study in all_studies.items()
    }

    study_names = list(studies_common.keys())
    frugal      = [s for s in study_names if s != hifi]

    config = dict(
        hifi=hifi, n_values=n_vals, n_bootstrap=n_bootstrap,
        flip_eps=flip_eps, score_key=score_key, seed=seed,
    )

    all_results: list[FlipResult] = []

    # ------------------------------------------------------------------
    # Scenario 1: External reference
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"SCENARIO 1 -- External reference: {hifi}")
    print("=" * 60)

    if hifi not in studies_common:
        print(f"  [warn] HF study '{hifi}' not loaded -- skipping Scenario 1.")
        ext_curve_data: dict[str, tuple] = {}
    else:
        hifi_study  = studies_common[hifi]
        hifi_rank   = [p for p in hifi_study.full_rank if p in common]

        # Bootstrap curves for each frugal source vs HF
        ext_curve_data: dict[str, tuple] = {}
        for src_name in frugal:
            src    = studies_common[src_name]
            max_n  = src.max_n
            n_scan = [n for n in n_vals if n <= max_n]
            if max_n not in n_scan:
                n_scan.append(max_n)

            curve = bootstrap_tau_curve(
                src, hifi_rank, common, n_scan, n_bootstrap,
                rng_seed=seed + abs(hash(src_name)) % 10000,
            )
            means = [curve.get(n, {}).get("mean_tau", float("nan")) for n in n_scan]
            stds  = [curve.get(n, {}).get("std_tau",  0.0)          for n in n_scan]
            ext_curve_data[src_name] = (n_scan, means, stds)

            c = full_data_ceiling(src, hifi_study, common)
            print(f"  {src_name}: ceiling={c:.4f}  "
                  f"tau@N=1={means[0]:.3f}  tau@N={n_scan[-1]}={means[-1]:.3f}")

        # Pairwise external FlipResults
        for src_name in frugal:
            for comp_name in frugal:
                if src_name == comp_name:
                    continue
                src_max_n = studies_common[src_name].max_n
                n_scan_src = [n for n in n_vals if n <= src_max_n]
                if src_max_n not in n_scan_src:
                    n_scan_src.append(src_max_n)
                result = external_flip_result(
                    source=studies_common[src_name],
                    competitor=studies_common[comp_name],
                    hifi=hifi_study,
                    policies=common,
                    n_values=n_scan_src,
                    n_bootstrap=n_bootstrap,
                    eps=flip_eps,
                    rng_seed=seed + abs(hash(src_name + comp_name)) % 10000,
                )
                all_results.append(result)
                flip_str = f"  N*={result.flip_n}" if result.flip_n else ""
                print(f"  {src_name} over {comp_name}: "
                      f"gap_ceiling={result.gap_ceiling:+.4f}  "
                      f"{result.verdict}{flip_str}")

    # External plot
    ceilings_vs_hifi: dict[str, float] = {}
    if hifi in studies_common:
        for src_name in frugal:
            ceilings_vs_hifi[src_name] = full_data_ceiling(
                studies_common[src_name], studies_common[hifi], common
            )
    if ext_curve_data:
        plot_external_curves(
            ext_curve_data, ceilings_vs_hifi, hifi,
            _OUT / "flip_external_all.png",
        )
        print("\n[03] Saved flip_external_all.png")

    # ------------------------------------------------------------------
    # Scenario 2: Mutual reference
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 2 -- Mutual reference")
    print("=" * 60)

    mutual_gap_mat  = np.zeros((len(study_names), len(study_names)))
    mutual_nstar    = np.full((len(study_names), len(study_names)), np.nan)
    name_idx        = {n: i for i, n in enumerate(study_names)}

    for src_name in study_names:
        for comp_name in study_names:
            if src_name == comp_name:
                continue

            src  = studies_common[src_name]
            comp = studies_common[comp_name]

            max_na = src.max_n
            n_scan = [n for n in n_vals if n <= max_na]
            if max_na not in n_scan:
                n_scan.append(max_na)

            result = mutual_flip_result(
                source=src, competitor=comp, policies=common,
                n_values=n_scan, n_bootstrap=n_bootstrap,
                eps=flip_eps, rng_seed=seed + abs(hash(src_name + comp_name)) % 10000,
            )
            all_results.append(result)

            i, j = name_idx[src_name], name_idx[comp_name]
            mutual_gap_mat[i, j] = result.gap_now if not np.isnan(result.gap_now) else 0.0
            if result.flip_n is not None:
                mutual_nstar[i, j] = result.flip_n

            print(
                f"  {src_name} vs {comp_name}: "
                f"tau_AB(1)={result.tau_source_at_1:.3f}  "
                f"tau_BA(1)={result.tau_competitor_at_1:.3f}  "
                f"gap={result.gap_now:+.3f}  "
                f"N*={'~'+str(result.flip_n) if result.flip_n else 'none in range'}  "
                f"{result.verdict}"
            )

    # Crossover heatmap
    plot_crossover_heatmap(
        study_names, mutual_gap_mat, mutual_nstar,
        _OUT / "flip_crossover_heatmap.png",
    )
    print("\n[03] Saved flip_crossover_heatmap.png")

    # ------------------------------------------------------------------
    # JSON + Markdown outputs
    # ------------------------------------------------------------------
    meta = build_meta(
        config=config,
        studies=study_names,
        common_policies=common,
        max_n_per_study={n: studies_common[n].max_n for n in study_names},
        missing_studies=missing,
    )

    write_flip_summary_json(all_results, meta, _OUT / "flip_test_summary.json")
    print(f"[03] Saved {_OUT / 'flip_test_summary.json'}")

    write_flip_summary_markdown(
        all_results, study_names, hifi,
        _OUT / "flip_summary.md", meta=meta,
    )
    print(f"[03] Saved {_OUT / 'flip_summary.md'}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Donor-flip test (Exp 03)")
    p.add_argument("--hifi", default=DEFAULT_HI_FI,
                   help="Name of the high-fidelity reference study")
    p.add_argument("--frugal", nargs="+",
                   help="Names of frugal studies (must be keys in DEFAULT_STUDIES "
                        "or provided via --studies)")
    p.add_argument("--studies", nargs="+", metavar="NAME=PATH",
                   help="Studies as name=db_path pairs")
    p.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES,
                   metavar="N")
    p.add_argument("--n-bootstrap", type=int, default=300)
    p.add_argument("--flip-eps", type=float, default=0.01)
    p.add_argument("--score-key", default="best_color_distance_mean")
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
    elif args.frugal:
        study_map = {k: v for k, v in DEFAULT_STUDIES.items()
                     if k == args.hifi or k in args.frugal}
    run(
        studies=study_map,
        hifi=args.hifi,
        n_values=args.n_values,
        n_bootstrap=args.n_bootstrap,
        flip_eps=args.flip_eps,
        score_key=args.score_key,
        seed=args.seed,
    )
