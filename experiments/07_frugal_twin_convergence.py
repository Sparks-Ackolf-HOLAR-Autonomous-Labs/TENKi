"""
Experiment 07 -- Frugal Twin Convergence.

Framing (swarm intelligence):
  - Each policy experiment = one "robot" making N measurements of the color space
  - "High-fidelity" reference = spectral engine (full Beer-Lambert + CIE D65 physics)
  - "Frugal twin"  = any cheaper study (mixbox, RYB, KM, set-op A/B/C)
  - "Symmetry" = frugal swarm ranking agrees with high-fidelity ranking (tau >= threshold)

Three questions answered:
  Q1. Internal convergence: How many robots (N experiments) from a single study are needed
      for that study's ranking to stabilize?
  Q2. Cross-study match: How many frugal robots from study X are needed to match the HF
      source? (limited by bias, not just variance)
  Q3. Swarm diversity: If you combine robots from K different studies, how quickly does
      the combined ranking converge to the HF source?

Swarm allocation modes (--swarm-mode):
  equal            -- split total N equally across all studies
  rho_squared      -- weight proportional to Spearman rho^2 (quality-aware)
  inverse_variance -- weight inversely proportional to bootstrap variance at N=1

Outputs
-------
results/frugal_twin_convergence.png   -- Q1+Q2 convergence curves, Q3 swarm diversity bar
results/frugal_twin_convergence.json  -- all convergence stats + swarm results
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from itertools import combinations

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.flip_data import load_many_studies, common_policy_subset
from analysis.flip_metrics import (
    bootstrap_tau_curve,
    full_data_ceiling,
    kendall_tau,
    rank_from_sample,
    spearman_rho,
)
from analysis.flip_data import StudyScores

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

DEFAULT_HI_FI             = "spectral"
DEFAULT_FRUGAL            = ["study_a", "study_b", "study_c", "mixbox", "ryb", "km"]
CONVERGENCE_THRESHOLD     = 0.80


# ---------------------------------------------------------------------------
# Weighted swarm helper
# ---------------------------------------------------------------------------

def combine_scores_weighted(
    score_dicts: list[StudyScores],
    n_allocations: dict[str, int],
    rng: np.random.Generator,
    policies: list[str],
) -> list[str]:
    """
    Pool experiments from multiple studies according to per-study allocations.

    Parameters
    ----------
    score_dicts:
        List of StudyScores to combine.
    n_allocations:
        {study_name: number_of_experiments_to_draw}
    rng:
        Random number generator.
    policies:
        Policy subset to rank.

    Returns
    -------
    Ranked list of policies (ascending, lower score = better).
    """
    combined_means: dict[str, list[float]] = {p: [] for p in policies}
    for study in score_dicts:
        n = n_allocations.get(study.name, 0)
        if n <= 0:
            continue
        for pol in policies:
            if pol not in study.policy_scores:
                continue
            sample = rng.choice(
                study.policy_scores[pol],
                size=n,
                replace=True,
            )
            combined_means[pol].extend(sample.tolist())

    agg = {p: float(np.mean(vals)) for p, vals in combined_means.items() if vals}
    return sorted(agg, key=agg.__getitem__)


def _compute_swarm_weights(
    frugal_studies: list[StudyScores],
    hifi_rank: list[str],
    policies: list[str],
    mode: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """
    Compute quality-aware swarm weights for frugal studies.

    Modes
    -----
    equal          -- uniform weights
    rho_squared    -- proportional to Spearman rho^2 at full N vs HF
    inverse_variance -- proportional to 1/variance_at_N1 (bootstrap)
    """
    if mode == "equal":
        n = len(frugal_studies)
        return {s.name: 1.0 / n for s in frugal_studies}

    if mode == "rho_squared":
        rho_sq: dict[str, float] = {}
        for study in frugal_studies:
            src_rank = [p for p in study.full_rank if p in policies]
            rho = spearman_rho(src_rank, hifi_rank)
            rho_sq[study.name] = max(0.0, rho) ** 2
        total = sum(rho_sq.values()) or 1.0
        return {n: v / total for n, v in rho_sq.items()}

    if mode == "inverse_variance":
        inv_var: dict[str, float] = {}
        for study in frugal_studies:
            curve = bootstrap_tau_curve(
                study, hifi_rank, policies, [1], n_bootstrap,
                rng_seed=seed + abs(hash(study.name)) % 10000,
            )
            std = curve.get(1, {}).get("std_tau", 1.0)
            inv_var[study.name] = 1.0 / (std ** 2 + 1e-9)
        total = sum(inv_var.values()) or 1.0
        return {n: v / total for n, v in inv_var.items()}

    raise ValueError(f"Unknown swarm mode: {mode!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    studies: dict[str, str] | None = None,
    hifi: str = DEFAULT_HI_FI,
    frugal_names: list[str] | None = None,
    swarm_mode: str = "equal",
    n_robots_per_study: int = 10,
    n_bootstrap: int = 200,
    score_key: str = "best_color_distance_mean",
    seed: int = 42,
) -> None:
    study_map = studies or DEFAULT_STUDIES

    print("[07] Loading study databases...")
    all_studies = load_many_studies(study_map, score_key=score_key)
    if not all_studies:
        print("[07] No studies found.")
        return

    for name, study in all_studies.items():
        print(f"  {name}: {study.n_policies} policies, up to {study.max_n} experiments each")

    common = common_policy_subset(all_studies)
    print(f"  Common policies: {len(common)}")
    if len(common) < 2:
        print("[07] Need at least 2 common policies -- aborting.")
        return

    if hifi not in all_studies:
        print(f"[07] HF study '{hifi}' not found.")
        return

    hifi_study = all_studies[hifi]
    hifi_rank  = [p for p in hifi_study.full_rank if p in common]

    _requested_frugal = frugal_names or DEFAULT_FRUGAL
    available_frugal  = [n for n in _requested_frugal if n in all_studies]
    if not available_frugal:
        print(f"[07] No frugal studies found from: {_requested_frugal}")
        return

    max_n = min(
        min(all_studies[hifi].max_n, 100),
        min(all_studies[f].max_n for f in available_frugal),
    )
    n_values = sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 30, 50, max_n]))

    # -- Q1 + Q2: Per-study convergence curves --------------------------------
    print("\n=== Q1+Q2: Per-study convergence (tau vs N robots) ===")
    convergence_results: dict[str, dict] = {}
    for name in [hifi] + available_frugal:
        study  = all_studies[name]
        curve  = bootstrap_tau_curve(
            study, hifi_rank, common, n_values, n_bootstrap,
            rng_seed=seed + abs(hash(name)) % 10000,
        )
        convergence_results[name] = curve

        n_thresh = next(
            (n for n in n_values if curve.get(n, {}).get("mean_tau", 0) >= CONVERGENCE_THRESHOLD),
            None,
        )
        final_tau = curve.get(max_n, {}).get("mean_tau", float("nan"))
        bias      = 1.0 - final_tau if not np.isnan(final_tau) else float("nan")
        print(
            f"  {name:<12}: tau@N=1={curve.get(1,{}).get('mean_tau',0):.3f}  "
            f"tau@N={max_n}={final_tau:.3f}  "
            f"bias={bias:.3f}  "
            f"N_thresh(tau>={CONVERGENCE_THRESHOLD})="
            f"{n_thresh if n_thresh else '>'+str(max_n)}"
        )

    # -- Q3: Swarm diversity -- combine K studies at FIXED total budget ----------
    # n_robots_per_study is reused as the fixed total budget N_TOTAL so that
    # diversity (k) is isolated from budget size.
    N_TOTAL = n_robots_per_study
    print(f"\n=== Q3: Swarm diversity (K studies, {N_TOTAL} total robots fixed, mode={swarm_mode}) ===")
    rng_q3    = np.random.default_rng(seed + 99)
    k_vals, best_taus_k = [], []

    frugal_study_objs = [all_studies[n] for n in available_frugal]
    hifi_rank_common  = [p for p in hifi_study.full_rank if p in common]

    for k in range(1, len(available_frugal) + 1):
        best_tau = -1.0
        best_combo = None
        for combo in combinations(available_frugal, k):
            combo_studies = [all_studies[c] for c in combo]
            weights = _compute_swarm_weights(
                combo_studies, hifi_rank_common, common, swarm_mode, n_bootstrap, seed
            )
            # Hamilton (largest-remainder) allocation: guarantees sum == N_TOTAL
            raw = {c: weights[c] * N_TOTAL for c in combo}
            floors = {c: int(v) for c, v in raw.items()}
            shortfall = N_TOTAL - sum(floors.values())
            order = sorted(combo, key=lambda c: raw[c] - floors[c], reverse=True)
            for c in order[:shortfall]:
                floors[c] += 1
            allocs = floors  # Hamilton guarantees sum == N_TOTAL; 0-alloc studies are skipped by combine_scores_weighted
            taus_c = [
                kendall_tau(
                    combine_scores_weighted(
                        combo_studies, allocs, rng_q3, common
                    ),
                    hifi_rank_common,
                )
                for _ in range(100)
            ]
            mean_tau = float(np.nanmean(taus_c))
            if mean_tau > best_tau:
                best_tau  = mean_tau
                best_combo = combo
        print(
            f"  k={k} studies, {N_TOTAL} total robots: "
            f"tau={best_tau:.3f}  best_combo={best_combo}"
        )
        k_vals.append(k)
        best_taus_k.append(best_tau)

    # -- Q3b: Quantity vs Diversity at same N_total ----------------------------
    print(f"\n=== Q3b: Quantity (1 study) vs Diversity (all studies) at same N_total ===")
    best_frugal = max(
        available_frugal,
        key=lambda n: convergence_results.get(n, {}).get(max_n, {}).get("mean_tau", 0),
    )
    print(f"  Best single frugal study = '{best_frugal}'")
    rng_q3b = np.random.default_rng(seed + 999)
    n_totals = [10, 20, 30, 50, max_n]
    q3b_rows: list[dict] = []
    print(f"  {'N_total':>8}  {'1-study':>12}  {'all-studies':>14}  {'improvement':>12}")

    for n_total in n_totals:
        # Single best frugal study
        tau_single = float(np.nanmean([
            kendall_tau(
                rank_from_sample(all_studies[best_frugal], common, n_total, rng_q3b),
                hifi_rank_common,
            )
            for _ in range(100)
        ]))

        # All frugal studies combined equally — Hamilton allocation, sum == n_total
        # (no max(1,...) so the budget guarantee holds even when n_total < k)
        n_each    = n_total // len(available_frugal)
        remainder = n_total % len(available_frugal)
        allocs    = {
            c: n_each + (1 if i < remainder else 0)
            for i, c in enumerate(available_frugal)
        }
        combo_studies = [all_studies[c] for c in available_frugal]
        tau_diverse = float(np.nanmean([
            kendall_tau(
                combine_scores_weighted(
                    combo_studies, allocs, rng_q3b, common
                ),
                hifi_rank_common,
            )
            for _ in range(100)
        ]))

        improvement = tau_diverse - tau_single
        marker = (
            " <-- diversity wins" if improvement > 0.02 else
            " <-- quantity wins"  if improvement < -0.02 else ""
        )
        print(
            f"  {n_total:>8}  {tau_single:>12.3f}  {tau_diverse:>14.3f}  "
            f"{improvement:>+12.3f}{marker}"
        )
        q3b_rows.append({
            "n_total":     n_total,
            "best_frugal": best_frugal,
            "tau_single":  round(tau_single,  6),
            "tau_diverse": round(tau_diverse, 6),
            "improvement": round(improvement, 6),
            "verdict":     "diversity_wins" if improvement > 0.02
                           else "quantity_wins" if improvement < -0.02
                           else "tied",
        })

    # -- Plots -----------------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, len(convergence_results)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for (name, curve), color in zip(convergence_results.items(), colors):
        ns    = sorted(curve.keys())
        means = [curve[n]["mean_tau"] for n in ns]
        p10   = [curve[n].get("p05_tau", curve[n]["mean_tau"]) for n in ns]
        p90   = [curve[n].get("p95_tau", curve[n]["mean_tau"]) for n in ns]
        lw = 2.5 if name == hifi else 1.5
        ls = "-" if name == hifi else ("--" if name.startswith("study") else ":")
        ax.plot(ns, means, color=color, linewidth=lw, linestyle=ls, label=name, marker=".")
        ax.fill_between(ns, p10, p90, color=color, alpha=0.12)
    ax.axhline(
        CONVERGENCE_THRESHOLD, color="black", linestyle="--", linewidth=1,
        label=f"threshold tau={CONVERGENCE_THRESHOLD}",
    )
    ax.set_xlabel("Number of robot experiments (N)")
    ax.set_ylabel("Kendall tau vs high-fidelity (spectral)")
    ax.set_title("Frugal twin convergence: tau vs N robots")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    ax2 = axes[1]
    ax2.bar(k_vals, best_taus_k, color="steelblue", alpha=0.8)
    ax2.axhline(
        CONVERGENCE_THRESHOLD, color="red", linestyle="--",
        label=f"threshold {CONVERGENCE_THRESHOLD}",
    )
    ax2.set_xlabel("Number of frugal study types combined (K)")
    ax2.set_ylabel("Best achievable tau vs high-fidelity")
    ax2.set_title(
        f"Swarm diversity: K study types, {N_TOTAL} total robots (fixed budget)\n"
        f"mode={swarm_mode}"
    )
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(_OUT / "frugal_twin_convergence.png", dpi=150)
    plt.close()
    print(f"\n[07] Saved frugal_twin_convergence.png")

    # JSON
    out = {
        "hifi_reference":       hifi,
        "hifi_ranking":         hifi_rank_common,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "swarm_mode":            swarm_mode,
        "per_study_convergence": {
            name: {str(n): stats for n, stats in curve.items()}
            for name, curve in convergence_results.items()
        },
        "swarm_diversity": {"k_vals": k_vals, "best_taus": best_taus_k},
        "q3b_quantity_vs_diversity": q3b_rows,
    }
    out_path = _OUT / "frugal_twin_convergence.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[07] Saved {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frugal twin convergence (Exp 07)")
    p.add_argument("--hifi",       default=DEFAULT_HI_FI)
    p.add_argument("--frugal",     nargs="+", default=DEFAULT_FRUGAL,
                   help="Names of frugal studies to include")
    p.add_argument("--swarm-mode", default="equal",
                   choices=["equal", "rho_squared", "inverse_variance"],
                   help="How to weight frugal studies in the combined swarm")
    p.add_argument("--n-robots",   type=int, default=10,
                   dest="n_robots_per_study",
                   help="Robots per study in swarm diversity experiment")
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--score-key",   default="best_color_distance_mean")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--studies",     nargs="+", metavar="NAME=PATH")
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
        frugal_names=args.frugal,
        swarm_mode=args.swarm_mode,
        n_robots_per_study=args.n_robots_per_study,
        n_bootstrap=args.n_bootstrap,
        score_key=args.score_key,
        seed=args.seed,
    )
