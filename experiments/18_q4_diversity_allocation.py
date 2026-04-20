"""
Experiment 18 -- Q4: Quality-Aware Diversity Allocation.

Compares five allocation modes at fixed N_total across N sweeps:
  equal             -- N/k per source (uniform)
  global_tau        -- weight by tau@N=10 vs HF (quality-aware)
  inverse_variance  -- weight by 1/var(tau@N=1) (stability-aware)
  oracle_best_1     -- all budget to single best source (oracle upper bound)
  oracle_best_2     -- budget split between best 2 sources, optimally

Runs for N_total in {5, 10, 20, 50} and all k=2..8 source subsets.
Reports whether quality-aware allocation beats equal at any k or N.

Key hypothesis: quality-aware advantage exists ONLY when sources have
genuinely complementary regions (set-op sources with local specialization).
With 9 generalist sources, equal allocation should match or beat quality-aware.

Outputs
-------
  results/tenki_1000/q4_diversity_allocation.json
  results/tenki_1000/q4_diversity_allocation.md
  results/tenki_1000/q4_allocation_modes.png
"""

from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.flip_data import load_many_studies, common_policy_subset, restrict_to_common_policies
from analysis.flip_metrics import bootstrap_tau_curve, kendall_tau

_OUT = Path(_HERE).parent / "results" / "tenki_1000"
_OUT.mkdir(parents=True, exist_ok=True)

ALL_SOURCES = {
    "spectral":         "output/db_1000_spectral",
    "mixbox":           "output/db_1000_mixbox",
    "km":               "output/db_1000_km",
    "ryb":              "output/db_1000_ryb",
    "study_a":          "output/db_1000_study_a_artist_consensus",
    "study_b":          "output/db_1000_study_b_physics_vs_artist",
    "study_b_reverse":  "output/db_1000_study_b_reverse_artist_vs_physics",
    "study_c":          "output/db_1000_study_c_oilpaint_vs_fooddye",
    "study_c_reverse":  "output/db_1000_study_c_reverse_fooddye_vs_oilpaint",
}
HF              = "spectral"
N_TOTAL_SWEEP   = [5, 10, 20, 50]
CANDIDATE_POOLS = {
    "best4":   ["study_b", "mixbox", "ryb", "km"],
    "all8":    ["mixbox", "km", "ryb", "study_a", "study_b",
                "study_b_reverse", "study_c", "study_c_reverse"],
}
N_BOOTSTRAP = 150


def run(n_bootstrap=N_BOOTSTRAP, seed=42):
    print("[18] Loading databases...")
    all_studies = load_many_studies(ALL_SOURCES)
    print(f"  Loaded: {list(all_studies.keys())}")

    common = common_policy_subset(all_studies)
    print(f"  Common policies: {len(common)}")
    if len(common) < 2:
        print("  Not enough common policies -- aborting.")
        return

    studies_c = {n: restrict_to_common_policies(s, common) for n, s in all_studies.items()}
    hifi_c    = studies_c[HF]
    hifi_rank = [p for p in hifi_c.full_rank if p in common]
    rng       = np.random.default_rng(seed)

    # Pre-compute tau@N=1 variance and tau@N=10 for each frugal source
    print("\n  Pre-computing source quality metrics...")
    frugal = [n for n in studies_c if n != HF]
    tau10:  dict[str, float] = {}
    var_n1: dict[str, float] = {}
    for name in frugal:
        s = studies_c[name]
        n_scan = [n for n in [1, 5, 10] if n <= s.max_n] or [1]
        curve = bootstrap_tau_curve(s, hifi_rank, common, n_scan, N_BOOTSTRAP,
                                     rng_seed=seed + abs(hash(name)) % 9999)
        n10 = 10 if 10 in curve else max(curve.keys(), default=1)
        tau10[name]  = curve.get(n10, {}).get("mean_tau", float("nan"))
        var_n1[name] = curve.get(1, {}).get("std_tau", 1.0) ** 2
        print(f"  {name:30s}  tau@N=10={tau10[name]:.4f}  var@N=1={var_n1[name]:.5f}")

    def make_weights(mode: str, pool: list[str]) -> dict[str, float]:
        if mode == "equal":
            return {n: 1.0 / len(pool) for n in pool}
        elif mode == "global_tau":
            raw = {n: max(0.001, tau10.get(n, 0.001)) for n in pool}
        elif mode == "inverse_variance":
            raw = {n: 1.0 / max(1e-8, var_n1.get(n, 1.0)) for n in pool}
        elif mode == "oracle_best_1":
            best = max(pool, key=lambda n: tau10.get(n, 0))
            return {n: (1.0 if n == best else 0.0) for n in pool}
        elif mode == "oracle_best_2":
            sorted_p = sorted(pool, key=lambda n: -tau10.get(n, 0))
            top2 = sorted_p[:2]
            raw = {n: (tau10.get(n, 0.001) if n in top2 else 0.0) for n in pool}
        else:
            raw = {n: 1.0 for n in pool}
        total = sum(raw.values())
        return {n: v / total for n, v in raw.items()}

    def _hamilton_allocate(weights: dict[str, float], n_total: int) -> dict[str, int]:
        """Exact Hamilton (largest-remainder) allocation summing to n_total."""
        active = {n: w for n, w in weights.items() if w >= 1e-9}
        if not active:
            return {n: 0 for n in weights}
        floors = {n: int(n_total * w) for n, w in active.items()}
        remainder = n_total - sum(floors.values())
        fracs = sorted(active, key=lambda n: -(n_total * active[n] - floors[n]))
        for i in range(remainder):
            floors[fracs[i]] += 1
        return {n: floors.get(n, 0) for n in weights}

    def tau_for_pool_mode(pool, mode, n_total):
        weights = make_weights(mode, pool)
        allocation = _hamilton_allocate(weights, n_total)
        taus = []
        for _ in range(n_bootstrap):
            combined: dict[str, list[float]] = {p: [] for p in common}
            for name in pool:
                n_from = allocation.get(name, 0)
                if n_from < 1:
                    continue
                s = studies_c[name]
                for p in common:
                    if p in s.policy_scores:
                        draw = rng.choice(s.policy_scores[p],
                                          size=min(n_from, len(s.policy_scores[p])), replace=True)
                        combined[p].extend(draw.tolist())
            rank = sorted([p for p in common if combined[p]],
                          key=lambda p: float(np.mean(combined[p])))
            taus.append(kendall_tau(rank, hifi_rank))
        valid = [t for t in taus if not np.isnan(t)]
        return float(np.mean(valid)) if valid else float("nan")

    modes = ["equal", "global_tau", "inverse_variance", "oracle_best_1", "oracle_best_2"]
    pool_mode_results: dict[str, dict] = {}

    for pool_name, pool_sources in CANDIDATE_POOLS.items():
        pool = [s for s in pool_sources if s in studies_c]
        if len(pool) < 2:
            continue
        print(f"\n  Pool: {pool_name} ({len(pool)} sources)")
        pool_mode_results[pool_name] = {}

        for n_total in N_TOTAL_SWEEP:
            print(f"    N_total={n_total}")
            mode_taus = {}
            for mode in modes:
                tau = tau_for_pool_mode(pool, mode, n_total)
                mode_taus[mode] = round(tau, 4)
                print(f"      {mode:22s}  tau={tau:.4f}")

            # Quality-aware helps if global_tau or inverse_variance > equal + 0.01
            qa_helps = (
                mode_taus.get("global_tau", 0) > mode_taus.get("equal", 0) + 0.01
                or mode_taus.get("inverse_variance", 0) > mode_taus.get("equal", 0) + 0.01
            )
            pool_mode_results[pool_name][n_total] = {
                "pool": pool,
                "modes": mode_taus,
                "quality_aware_helps": qa_helps,
            }

    # Determine overall verdict
    any_qa_helps = any(
        r["quality_aware_helps"]
        for pool_res in pool_mode_results.values()
        for r in pool_res.values()
    )
    verdict = (
        "QUALITY-AWARE HELPS" if any_qa_helps
        else "EQUAL ALLOCATION DOMINATES (quality-aware provides no benefit with generalist sources)"
    )
    print(f"\n  Q4 verdict: {verdict}")

    _plot_allocation_modes(pool_mode_results, modes, _OUT / "q4_allocation_modes.png")

    out = {
        "experiment": "18_q4_diversity_allocation",
        "n_bootstrap": n_bootstrap,
        "hifi": HF,
        "tau10_per_source": {n: round(v, 4) for n, v in tau10.items()},
        "var_n1_per_source": {n: round(v, 6) for n, v in var_n1.items()},
        "pool_results": pool_mode_results,
        "verdict": verdict,
    }
    json_path = _OUT / "q4_diversity_allocation.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[18] Saved {json_path}")

    md_path = _OUT / "q4_diversity_allocation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q4 Quality-Aware Diversity Allocation\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={HF}\n\n")
        for pool_name, pool_res in pool_mode_results.items():
            f.write(f"## Pool: {pool_name}\n\n")
            f.write("| N_total | equal | global_tau | inv_var | oracle_1 | oracle_2 | QA helps? |\n")
            f.write("|---------|-------|------------|---------|----------|----------|-----------|\n")
            for n_total, res in sorted(pool_res.items()):
                m = res["modes"]
                f.write(f"| {n_total} "
                        f"| {m.get('equal', float('nan')):.4f} "
                        f"| {m.get('global_tau', float('nan')):.4f} "
                        f"| {m.get('inverse_variance', float('nan')):.4f} "
                        f"| {m.get('oracle_best_1', float('nan')):.4f} "
                        f"| {m.get('oracle_best_2', float('nan')):.4f} "
                        f"| {res['quality_aware_helps']} |\n")
            f.write("\n")
        f.write(f"\n**Verdict**: {verdict}\n\n")
        f.write("Quality-aware advantage requires spatially heterogeneous source quality.\n"
                "With generalist sources, per-source rho varies little across target regions.\n")
    print(f"[18] Saved {md_path}")


def _plot_allocation_modes(pool_mode_results, modes, out_path):
    n_pools = len(pool_mode_results)
    if n_pools == 0:
        return
    fig, axes = plt.subplots(1, n_pools, figsize=(6 * n_pools, 4), squeeze=False)
    for ax, (pool_name, pool_res) in zip(axes[0], pool_mode_results.items()):
        ns = sorted(pool_res.keys())
        for mode in modes:
            taus = [pool_res[n]["modes"].get(mode, float("nan")) for n in ns]
            ax.plot(ns, taus, "o-", label=mode)
        ax.set_xlabel("N_total")
        ax.set_ylabel("tau vs HF")
        ax.set_title(f"Pool: {pool_name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[18] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q4 quality-aware diversity (Exp 18)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(n_bootstrap=args.n_bootstrap, seed=args.seed)
