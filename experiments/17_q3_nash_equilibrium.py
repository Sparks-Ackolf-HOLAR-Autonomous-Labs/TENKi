"""
Experiment 17 -- Q3: Nash Equilibrium Under Fixed Budget.

For every non-empty subset S of frugal sources, with equal allocation
(N_total / |S| experiments per source), computes tau vs HF ranking.
Also tests rho-proportional allocation for comparison.

Questions answered:
  1. Does the optimum stay concentrated on 1-2 sources or spread?
  2. Does the current 9-source ordering change under full budget?
  3. Is concentration a property of source quality or of budget?

Budget sweep: N_total in {5, 10, 20, 50}.

Outputs
-------
  results/tenki_1000/q3_nash_equilibrium.json
  results/tenki_1000/q3_nash_equilibrium.md
  results/tenki_1000/q3_nash_best_vs_k.png
"""

from __future__ import annotations
import argparse, json, sys, os
from itertools import combinations
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.flip_data import load_many_studies, common_policy_subset, restrict_to_common_policies
from analysis.flip_metrics import bootstrap_tau_curve, kendall_tau
from analysis.flip_data import StudyScores

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
HF = "spectral"
N_BUDGET_SWEEP = [5, 10, 20, 50]
N_BOOTSTRAP    = 200


def _tau_at_n_equal(
    sources: list[str],
    studies: dict[str, StudyScores],
    common: list[str],
    hifi_rank: list[str],
    n_total: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    n_per = max(1, n_total // len(sources))
    taus = []
    for _ in range(n_bootstrap):
        combined: dict[str, list[float]] = {p: [] for p in common}
        for name in sources:
            s = studies[name]
            for p in common:
                if p in s.policy_scores:
                    draw = rng.choice(s.policy_scores[p],
                                      size=min(n_per, len(s.policy_scores[p])), replace=True)
                    combined[p].extend(draw.tolist())
        rank = sorted([p for p in common if combined[p]],
                      key=lambda p: float(np.mean(combined[p])))
        taus.append(kendall_tau(rank, hifi_rank))
    valid = [t for t in taus if not np.isnan(t)]
    return float(np.mean(valid)) if valid else float("nan")


def _tau_at_n_rho_weighted(
    sources: list[str],
    studies: dict[str, StudyScores],
    common: list[str],
    hifi_rank: list[str],
    full_rank_per_source: dict[str, list[str]],
    n_total: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    from scipy.stats import kendalltau as _kt
    # Weight by tau of full-data ranking vs HF
    weights_raw = {}
    for name in sources:
        frd = full_rank_per_source.get(name, [])
        tau_full, _ = _kt(
            [frd.index(p) for p in common if p in frd],
            [hifi_rank.index(p) for p in common if p in hifi_rank],
        )
        weights_raw[name] = max(0.01, float(tau_full))
    total = sum(weights_raw.values())
    weights = {n: w / total for n, w in weights_raw.items()}

    taus = []
    for _ in range(n_bootstrap):
        combined: dict[str, list[float]] = {p: [] for p in common}
        for name in sources:
            n_from = max(1, int(round(n_total * weights[name])))
            s = studies[name]
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


def run(n_budget_sweep=None, n_bootstrap=N_BOOTSTRAP, seed=42):
    n_budgets = sorted(set(n_budget_sweep or N_BUDGET_SWEEP))

    print("[17] Loading databases...")
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
    frugal    = [n for n in studies_c if n != HF]
    full_ranks = {n: [p for p in studies_c[n].full_rank if p in common] for n in frugal}

    rng = np.random.default_rng(seed)

    all_results: dict[int, list[dict]] = {}
    for n_total in n_budgets:
        print(f"\n  === Budget N_total={n_total} ===")
        records = []
        for k in range(1, len(frugal) + 1):
            for subset in combinations(frugal, k):
                subset_list = list(subset)
                tau_eq  = _tau_at_n_equal(subset_list, studies_c, common, hifi_rank,
                                           n_total, n_bootstrap, rng)
                tau_rho = _tau_at_n_rho_weighted(subset_list, studies_c, common, hifi_rank,
                                                  full_ranks, n_total, n_bootstrap, rng)
                records.append({
                    "k": k, "sources": subset_list,
                    "tau_equal": round(tau_eq,  4),
                    "tau_rho":   round(tau_rho, 4),
                })
        records.sort(key=lambda r: -r["tau_equal"])
        all_results[n_total] = records

        best_eq  = records[0]
        best_rho = max(records, key=lambda r: r["tau_rho"])
        best_k1  = max((r for r in records if r["k"] == 1), key=lambda r: r["tau_equal"])
        conc_eq  = best_k1["tau_equal"] >= best_eq["tau_equal"] - 0.01

        print(f"  Best equal    : k={best_eq['k']}  tau={best_eq['tau_equal']:.4f}  {best_eq['sources']}")
        print(f"  Best rho-prop : k={best_rho['k']} tau={best_rho['tau_rho']:.4f}  {best_rho['sources']}")
        print(f"  Best single   : k=1  tau={best_k1['tau_equal']:.4f}  {best_k1['sources']}")
        print(f"  Concentration holds (equal): {conc_eq}")

    # Nash concentration summary across budgets
    nash_summary = {}
    for n_total, records in all_results.items():
        best_eq = records[0]
        best_k1 = max((r for r in records if r["k"] == 1), key=lambda r: r["tau_equal"])
        delta = best_eq["tau_equal"] - best_k1["tau_equal"]
        nash_summary[n_total] = {
            "best_sources":   best_eq["sources"],
            "best_k":         best_eq["k"],
            "best_tau_equal": best_eq["tau_equal"],
            "best_k1_tau":    best_k1["tau_equal"],
            "best_k1_source": best_k1["sources"][0],
            "k1_to_best_delta": round(delta, 4),
            "concentration_holds": delta <= 0.01,
        }

    _plot_best_vs_k(all_results, n_budgets, _OUT / "q3_nash_best_vs_k.png")

    out = {
        "experiment": "17_q3_nash_equilibrium",
        "n_bootstrap": n_bootstrap,
        "hifi": HF,
        "nash_summary": nash_summary,
        "top10_per_budget": {
            str(n): all_results[n][:10] for n in n_budgets
        },
    }
    json_path = _OUT / "q3_nash_equilibrium.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[17] Saved {json_path}")

    md_path = _OUT / "q3_nash_equilibrium.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q3 Nash Equilibrium Under Fixed Budget\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={HF}\n\n")
        f.write("Allocation modes: equal (N/k per source), rho-proportional (tau-weighted).\n\n")
        f.write("## Nash concentration summary\n\n")
        f.write("| N_total | best k | best sources | best tau | k=1 tau | delta | concentrated? |\n")
        f.write("|---------|--------|--------------|----------|---------|-------|---------------|\n")
        for n_total, s in nash_summary.items():
            srcs = ", ".join(s["best_sources"][:3]) + ("..." if len(s["best_sources"]) > 3 else "")
            f.write(f"| {n_total} | {s['best_k']} | {srcs} "
                    f"| {s['best_tau_equal']:.4f} | {s['best_k1_tau']:.4f} "
                    f"| {s['k1_to_best_delta']:+.4f} | {s['concentration_holds']} |\n")
        f.write("\n**Concentration holds** when the best single-source tau is within 0.01 of the "
                "best multi-source tau (adding sources hurts or is neutral).\n\n")
        f.write("## Top-5 collections at each budget\n\n")
        for n_total in n_budgets:
            f.write(f"### N_total={n_total}\n\n")
            f.write("| k | sources | tau_equal | tau_rho |\n|---|---------|-----------|--------|\n")
            for r in all_results[n_total][:5]:
                f.write(f"| {r['k']} | {', '.join(r['sources'])} "
                        f"| {r['tau_equal']:.4f} | {r['tau_rho']:.4f} |\n")
            f.write("\n")
    print(f"[17] Saved {md_path}")


def _plot_best_vs_k(all_results, n_budgets, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_budgets)))
    for i, n_total in enumerate(n_budgets):
        records = all_results[n_total]
        by_k: dict[int, list[float]] = {}
        for r in records:
            by_k.setdefault(r["k"], []).append(r["tau_equal"])
        ks   = sorted(by_k.keys())
        best = [max(by_k[k]) for k in ks]
        ax.plot(ks, best, "o-", color=colors[i], label=f"N_total={n_total}")
    ax.set_xlabel("Number of sources k (equal allocation)")
    ax.set_ylabel("Best tau@N vs HF")
    ax.set_title("Q3: Does the Nash equilibrium concentrate or spread?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[17] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q3 Nash equilibrium (Exp 17)")
    p.add_argument("--n-budget-sweep", nargs="+", type=int, default=N_BUDGET_SWEEP)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(n_budget_sweep=args.n_budget_sweep, n_bootstrap=args.n_bootstrap, seed=args.seed)
