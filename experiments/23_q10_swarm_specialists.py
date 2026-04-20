"""
Experiment 23 -- Q10: Swarm Advantage with True Local Specialists.

Compares ensemble (equal weights) vs swarm (kNN-local weights) aggregation
across three source pools:
  1. generalists_4  -- spectral (HF), mixbox, km, ryb
  2. all_9          -- all 9 current TENKi-1000 sources
  3. studies_only   -- the 6 set-operation study sources (A, B, B_rev, C, C_rev)

Swarm weights: for each policy, weight_source = 1 / (mean_error of source
over its kNN(k=5) neighbourhood in policy-space).  Sources that are locally
better near a given policy get higher weight.

Success criterion:
  - Primary: swarm tau > ensemble tau + 0.01 at any N
  - Secondary: the highest-weighted source at N=10 is locally better
    than average (local rank vs global rank gap > 1 position)

This tests whether the current 9 generalist sources produce enough spatial
heterogeneity for swarm re-weighting to pay off, or whether equal allocation
already saturates tau.

Outputs
-------
  results/tenki_1000/q10_swarm_specialists.json
  results/tenki_1000/q10_swarm_specialists.md
  results/tenki_1000/q10_swarm_vs_ensemble.png
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
from analysis.flip_metrics import kendall_tau

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

POOLS = {
    # spectral is the HF reference (tau is measured against it) — excluded from all predictor pools
    "generalists_4": ["mixbox", "km", "ryb"],
    "all_8":         ["mixbox", "km", "ryb",
                      "study_a", "study_b", "study_b_reverse",
                      "study_c", "study_c_reverse"],
    "studies_only":  ["study_a", "study_b", "study_b_reverse",
                      "study_c", "study_c_reverse"],
}

N_VALUES    = [1, 2, 5, 10, 20, 50]
N_BOOTSTRAP = 200
KNN_K       = 5


def _build_score_matrix(studies_c, pool, common):
    """Return matrix [n_policies, n_sources] of mean scores and error list."""
    policies = list(common)
    sources  = pool
    score_mat = np.zeros((len(policies), len(sources)))
    for j, src in enumerate(sources):
        s = studies_c.get(src)
        if s is None:
            continue
        for i, p in enumerate(policies):
            vals = s.policy_scores.get(p, [])
            score_mat[i, j] = float(np.mean(vals)) if vals else float("nan")
    return policies, sources, score_mat


def _swarm_weights(score_mat, query_idx, k=KNN_K):
    """
    Local kNN weights for a single query policy.
    Neighbourhood defined by L2 distance in score_mat rows.
    Weight_j = 1 / (mean_error in kNN(query)) where error = score (lower = better
    since scores are color distance, so lower means the source is better at this policy).
    """
    n_policies, n_sources = score_mat.shape
    q = score_mat[query_idx]  # shape (n_sources,)

    # L2 distance between policy vectors (across sources)
    dists = np.linalg.norm(score_mat - q[np.newaxis, :], axis=1)  # (n_policies,)
    dists[query_idx] = np.inf  # exclude self
    nn_idxs = np.argsort(dists)[:k]

    # Local mean score per source in neighbourhood (lower = locally better)
    local_scores = np.nanmean(score_mat[nn_idxs, :], axis=0)  # (n_sources,)
    # Avoid division by zero; add small floor
    local_scores = np.where(np.isnan(local_scores), 1.0, np.maximum(local_scores, 1e-3))
    raw_w = 1.0 / local_scores
    raw_w = np.where(np.isnan(raw_w), 0.0, raw_w)
    total = raw_w.sum()
    return raw_w / total if total > 0 else np.ones(n_sources) / n_sources


def _hamilton_allocate(weights_arr: np.ndarray, n_total: int) -> np.ndarray:
    """Exact Hamilton (largest-remainder) allocation summing to n_total."""
    floors = np.floor(n_total * weights_arr).astype(int)
    remainder = n_total - floors.sum()
    fracs = n_total * weights_arr - floors
    top_idx = np.argsort(-fracs)[:remainder]
    floors[top_idx] += 1
    return floors


def _tau_ensemble(studies_c, pool, common, hifi_rank, n_total, rng):
    """Equal-weight ensemble: exact Hamilton allocation across pool, rank."""
    k = len(pool)
    equal_w = np.ones(k) / k
    alloc = _hamilton_allocate(equal_w, n_total)
    combined = {p: [] for p in common}
    for j, src in enumerate(pool):
        n_from = int(alloc[j])
        if n_from < 1:
            continue
        s = studies_c.get(src)
        if s is None:
            continue
        for p in common:
            vals = s.policy_scores.get(p, [])
            if vals:
                draw = rng.choice(vals, size=min(n_from, len(vals)), replace=True)
                combined[p].extend(draw.tolist())
    rank = sorted([p for p in common if combined[p]],
                  key=lambda p: float(np.mean(combined[p])))
    return kendall_tau(rank, hifi_rank)


def _tau_swarm(studies_c, pool, common, hifi_rank, score_mat, n_total, rng):
    """
    Swarm: for each policy, compute local kNN weights, draw via exact Hamilton
    allocation summing to exactly n_total across sources.
    """
    policies = list(common)

    # Per-policy swarm draws
    combined = {p: [] for p in common}
    for i, p in enumerate(policies):
        w = _swarm_weights(score_mat, i)  # sums to 1
        alloc = _hamilton_allocate(w, n_total)
        for j, src in enumerate(pool):
            n_from = int(alloc[j])
            if n_from < 1:
                continue
            s = studies_c.get(src)
            if s is None:
                continue
            vals = s.policy_scores.get(p, [])
            if vals:
                draw = rng.choice(vals, size=min(n_from, len(vals)), replace=True)
                combined[p].extend(draw.tolist())

    rank = sorted([p for p in common if combined[p]],
                  key=lambda p: float(np.mean(combined[p])))
    return kendall_tau(rank, hifi_rank)


def _local_vs_global_heterogeneity(score_mat, policies, sources):
    """
    Quantify spatial heterogeneity: for each policy, find the locally best source
    (lowest local mean score in kNN) and compare to its global rank.
    Returns mean rank gap = mean |global_rank(local_best) - 1|.
    If all sources are equally good globally, rank gap ~ 0.
    If local specialists exist, rank gap > 1.
    """
    global_ranks = np.argsort(np.nanmean(score_mat, axis=0))  # ascending = lower score first
    rank_gaps = []
    for i in range(len(policies)):
        w = _swarm_weights(score_mat, i)
        local_best_j = int(np.argmax(w))  # highest weight = locally best
        global_rank_of_local_best = int(np.where(global_ranks == local_best_j)[0][0])
        rank_gaps.append(global_rank_of_local_best)
    return float(np.mean(rank_gaps)), float(np.std(rank_gaps))


def run(n_values=None, n_bootstrap=N_BOOTSTRAP, seed=42):
    n_vals = sorted(set(n_values or N_VALUES))

    print("[23] Loading databases...")
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

    pool_results = {}

    for pool_name, pool_sources in POOLS.items():
        pool = [s for s in pool_sources if s in studies_c]
        if len(pool) < 2:
            print(f"  SKIP pool {pool_name}: only {len(pool)} sources available")
            continue
        print(f"\n  Pool: {pool_name}  ({len(pool)} sources: {pool})")

        # Build score matrix for swarm weights
        policies, sources, score_mat = _build_score_matrix(studies_c, pool, common)

        # Spatial heterogeneity diagnostic
        rk_mean, rk_std = _local_vs_global_heterogeneity(score_mat, policies, sources)
        print(f"    Local specialist rank gap: mean={rk_mean:.2f}  std={rk_std:.2f}")

        n_results = {}
        for n_total in n_vals:
            tau_ens_list, tau_swarm_list = [], []
            for _ in range(n_bootstrap):
                t_ens = _tau_ensemble(studies_c, pool, common, hifi_rank, n_total, rng)
                t_sw  = _tau_swarm(studies_c, pool, common, hifi_rank,
                                   score_mat, n_total, rng)
                if not np.isnan(t_ens):
                    tau_ens_list.append(t_ens)
                if not np.isnan(t_sw):
                    tau_swarm_list.append(t_sw)

            tau_ens   = float(np.mean(tau_ens_list))   if tau_ens_list   else float("nan")
            tau_sw    = float(np.mean(tau_swarm_list)) if tau_swarm_list else float("nan")
            delta     = round(tau_sw - tau_ens, 4) if not (np.isnan(tau_ens) or np.isnan(tau_sw)) else float("nan")
            sw_helps  = delta > 0.01 if not np.isnan(delta) else False

            n_results[n_total] = {
                "tau_ensemble": round(tau_ens, 4),
                "tau_swarm":    round(tau_sw,  4),
                "delta":        delta,
                "swarm_helps":  sw_helps,
            }
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "nan"
            print(f"    N={n_total:3d}  ensemble={tau_ens:.4f}  swarm={tau_sw:.4f}  delta={delta_str}")

        any_helps = any(r["swarm_helps"] for r in n_results.values())
        verdict = "SWARM_WINS" if any_helps else "ENSEMBLE_DOMINATES"
        print(f"    => {verdict}  (local_rank_gap={rk_mean:.2f})")

        pool_results[pool_name] = {
            "pool_sources":     pool,
            "local_rank_gap":   round(rk_mean, 3),
            "local_rank_gap_std": round(rk_std, 3),
            "n_results":        n_results,
            "any_swarm_helps":  any_helps,
            "verdict":          verdict,
        }

    _plot_swarm_vs_ensemble(pool_results, n_vals, _OUT / "q10_swarm_vs_ensemble.png")

    overall_verdict = (
        "SWARM_WINS_IN_SOME_POOL"
        if any(v["any_swarm_helps"] for v in pool_results.values())
        else "EQUAL_ALLOCATION_SUFFICIENT"
    )
    print(f"\n  Overall verdict: {overall_verdict}")

    out = {
        "experiment": "23_q10_swarm_specialists",
        "n_bootstrap": n_bootstrap,
        "hifi": HF,
        "knn_k": KNN_K,
        "pool_results": pool_results,
        "overall_verdict": overall_verdict,
    }
    json_path = _OUT / "q10_swarm_specialists.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[23] Saved {json_path}")

    md_path = _OUT / "q10_swarm_specialists.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q10 Swarm Advantage with True Local Specialists\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={HF}  kNN k={KNN_K}\n\n")
        f.write(f"**Overall verdict**: {overall_verdict}\n\n")
        f.write("Swarm criterion: swarm tau > ensemble tau + 0.01 at any N.\n\n")
        f.write("## Pool summary\n\n")
        f.write("| Pool | local rank gap | verdict | N=10 ensemble | N=10 swarm | delta |\n")
        f.write("|------|---------------|---------|---------------|------------|-------|\n")
        for pname, res in pool_results.items():
            r10 = res["n_results"].get(10, {})
            _d = r10.get('delta', float('nan'))
            _d_str = f"{_d:.4f}" if isinstance(_d, float) and not np.isnan(_d) else "nan"
            f.write(f"| {pname} "
                    f"| {res['local_rank_gap']:.2f} "
                    f"| {res['verdict']} "
                    f"| {r10.get('tau_ensemble', float('nan')):.4f} "
                    f"| {r10.get('tau_swarm', float('nan')):.4f} "
                    f"| {_d_str} |\n")
        f.write("\n## Per-pool detail\n\n")
        for pname, res in pool_results.items():
            f.write(f"### Pool: {pname}\n\n")
            f.write(f"Sources: {', '.join(res['pool_sources'])}\n\n")
            f.write(f"Local specialist rank gap: {res['local_rank_gap']:.2f} +/- "
                    f"{res['local_rank_gap_std']:.2f} (higher = more spatial heterogeneity)\n\n")
            f.write("| N | ensemble tau | swarm tau | delta | swarm helps? |\n")
            f.write("|---|-------------|-----------|-------|-------------|\n")
            for n_total, r in sorted(res["n_results"].items()):
                delta_str = f"{r['delta']:+.4f}" if isinstance(r["delta"], float) and not np.isnan(r["delta"]) else "nan"
                f.write(f"| {n_total} | {r['tau_ensemble']:.4f} | {r['tau_swarm']:.4f} "
                        f"| {delta_str} | {r['swarm_helps']} |\n")
            f.write(f"\n**Verdict**: {res['verdict']}\n\n")
        f.write("## Interpretation\n\n")
        f.write("A large local rank gap (>> 1) means local specialists exist and swarm "
                "should win. A gap near 0 means sources are globally homogeneous and "
                "equal allocation is sufficient.\n\n")
        f.write("If `EQUAL_ALLOCATION_SUFFICIENT` for all pools, it confirms that the "
                "9 current TENKi-1000 sources are generalists with no meaningful spatial "
                "specialization, and swarm adds overhead without benefit.\n")
    print(f"[23] Saved {md_path}")


def _plot_swarm_vs_ensemble(pool_results, n_vals, out_path):
    n_pools = len(pool_results)
    if n_pools == 0:
        return
    fig, axes = plt.subplots(1, n_pools, figsize=(5 * n_pools, 4), squeeze=False)
    for ax, (pool_name, res) in zip(axes[0], pool_results.items()):
        ns     = sorted(res["n_results"].keys())
        t_ens  = [res["n_results"][n]["tau_ensemble"] for n in ns]
        t_sw   = [res["n_results"][n]["tau_swarm"]    for n in ns]
        ax.plot(ns, t_ens, "o-",  color="royalblue",  label="ensemble (equal)", linewidth=2)
        ax.plot(ns, t_sw,  "s--", color="darkorange",  label="swarm (kNN)",     linewidth=2)
        ax.fill_between(ns, t_ens, t_sw, alpha=0.15,
                        color="green" if res["any_swarm_helps"] else "red")
        ax.set_xlabel("N robots")
        ax.set_ylabel("tau vs HF")
        ax.set_title(f"Pool: {pool_name}\n({res['verdict']})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        # Annotate rank gap
        ax.text(0.05, 0.05,
                f"rank gap={res['local_rank_gap']:.2f}",
                transform=ax.transAxes, fontsize=8, color="navy")
    plt.suptitle("Q10: Swarm vs Ensemble across source pools", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[23] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q10 swarm vs ensemble specialists (Exp 23)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(n_bootstrap=args.n_bootstrap, seed=args.seed)
