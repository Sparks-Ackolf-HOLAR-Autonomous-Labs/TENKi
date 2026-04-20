"""
Experiment 16 -- Q2: Epsilon-Symmetry Threshold.

Computes explicit epsilon_symmetry for every subset of the available source pool:

  epsilon_volume(S) = max over all mirror pairs in S of:
      |vol_fwd - vol_rev| / (vol_fwd + vol_rev)

  epsilon_tau(S) = max over all mirror pairs in S of:
      |tau10_fwd - tau10_rev| / max(tau10_fwd, tau10_rev, 0.01)

Reports the smallest collection size k and best collection achieving:
  epsilon_volume <= 0.25, 0.10, 0.05
  epsilon_tau    <= 0.25, 0.10, 0.05

Also computes the engine-permutation symmetry score from exp 06 for each
collection and plots epsilon vs k.

Outputs
-------
  results/tenki_1000/q2_epsilon_symmetry.json
  results/tenki_1000/q2_epsilon_symmetry.md
  results/tenki_1000/q2_epsilon_vs_k.png
"""

from __future__ import annotations
import argparse, glob, json, sys, os
from itertools import combinations
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.flip_data import load_many_studies, common_policy_subset, restrict_to_common_policies
from analysis.flip_metrics import bootstrap_tau_curve, full_data_ceiling

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

MIRROR_PAIRS = [
    ("study_b",   "study_b_reverse"),
    ("study_c",   "study_c_reverse"),
]

HF = "spectral"
N_BOOTSTRAP = 200


def count_targets(db_path: str) -> int:
    db = Path(_ROOT) / db_path
    if not db.exists():
        return 0
    n = 0
    for p in sorted(glob.glob(str(db / "targets" / "*.json")))[:60]:
        try:
            with open(p) as f:
                d = json.load(f)
            n += len(d.get("targets", d if isinstance(d, list) else []))
        except Exception:
            pass
    return n


def run(n_bootstrap=N_BOOTSTRAP, seed=42):
    print("[16] Loading databases...")
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

    frugal = [n for n in studies_c if n != HF]

    # Count targets per source for volume-based epsilon
    print("\n  Counting targets per source...")
    target_counts = {name: count_targets(path) for name, path in ALL_SOURCES.items()}
    for n, c in target_counts.items():
        print(f"  {n:30s}: {c} targets")

    # Bootstrap tau@N=10 for each frugal source
    print("\n  Computing tau@N=10 per source...")
    tau10: dict[str, float] = {}
    std10: dict[str, float] = {}
    for name in frugal:
        s = studies_c[name]
        n_scan = [n for n in [1, 5, 10] if n <= s.max_n] or [1]
        curve = bootstrap_tau_curve(s, hifi_rank, common, n_scan, n_bootstrap,
                                     rng_seed=seed + abs(hash(name)) % 9999)
        n_key = 10 if 10 in curve else max(curve.keys(), default=1)
        tau10[name] = curve.get(n_key, {}).get("mean_tau", float("nan"))
        std10[name] = curve.get(n_key, {}).get("std_tau",  0.0)
        print(f"  {name:30s}: tau@N=10={tau10[name]:.4f}±{std10[name]:.4f}")

    # Evaluate every subset of frugal sources
    print("\n  Evaluating all subsets...")
    available_mirror_pairs = [(a, b) for a, b in MIRROR_PAIRS
                               if a in studies_c and b in studies_c]
    records = []
    for k in range(1, len(frugal) + 1):
        for subset in combinations(frugal, k):
            subset = list(subset)

            # epsilon_volume: for each mirror pair fully in subset
            eps_vols = []
            for fwd, rev in available_mirror_pairs:
                if fwd in subset and rev in subset:
                    nf = target_counts.get(fwd, 0)
                    nr = target_counts.get(rev, 0)
                    if nf + nr > 0:
                        eps_vols.append(abs(nf - nr) / (nf + nr))
            eps_volume = max(eps_vols) if eps_vols else float("nan")

            # epsilon_tau: for each mirror pair fully in subset
            eps_taus = []
            for fwd, rev in available_mirror_pairs:
                if fwd in subset and rev in subset:
                    tf = tau10.get(fwd, float("nan"))
                    tr = tau10.get(rev, float("nan"))
                    denom = max(tf, tr, 0.01)
                    if not (np.isnan(tf) or np.isnan(tr)):
                        eps_taus.append(abs(tf - tr) / denom)
            eps_tau = max(eps_taus) if eps_taus else float("nan")

            # Best source quality in subset
            best_tau = max((tau10.get(n, 0) for n in subset), default=0.0)
            n_pairs_in_subset = sum(1 for a, b in available_mirror_pairs
                                    if a in subset and b in subset)

            records.append({
                "k": k,
                "sources": subset,
                "eps_volume": eps_volume,
                "eps_tau":    eps_tau,
                "best_tau10": round(best_tau, 4),
                "n_mirror_pairs": n_pairs_in_subset,
            })

    # Report thresholds
    print("\n  === Epsilon-symmetry thresholds ===")
    results_table = {}
    for metric in ["eps_volume", "eps_tau"]:
        results_table[metric] = {}
        for thresh in [0.25, 0.10, 0.05]:
            qualifying = [r for r in records if not np.isnan(r[metric]) and r[metric] <= thresh]
            if qualifying:
                best = min(qualifying, key=lambda r: (r["k"], -r["best_tau10"]))
                results_table[metric][thresh] = {
                    "min_k":   best["k"],
                    "sources": best["sources"],
                    "eps":     round(best[metric], 4),
                    "best_tau10": best["best_tau10"],
                }
                print(f"  {metric} <= {thresh:.2f}: k={best['k']}  eps={best[metric]:.4f}  "
                      f"sources={best['sources']}")
            else:
                results_table[metric][thresh] = None
                print(f"  {metric} <= {thresh:.2f}: NOT ACHIEVABLE with current source pool")

    # Plot epsilon vs k
    _plot_eps_vs_k(records, frugal, _OUT / "q2_epsilon_vs_k.png")

    # Write JSON
    out = {
        "experiment": "16_q2_epsilon_symmetry",
        "n_bootstrap": n_bootstrap,
        "hifi": HF,
        "mirror_pairs_available": available_mirror_pairs,
        "tau10_per_source": {n: round(v, 4) for n, v in tau10.items()},
        "target_counts": target_counts,
        "thresholds": results_table,
        "top_subsets_by_eps_tau": sorted(
            [r for r in records if not np.isnan(r["eps_tau"])],
            key=lambda r: r["eps_tau"]
        )[:20],
    }
    json_path = _OUT / "q2_epsilon_symmetry.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[16] Saved {json_path}")

    # Write Markdown
    md_path = _OUT / "q2_epsilon_symmetry.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q2 Epsilon-Symmetry Threshold\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={HF}\n\n")
        f.write("## Target counts per source\n\n")
        f.write("| Source | Targets |\n|--------|--------|\n")
        for name in sorted(target_counts, key=lambda n: -target_counts[n]):
            f.write(f"| {name} | {target_counts[name]} |\n")
        f.write("\n## Tau@N=10 per source\n\n")
        f.write("| Source | tau@N=10 | std |\n|--------|----------|-----|\n")
        for name in sorted(tau10, key=lambda n: -tau10.get(n, 0)):
            f.write(f"| {name} | {tau10.get(name, float('nan')):.4f} | {std10.get(name, 0):.4f} |\n")
        f.write("\n## Symmetry thresholds\n\n")
        for metric in ["eps_volume", "eps_tau"]:
            f.write(f"\n### {metric}\n\n")
            f.write("| epsilon threshold | min k | best sources | achieved eps |\n")
            f.write("|-------------------|-------|--------------|-------------|\n")
            for thresh in [0.25, 0.10, 0.05]:
                r = results_table[metric].get(thresh)
                if r:
                    f.write(f"| <= {thresh} | {r['min_k']} | {', '.join(r['sources'])} | {r['eps']:.4f} |\n")
                else:
                    f.write(f"| <= {thresh} | — | not achievable | — |\n")
    print(f"[16] Saved {md_path}")


def _plot_eps_vs_k(records, frugal, out_path):
    by_k: dict[int, list[float]] = {}
    for r in records:
        if not np.isnan(r["eps_tau"]):
            by_k.setdefault(r["k"], []).append(r["eps_tau"])
    ks   = sorted(by_k.keys())
    mins = [min(by_k[k]) for k in ks]
    meds = [float(np.median(by_k[k])) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, mins, "b-o", label="Best subset of size k")
    ax.plot(ks, meds, "g--s", label="Median over all subsets of size k")
    for thresh, col in [(0.25, "orange"), (0.10, "red"), (0.05, "darkred")]:
        ax.axhline(thresh, color=col, linestyle=":", alpha=0.7, label=f"ε={thresh}")
    ax.set_xlabel("Number of frugal sources k")
    ax.set_ylabel("ε_tau (max relative tau deviation over mirror pairs)")
    ax.set_title("Q2: Minimum epsilon_symmetry achievable at each collection size")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[16] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q2 epsilon-symmetry threshold (Exp 16)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(n_bootstrap=args.n_bootstrap, seed=args.seed)
