"""
Experiment 19 -- Q5: Per-Robot Information Weighting by Target Difficulty.

Difficulty = minimum achievable HF score per experiment (lower = easier).
Splits the experiment pool into thirds: easy / medium / hard.

For each frugal source, computes tau@N curves separately per difficulty bin,
then compares them to the full-pool curve.

Key questions:
  1. Do hard targets carry more or less transferable information per robot?
  2. Can a "receiver" source (low overall tau) be a donor on easy targets?
  3. Does difficulty-weighted sampling improve tau vs uniform sampling?

Also reports: what fraction of the transfer signal comes from each bin.

Outputs
-------
  results/tenki_1000/q5_robot_difficulty.json
  results/tenki_1000/q5_robot_difficulty.md
  results/tenki_1000/q5_difficulty_curves.png
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

from analysis.flip_data import load_many_studies, common_policy_subset, restrict_to_common_policies, StudyScores
from analysis.flip_metrics import bootstrap_tau_curve, full_data_ceiling, kendall_tau

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
HF          = "spectral"
N_VALUES    = [1, 2, 3, 5, 8, 10, 20, 30]
N_BOOTSTRAP = 200
FOCUS_SOURCES = ["study_b", "mixbox", "ryb", "km", "study_a", "study_c"]


def _make_bin_study(name, src_scores, hf_scores, exp_idxs, common):
    """Build a StudyScores restricted to experiments in exp_idxs."""
    bin_lf = {
        p: [src_scores[p][i] for i in exp_idxs if i < len(src_scores.get(p, []))]
        for p in common if p in src_scores
    }
    bin_lf = {p: v for p, v in bin_lf.items() if len(v) >= 2}
    if not bin_lf:
        return None, None

    bin_hf_rank = sorted(
        [p for p in common if p in hf_scores and any(i < len(hf_scores[p]) for i in exp_idxs)],
        key=lambda p: float(np.mean([hf_scores[p][i] for i in exp_idxs
                                     if i < len(hf_scores[p])])),
    )
    study = StudyScores(
        name=f"{name}_bin",
        db_path="",
        policy_scores=bin_lf,
        full_rank=sorted(bin_lf, key=lambda p: float(np.mean(bin_lf[p]))),
        n_policies=len(bin_lf),
        max_n=min(len(v) for v in bin_lf.values()),
    )
    return study, bin_hf_rank


def run(n_values=None, n_bootstrap=N_BOOTSTRAP, seed=42):
    n_vals = sorted(set(n_values or N_VALUES))

    print("[19] Loading databases...")
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
    n_exps    = hifi_c.max_n
    hf_scores = hifi_c.policy_scores

    # Compute per-experiment difficulty: min HF score across policies
    per_exp_diff = []
    for i in range(n_exps):
        scores = [hf_scores[p][i] for p in common if i < len(hf_scores.get(p, []))]
        per_exp_diff.append(float(np.min(scores)) if scores else float("nan"))

    diff_arr = np.array([d for d in per_exp_diff if not np.isnan(d)])
    q33, q67 = np.quantile(diff_arr, [0.33, 0.67])
    bins = {
        "easy":   [i for i, d in enumerate(per_exp_diff) if not np.isnan(d) and d <= q33],
        "medium": [i for i, d in enumerate(per_exp_diff) if not np.isnan(d) and q33 < d <= q67],
        "hard":   [i for i, d in enumerate(per_exp_diff) if not np.isnan(d) and d > q67],
    }
    print(f"\n  Difficulty bins (q33={q33:.2f}, q67={q67:.2f}):")
    for bname, idxs in bins.items():
        mean_d = np.mean([per_exp_diff[i] for i in idxs])
        print(f"    {bname:6s}: {len(idxs)} exps  mean_difficulty={mean_d:.2f}")

    # For each source, compute tau@N curves per bin + full pool
    focus = [s for s in FOCUS_SOURCES if s in studies_c]
    all_source_results = {}

    for src_name in focus:
        src = studies_c[src_name]
        src_scores = src.policy_scores
        print(f"\n  {src_name}  (max_n={src.max_n})")

        # Full-pool tau@N curve
        n_scan = [n for n in n_vals if n <= src.max_n] or [1]
        curve_full = bootstrap_tau_curve(src, hifi_rank, common, n_scan, n_bootstrap,
                                          rng_seed=seed + abs(hash(src_name)) % 9999)
        full_tau = {n: curve_full.get(n, {}).get("mean_tau", float("nan")) for n in n_scan}
        ceiling  = full_data_ceiling(src, hifi_c, common)

        bin_results = {}
        for bin_name, exp_idxs in bins.items():
            if len(exp_idxs) < 5:
                continue
            bin_study, bin_hf_rank = _make_bin_study(
                src_name, src_scores, hf_scores, exp_idxs, common
            )
            if bin_study is None or bin_study.max_n < 1:
                continue

            n_scan_bin = [n for n in n_vals if n <= bin_study.max_n] or [1]
            curve_bin = bootstrap_tau_curve(
                bin_study, bin_hf_rank, list(bin_study.policy_scores.keys()),
                n_scan_bin, n_bootstrap,
                rng_seed=seed + abs(hash(src_name + bin_name)) % 9999,
            )
            bin_tau = {n: curve_bin.get(n, {}).get("mean_tau", float("nan")) for n in n_scan_bin}
            bin_results[bin_name] = {
                "n_exps":       len(exp_idxs),
                "mean_diff":    round(float(np.mean([per_exp_diff[i] for i in exp_idxs])), 3),
                "tau_per_n":    {str(n): round(v, 4) for n, v in bin_tau.items()},
                "tau@N=1":      round(bin_tau.get(min(n_scan_bin), float("nan")), 4),
            }
            tau1 = bin_results[bin_name]["tau@N=1"]
            full_tau1 = full_tau.get(1, float("nan"))
            print(f"    {bin_name:6s}  tau@N=1={tau1:.3f}  (full pool tau@N=1={full_tau1:.3f})")

        # Information contribution: does tau on easy exps predict tau on all?
        easy_tau1 = bin_results.get("easy",   {}).get("tau@N=1", float("nan"))
        hard_tau1 = bin_results.get("hard",   {}).get("tau@N=1", float("nan"))
        full_tau1_val = full_tau.get(1, float("nan"))

        if not (np.isnan(easy_tau1) or np.isnan(hard_tau1)):
            if hard_tau1 > easy_tau1 + 0.05:
                info_pattern = "HARD TARGETS MORE INFORMATIVE"
            elif easy_tau1 > hard_tau1 + 0.05:
                info_pattern = "EASY TARGETS MORE INFORMATIVE"
            else:
                info_pattern = "UNIFORM ACROSS DIFFICULTY"
        else:
            info_pattern = "INSUFFICIENT DATA"

        all_source_results[src_name] = {
            "ceiling":       round(ceiling, 4),
            "tau@N=1_full":  round(full_tau1_val, 4),
            "full_tau_per_n": {str(n): round(v, 4) for n, v in full_tau.items()},
            "bins":           bin_results,
            "info_pattern":   info_pattern,
        }
        print(f"    => {info_pattern}")

    # Difficulty-weighted sampling: does weighting experiments by (1/difficulty) improve tau?
    print("\n  === Difficulty-weighted sampling test ===")
    rng = np.random.default_rng(seed + 1)
    weighted_taus = {}
    for src_name in focus[:4]:  # limit for speed
        if src_name not in studies_c:
            continue
        src = studies_c[src_name]
        n_total = 10
        # Inverse-difficulty weights (hard targets downweighted)
        weights = np.array([1.0 / max(0.1, per_exp_diff[i]) if i < len(per_exp_diff) else 0
                            for i in range(n_exps)])
        weights /= weights.sum()
        taus_weighted, taus_uniform = [], []
        for _ in range(n_bootstrap):
            # Weighted draw
            idxs_w = rng.choice(n_exps, size=min(n_total, n_exps), replace=True, p=weights)
            # Uniform draw
            idxs_u = rng.choice(n_exps, size=min(n_total, n_exps), replace=True)
            for taus_list, idxs in [(taus_weighted, idxs_w), (taus_uniform, idxs_u)]:
                scores: dict[str, list[float]] = {p: [] for p in common}
                for i in idxs:
                    for p in common:
                        if p in src.policy_scores and i < len(src.policy_scores[p]):
                            scores[p].append(src.policy_scores[p][i])
                rank = sorted([p for p in common if scores[p]],
                               key=lambda p: float(np.mean(scores[p])))
                taus_list.append(kendall_tau(rank, hifi_rank))
        tw = float(np.nanmean(taus_weighted))
        tu = float(np.nanmean(taus_uniform))
        weighted_taus[src_name] = {"tau_uniform": round(tu, 4), "tau_weighted": round(tw, 4),
                                    "delta": round(tw - tu, 4)}
        print(f"  {src_name:20s}  uniform={tu:.4f}  weighted={tw:.4f}  delta={tw-tu:+.4f}")

    _plot_difficulty_curves(all_source_results, focus, _OUT / "q5_difficulty_curves.png")

    out = {
        "experiment": "19_q5_robot_difficulty",
        "n_bootstrap": n_bootstrap,
        "hifi": HF,
        "difficulty_quantiles": {"q33": round(q33, 3), "q67": round(q67, 3)},
        "difficulty_bins": {k: len(v) for k, v in bins.items()},
        "source_results": all_source_results,
        "weighted_sampling": weighted_taus,
    }
    json_path = _OUT / "q5_robot_difficulty.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[19] Saved {json_path}")

    md_path = _OUT / "q5_robot_difficulty.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q5 Per-Robot Information Weighting by Target Difficulty\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={HF}\n")
        f.write(f"Difficulty quantiles: q33={q33:.2f}  q67={q67:.2f}  "
                f"(min HF score per experiment)\n\n")
        f.write("| Source | tau@N=1 easy | tau@N=1 medium | tau@N=1 hard | tau@N=1 full | Pattern |\n")
        f.write("|--------|-------------|----------------|-------------|-------------|--------|\n")
        for src_name, res in all_source_results.items():
            easy1  = res["bins"].get("easy",   {}).get("tau@N=1", float("nan"))
            med1   = res["bins"].get("medium", {}).get("tau@N=1", float("nan"))
            hard1  = res["bins"].get("hard",   {}).get("tau@N=1", float("nan"))
            full1  = res.get("tau@N=1_full", float("nan"))
            f.write(f"| {src_name} | {easy1:.3f} | {med1:.3f} | {hard1:.3f} "
                    f"| {full1:.3f} | {res['info_pattern']} |\n")
        f.write("\n## Difficulty-weighted sampling\n\n")
        f.write("| Source | uniform tau@N=10 | weighted tau@N=10 | delta |\n")
        f.write("|--------|-----------------|-------------------|-------|\n")
        for src_name, res in weighted_taus.items():
            f.write(f"| {src_name} | {res['tau_uniform']:.4f} | {res['tau_weighted']:.4f} "
                    f"| {res['delta']:+.4f} |\n")
    print(f"[19] Saved {md_path}")


def _plot_difficulty_curves(all_source_results, focus, out_path):
    n_src = len([s for s in focus if s in all_source_results])
    if n_src == 0:
        return
    cols = min(3, n_src)
    rows = (n_src + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    idx = 0
    colors = {"easy": "green", "medium": "orange", "hard": "red", "full": "black"}
    for src_name in focus:
        if src_name not in all_source_results:
            continue
        res = all_source_results[src_name]
        ax  = axes[idx // cols][idx % cols]
        # Full curve
        full_ns = sorted(int(n) for n in res["full_tau_per_n"])
        full_tv = [res["full_tau_per_n"][str(n)] for n in full_ns]
        ax.plot(full_ns, full_tv, "k-o", linewidth=2, label="full", markersize=4)
        # Bin curves
        for bin_name, bin_res in res.get("bins", {}).items():
            ns = sorted(int(n) for n in bin_res.get("tau_per_n", {}))
            tv = [bin_res["tau_per_n"][str(n)] for n in ns]
            if ns:
                ax.plot(ns, tv, "-s", color=colors.get(bin_name, "gray"),
                        label=bin_name, markersize=3, linewidth=1.5)
        ax.set_title(src_name, fontsize=9)
        ax.set_xlabel("N robots")
        ax.set_ylabel("tau vs HF")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        idx += 1
    # Hide unused axes
    for i in range(idx, rows * cols):
        axes[i // cols][i % cols].set_visible(False)
    plt.suptitle("Q5: tau@N curves by target difficulty (easy/medium/hard)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[19] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q5 per-robot difficulty weighting (Exp 19)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(n_bootstrap=args.n_bootstrap, seed=args.seed)
