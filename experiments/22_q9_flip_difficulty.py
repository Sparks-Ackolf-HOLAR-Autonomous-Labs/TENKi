"""
Experiment 22 -- Q9: Flip N* Sensitivity to Target Difficulty.

Stratifies experiments by target difficulty (min achievable HF score)
into easy / medium / hard thirds, then computes tau(N) and N* separately
per difficulty bin for each frugal source.

Decision criterion:
  - If a source flips on easy but never on hard  -> "CONDITIONALLY_USEFUL"
  - If a source flips on all three bins           -> "ROBUST_DONOR"
  - If a source never flips on any bin            -> "PERMANENT_GAP"
  - If a source flips on hard but not easy        -> "HARD_SPECIALIST"

This separates receivers that are "good enough for easy targets" from
ones that are genuinely orthogonal donors across the full difficulty range.

Outputs
-------
  results/tenki_1000/q9_flip_difficulty.json
  results/tenki_1000/q9_flip_difficulty.md
  results/tenki_1000/q9_flip_nstar_heatmap.png
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

from analysis.flip_data import (
    load_many_studies, common_policy_subset, restrict_to_common_policies, StudyScores
)
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
FLIP_THRESHOLD = 0.5   # tau > 0.5 counts as "flipped to useful"
N_STAR_MIN     = 1
N_STAR_MAX     = 30
FOCUS_SOURCES = [
    "study_b", "study_b_reverse", "mixbox", "ryb",
    "km", "study_a", "study_c", "study_c_reverse",
]


def _make_bin_study(src_name, src_scores, hf_scores, exp_idxs, common):
    """Build a StudyScores restricted to experiment indices in exp_idxs."""
    bin_lf = {}
    for p in common:
        if p not in src_scores:
            continue
        vals = [src_scores[p][i] for i in exp_idxs if i < len(src_scores[p])]
        if len(vals) >= 2:
            bin_lf[p] = vals
    if not bin_lf:
        return None, None

    bin_hf_rank = sorted(
        [p for p in common if p in hf_scores and p in bin_lf],
        key=lambda p: float(np.mean([hf_scores[p][i] for i in exp_idxs
                                     if i < len(hf_scores[p])])),
    )
    study = StudyScores(
        name=f"{src_name}_bin",
        db_path="",
        policy_scores=bin_lf,
        full_rank=sorted(bin_lf, key=lambda p: float(np.mean(bin_lf[p]))),
        n_policies=len(bin_lf),
        max_n=min(len(v) for v in bin_lf.values()),
    )
    return study, bin_hf_rank


def _find_nstar(tau_per_n: dict[int, float], threshold: float) -> int | None:
    """Return smallest N where mean_tau >= threshold, or None."""
    for n in sorted(tau_per_n):
        if tau_per_n[n] >= threshold:
            return n
    return None


def run(n_values=None, n_bootstrap=N_BOOTSTRAP, seed=42):
    n_vals = sorted(set(n_values or N_VALUES))

    print("[22] Loading databases...")
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

    # Per-experiment difficulty: min HF score across policies
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
    print(f"\n  Difficulty bins  q33={q33:.2f}  q67={q67:.2f}")
    for bname, idxs in bins.items():
        print(f"    {bname}: {len(idxs)} experiments")

    focus = [s for s in FOCUS_SOURCES if s in studies_c]
    all_results = {}

    for src_name in focus:
        src = studies_c[src_name]
        src_scores = src.policy_scores
        print(f"\n  {src_name}  (max_n={src.max_n})")

        # Full-pool tau curve and N*
        n_scan = [n for n in n_vals if n <= src.max_n] or [1]
        curve_full = bootstrap_tau_curve(src, hifi_rank, common, n_scan, n_bootstrap,
                                          rng_seed=seed + abs(hash(src_name)) % 9999)
        tau_full = {n: curve_full.get(n, {}).get("mean_tau", float("nan")) for n in n_scan}
        nstar_full = _find_nstar(tau_full, FLIP_THRESHOLD)
        ceiling = full_data_ceiling(src, hifi_c, common)

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
            bin_policies = list(bin_study.policy_scores.keys())
            curve_bin = bootstrap_tau_curve(
                bin_study, bin_hf_rank, bin_policies,
                n_scan_bin, n_bootstrap,
                rng_seed=seed + abs(hash(src_name + bin_name)) % 9999,
            )
            tau_bin = {n: curve_bin.get(n, {}).get("mean_tau", float("nan")) for n in n_scan_bin}
            tau_bin_std = {n: curve_bin.get(n, {}).get("std_tau", float("nan")) for n in n_scan_bin}
            nstar_bin = _find_nstar(tau_bin, FLIP_THRESHOLD)

            tau_at_n10 = tau_bin.get(10, tau_bin.get(max(n_scan_bin), float("nan")))
            flipped = tau_at_n10 >= FLIP_THRESHOLD

            bin_results[bin_name] = {
                "n_exps":       len(exp_idxs),
                "mean_diff":    round(float(np.mean([per_exp_diff[i] for i in exp_idxs])), 3),
                "tau_per_n":    {str(n): round(v, 4) for n, v in tau_bin.items()},
                "tau_std_per_n": {str(n): round(v, 4) for n, v in tau_bin_std.items()},
                "tau@N=10":     round(tau_at_n10, 4),
                "nstar":        nstar_bin,
                "flipped":      flipped,
            }
            flip_str = "FLIP" if flipped else "no flip"
            nstar_str = f"N*={nstar_bin}" if nstar_bin else "N*=None"
            print(f"    {bin_name:6s}  tau@10={tau_at_n10:.3f}  {nstar_str}  {flip_str}")

        # Classify this source's difficulty-stratified flip pattern
        flipped_easy   = bin_results.get("easy",   {}).get("flipped", False)
        flipped_medium = bin_results.get("medium", {}).get("flipped", False)
        flipped_hard   = bin_results.get("hard",   {}).get("flipped", False)

        n_flipped = sum([flipped_easy, flipped_medium, flipped_hard])
        if n_flipped == 3:
            diff_class = "ROBUST_DONOR"
        elif flipped_easy and not flipped_hard:
            diff_class = "CONDITIONALLY_USEFUL"
        elif flipped_hard and not flipped_easy:
            diff_class = "HARD_SPECIALIST"
        elif n_flipped == 0:
            if nstar_full is None:
                diff_class = "PERMANENT_GAP"
            else:
                diff_class = "MARGINAL"  # flips overall but bins don't (small-sample noise)
        else:
            diff_class = "MIXED"

        print(f"    => {diff_class}  (ceiling={ceiling:.3f}  nstar_full={nstar_full})")

        all_results[src_name] = {
            "ceiling":        round(ceiling, 4),
            "nstar_full":     nstar_full,
            "tau_full_per_n": {str(n): round(v, 4) for n, v in tau_full.items()},
            "bins":           bin_results,
            "difficulty_class": diff_class,
            "flipped_easy":   flipped_easy,
            "flipped_medium": flipped_medium,
            "flipped_hard":   flipped_hard,
        }

    _plot_nstar_heatmap(all_results, focus, bins, _OUT / "q9_flip_nstar_heatmap.png")

    out = {
        "experiment": "22_q9_flip_difficulty",
        "n_bootstrap": n_bootstrap,
        "hifi": HF,
        "flip_threshold": FLIP_THRESHOLD,
        "difficulty_quantiles": {"q33": round(q33, 3), "q67": round(q67, 3)},
        "difficulty_bin_sizes": {k: len(v) for k, v in bins.items()},
        "source_results": all_results,
    }
    json_path = _OUT / "q9_flip_difficulty.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[22] Saved {json_path}")

    md_path = _OUT / "q9_flip_difficulty.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q9 Flip N* Sensitivity to Target Difficulty\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={HF}  flip_threshold={FLIP_THRESHOLD}\n")
        f.write(f"Difficulty quantiles: q33={q33:.2f}  q67={q67:.2f}  "
                f"(min HF score per experiment)\n\n")
        f.write("## Difficulty-stratified flip classification\n\n")
        f.write("| Source | easy flip | medium flip | hard flip | N* full | ceiling | class |\n")
        f.write("|--------|-----------|-------------|-----------|---------|---------|-------|\n")
        for src_name, res in all_results.items():
            fe = res["flipped_easy"]
            fm = res["flipped_medium"]
            fh = res["flipped_hard"]
            f.write(f"| {src_name} "
                    f"| {'Y' if fe else 'n'} "
                    f"| {'Y' if fm else 'n'} "
                    f"| {'Y' if fh else 'n'} "
                    f"| {res['nstar_full'] or 'None'} "
                    f"| {res['ceiling']:.3f} "
                    f"| {res['difficulty_class']} |\n")
        f.write("\n**Class definitions**:\n")
        f.write("- ROBUST_DONOR: flips in all three difficulty bins\n")
        f.write("- CONDITIONALLY_USEFUL: flips on easy but not hard\n")
        f.write("- HARD_SPECIALIST: flips on hard but not easy\n")
        f.write("- PERMANENT_GAP: never flips regardless of difficulty\n")
        f.write("- MARGINAL: full-pool flips but bins have insufficient data\n")
        f.write("- MIXED: other partial pattern\n\n")
        f.write("## Per-source bin detail\n\n")
        for src_name, res in all_results.items():
            f.write(f"### {src_name} — {res['difficulty_class']}\n\n")
            f.write(f"ceiling={res['ceiling']:.3f}  N*_full={res['nstar_full']}\n\n")
            f.write("| Bin | n_exps | mean_diff | tau@N=10 | N* |\n")
            f.write("|-----|--------|-----------|----------|----|\n")

            for bname in ["easy", "medium", "hard"]:
                b = res["bins"].get(bname)
                if b:
                    f.write(f"| {bname} | {b['n_exps']} | {b['mean_diff']:.3f} "
                            f"| {b['tau@N=10']:.4f} | {b['nstar'] or 'None'} |\n")
            f.write("\n")
    print(f"[22] Saved {md_path}")


def _plot_nstar_heatmap(all_results, focus, bins, out_path):
    src_names = [s for s in focus if s in all_results]
    if not src_names:
        return
    bin_names = ["easy", "medium", "hard"]
    # Build tau@N=10 matrix
    matrix = np.full((len(bin_names), len(src_names)), float("nan"))
    for j, src in enumerate(src_names):
        for i, bname in enumerate(bin_names):
            matrix[i, j] = all_results[src]["bins"].get(bname, {}).get("tau@N=10", float("nan"))

    fig, ax = plt.subplots(figsize=(max(5, len(src_names) * 1.4), 3))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(src_names)))
    ax.set_xticklabels(src_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(bin_names)))
    ax.set_yticklabels(bin_names)
    plt.colorbar(im, ax=ax, label="tau@N=10")
    # Annotate with tau values
    for i in range(len(bin_names)):
        for j in range(len(src_names)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.2 < val < 0.8 else "white")
    # Annotate difficulty class
    for j, src in enumerate(src_names):
        dc = all_results[src]["difficulty_class"]
        short = {"ROBUST_DONOR": "RD", "CONDITIONALLY_USEFUL": "CU",
                 "HARD_SPECIALIST": "HS", "PERMANENT_GAP": "PG",
                 "MARGINAL": "MG", "MIXED": "MX"}.get(dc, dc[:2])
        ax.text(j, len(bin_names) - 0.5, short, ha="center", va="bottom",
                fontsize=7, color="navy", fontweight="bold")
    ax.set_title(f"Q9: Flip tau@N=10 by difficulty  (threshold={0.5})", fontsize=10)
    ax.axhline(len(bin_names) - 0.5, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[22] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q9 Flip N* by difficulty (Exp 22)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(n_bootstrap=args.n_bootstrap, seed=args.seed)
