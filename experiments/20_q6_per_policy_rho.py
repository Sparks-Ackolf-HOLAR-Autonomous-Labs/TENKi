"""
Experiment 20 -- Q6: Per-Policy Rho for MFMC.

For each paired LF source (mixbox, km, ryb), computes Pearson rho and
Spearman rho per policy between HF and LF per-experiment scores.

Reports:
  - Robust policies: rho > global_mean + 0.05
  - Fragile policies: rho < global_mean - 0.05
  - Policy-specific optimal LF:HF ratio (MFMC formula):
      r_p = |rho_p| * sqrt(cost_ratio) / sqrt(1 - rho_p^2)
  - Whether policy-specific rho meaningfully beats global rho in MSE reduction

Outputs
-------
  results/tenki_1000/q6_per_policy_rho.json
  results/tenki_1000/q6_per_policy_rho.md
  results/tenki_1000/q6_rho_heatmap.png
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
from scipy.stats import spearmanr, pearsonr

from analysis.flip_data import load_many_studies, common_policy_subset, restrict_to_common_policies

_OUT = Path(_HERE).parent / "results" / "tenki_1000"
_OUT.mkdir(parents=True, exist_ok=True)

HF_SOURCE = "spectral"
HF_PATH   = "output/db_1000_spectral"

# Paired sources only (share experiment indices with spectral)
PAIRED_SOURCES = {
    "mixbox": ("output/db_1000_mixbox",  0.3),   # (path, rel_cost)
    "km":     ("output/db_1000_km",      0.4),
    "ryb":    ("output/db_1000_ryb",     0.2),
}


def run(seed=42):
    study_map = {HF_SOURCE: HF_PATH, **{n: p for n, (p, _) in PAIRED_SOURCES.items()}}
    print("[20] Loading databases...")
    all_studies = load_many_studies(study_map)
    print(f"  Loaded: {list(all_studies.keys())}")

    common = common_policy_subset(all_studies)
    print(f"  Common policies: {len(common)}")
    if len(common) < 2:
        print("  Not enough common policies -- aborting.")
        return

    studies_c = {n: restrict_to_common_policies(s, common) for n, s in all_studies.items()}
    hifi_c    = studies_c[HF_SOURCE]

    source_results = {}
    for src_name, (_, rel_cost) in PAIRED_SOURCES.items():
        if src_name not in studies_c:
            print(f"  SKIP {src_name}: not loaded")
            continue
        src = studies_c[src_name]
        print(f"\n  {src_name}  (rel_cost={rel_cost})")

        policy_stats: dict[str, dict] = {}
        pearson_rhos, spearman_rhos = [], []

        for p in common:
            hf_scores = hifi_c.policy_scores.get(p, [])
            lf_scores = src.policy_scores.get(p, [])
            n = min(len(hf_scores), len(lf_scores))
            if n < 5:
                continue
            hf = np.array(hf_scores[:n])
            lf = np.array(lf_scores[:n])

            r_pearson, _  = pearsonr(hf, lf)
            r_spearman, _ = spearmanr(hf, lf)
            r_pearson  = float(r_pearson)
            r_spearman = float(r_spearman)

            # MFMC optimal ratio for this policy (uses Pearson rho, per Peherstorfer 2016)
            r_abs = abs(r_pearson)
            if r_abs < 0.999:
                mfmc_ratio = r_abs * np.sqrt(1.0 / rel_cost) / np.sqrt(1 - r_abs**2)
            else:
                mfmc_ratio = float("inf")

            # Variance reduction factor (classical MFMC, from Peherstorfer 2016)
            var_reduction = 1 - r_pearson**2 if abs(r_pearson) < 1 else 0.0

            policy_stats[p] = {
                "pearson_rho":   round(r_pearson,  4),
                "spearman_rho":  round(r_spearman, 4),
                "mfmc_ratio":    round(mfmc_ratio, 3) if not np.isinf(mfmc_ratio) else None,
                "var_reduction": round(var_reduction, 4),
                "n_paired":      n,
            }
            pearson_rhos.append(r_pearson)
            spearman_rhos.append(r_spearman)

        if not policy_stats:
            continue

        global_pearson  = float(np.mean(pearson_rhos))
        global_spearman = float(np.mean(spearman_rhos))
        std_pearson     = float(np.std(pearson_rhos))
        std_spearman    = float(np.std(spearman_rhos))

        robust   = sorted([p for p, s in policy_stats.items() if s["spearman_rho"] >= global_spearman + 0.05])
        fragile  = sorted([p for p, s in policy_stats.items() if s["spearman_rho"] <= global_spearman - 0.05])
        moderate = sorted([p for p in policy_stats if p not in robust and p not in fragile])

        # Global MFMC ratio (uses Pearson rho, per Peherstorfer 2016)
        r_global = abs(global_pearson)
        global_mfmc_ratio = (r_global * np.sqrt(1.0 / rel_cost) / np.sqrt(1 - r_global**2)
                              if r_global < 0.999 else float("inf"))

        # Does policy-specific ratio improve over global?
        per_policy_ratios = [s["mfmc_ratio"] for s in policy_stats.values() if s["mfmc_ratio"] is not None]
        ratio_std = float(np.std(per_policy_ratios)) if per_policy_ratios else 0.0
        per_policy_useful = ratio_std > 0.5  # if ratios vary significantly across policies

        source_results[src_name] = {
            "global_pearson_rho":   round(global_pearson,  4),
            "global_spearman_rho":  round(global_spearman, 4),
            "std_spearman_rho":     round(std_spearman,    4),
            "global_mfmc_ratio":    round(global_mfmc_ratio, 3),
            "per_policy_ratio_std": round(ratio_std, 3),
            "per_policy_useful":    per_policy_useful,
            "n_policies":           len(policy_stats),
            "n_robust":             len(robust),
            "n_fragile":            len(fragile),
            "robust_policies":      robust,
            "fragile_policies":     fragile,
            "per_policy":           policy_stats,
        }

        print(f"  global rho (Spearman)={global_spearman:.4f}±{std_spearman:.4f}  "
              f"mfmc_ratio={global_mfmc_ratio:.3f}")
        print(f"  robust={len(robust)} fragile={len(fragile)} moderate={len(moderate)}")
        print(f"  per-policy useful (ratio_std > 0.5): {per_policy_useful}")
        if robust:
            print(f"  robust: {robust[:5]}")
        if fragile:
            print(f"  fragile: {fragile[:5]}")

    _plot_rho_heatmap(source_results, common, _OUT / "q6_rho_heatmap.png")

    out = {
        "experiment": "20_q6_per_policy_rho",
        "hifi": HF_SOURCE,
        "paired_sources": list(PAIRED_SOURCES.keys()),
        "sources": source_results,
    }
    json_path = _OUT / "q6_per_policy_rho.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[20] Saved {json_path}")

    md_path = _OUT / "q6_per_policy_rho.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q6 Per-Policy Rho for MFMC\n\n")
        f.write(f"hifi={HF_SOURCE}  paired sources: {', '.join(PAIRED_SOURCES.keys())}\n\n")
        f.write("## Global summary\n\n")
        f.write("| Source | global rho | std | global MFMC ratio | ratio_std | per-policy useful? |\n")
        f.write("|--------|-----------|-----|-------------------|-----------|-----------------|\n")
        for src_name, res in source_results.items():
            f.write(f"| {src_name} | {res['global_spearman_rho']:.4f} | {res['std_spearman_rho']:.4f} "
                    f"| {res['global_mfmc_ratio']:.3f} | {res['per_policy_ratio_std']:.3f} "
                    f"| {res['per_policy_useful']} |\n")
        f.write("\n## Robust and fragile policies per source\n\n")
        for src_name, res in source_results.items():
            f.write(f"### {src_name}\n\n")
            f.write(f"- global rho = {res['global_spearman_rho']:.4f} ± {res['std_spearman_rho']:.4f}\n")
            f.write(f"- Robust (rho > global+0.05): {', '.join(res['robust_policies']) or 'none'}\n")
            f.write(f"- Fragile (rho < global-0.05): {', '.join(res['fragile_policies']) or 'none'}\n\n")
            f.write("| Policy | Spearman rho | Pearson rho | MFMC ratio | Var reduction |\n")
            f.write("|--------|-------------|-------------|------------|---------------|\n")
            for p in sorted(res["per_policy"], key=lambda p: -res["per_policy"][p]["spearman_rho"]):
                s = res["per_policy"][p]
                f.write(f"| {p} | {s['spearman_rho']:.4f} | {s['pearson_rho']:.4f} "
                        f"| {s['mfmc_ratio'] or 'inf'} | {s['var_reduction']:.4f} |\n")
            f.write("\n")
    print(f"[20] Saved {md_path}")


def _plot_rho_heatmap(source_results, common, out_path):
    src_names = list(source_results.keys())
    if not src_names:
        return
    policies = sorted(common)
    matrix = np.full((len(policies), len(src_names)), float("nan"))
    for j, src_name in enumerate(src_names):
        for i, p in enumerate(policies):
            rho = source_results[src_name]["per_policy"].get(p, {}).get("spearman_rho", float("nan"))
            matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(max(4, len(src_names) * 2), max(4, len(policies) * 0.4 + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(src_names)))
    ax.set_xticklabels(src_names, rotation=30, ha="right")
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies, fontsize=7)
    plt.colorbar(im, ax=ax, label="Spearman rho (LF vs HF score correlation)")
    ax.set_title("Q6: Per-policy LF/HF Spearman rho by source")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[20] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q6 per-policy rho for MFMC (Exp 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(seed=args.seed)
