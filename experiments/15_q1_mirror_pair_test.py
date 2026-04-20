"""
Experiment 15 -- Q1: Mirror-Pair Symmetry Test.

For each mirror pair (G_A\\G_B, G_B\\G_A), compare:
  tau@N=1, tau@N=10, ceiling, bias floor, donor score
  N* (mutual flip), bootstrap CI overlap at N=10

Decision: mirror symmetry supported only if 1-sigma bootstrap CIs overlap
at N=10 AND ceiling difference < 0.05.

Outputs
-------
  results/tenki_1000/q1_mirror_pairs.json
  results/tenki_1000/q1_mirror_pairs.md
"""

from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
from analysis.flip_data import load_many_studies, common_policy_subset, restrict_to_common_policies
from analysis.flip_metrics import bootstrap_tau_curve, full_data_ceiling, kendall_tau
from analysis.flip_models import mutual_flip_result

_OUT = Path(_HERE).parent / "results" / "tenki_1000"
_OUT.mkdir(parents=True, exist_ok=True)

# Mirror pairs: (forward_name, forward_path, reverse_name, reverse_path)
DEFAULT_MIRROR_PAIRS = [
    (
        "study_b",
        "output/db_1000_study_b_physics_vs_artist",
        "study_b_reverse",
        "output/db_1000_study_b_reverse_artist_vs_physics",
    ),
    (
        "study_c",
        "output/db_1000_study_c_oilpaint_vs_fooddye",
        "study_c_reverse",
        "output/db_1000_study_c_reverse_fooddye_vs_oilpaint",
    ),
]

DEFAULT_HF = ("spectral", "output/db_1000_spectral")
DEFAULT_N_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
N_BOOTSTRAP = 300


def run(mirror_pairs=None, hf_pair=None, n_values=None, n_bootstrap=N_BOOTSTRAP, seed=42):
    mirror_pairs = mirror_pairs or DEFAULT_MIRROR_PAIRS
    hf_name, hf_path = hf_pair or DEFAULT_HF
    n_vals = sorted(set(n_values or DEFAULT_N_VALUES))

    # Load all databases
    study_map = {hf_name: hf_path}
    for fwd_name, fwd_path, rev_name, rev_path in mirror_pairs:
        study_map[fwd_name] = fwd_path
        study_map[rev_name] = rev_path

    print("[15] Loading databases...")
    all_studies = load_many_studies(study_map)
    loaded = list(all_studies.keys())
    print(f"  Loaded: {loaded}")

    hifi = all_studies.get(hf_name)
    common = common_policy_subset(all_studies)
    print(f"  Common policies: {len(common)}")
    if len(common) < 2:
        print("  Not enough common policies -- aborting.")
        return

    studies_c = {n: restrict_to_common_policies(s, common) for n, s in all_studies.items()}
    hifi_c    = studies_c[hf_name]
    hifi_rank = [p for p in hifi_c.full_rank if p in common]

    results = []
    for fwd_name, _, rev_name, _ in mirror_pairs:
        if fwd_name not in studies_c or rev_name not in studies_c:
            print(f"  SKIP {fwd_name} <-> {rev_name}: not loaded")
            continue

        s_fwd = studies_c[fwd_name]
        s_rev = studies_c[rev_name]

        print(f"\n  {fwd_name} <-> {rev_name}")

        # Bootstrap tau@N curves for each direction vs HF
        n_scan_fwd = [n for n in n_vals if n <= s_fwd.max_n] or [1]
        n_scan_rev = [n for n in n_vals if n <= s_rev.max_n] or [1]

        curve_fwd = bootstrap_tau_curve(s_fwd, hifi_rank, common, n_scan_fwd, n_bootstrap,
                                        rng_seed=seed + abs(hash(fwd_name)) % 9999)
        curve_rev = bootstrap_tau_curve(s_rev, hifi_rank, common, n_scan_rev, n_bootstrap,
                                        rng_seed=seed + abs(hash(rev_name)) % 9999)

        tau1_fwd  = curve_fwd.get(1,  {}).get("mean_tau", float("nan"))
        tau10_fwd = curve_fwd.get(10, {}).get("mean_tau", float("nan"))
        std10_fwd = curve_fwd.get(10, {}).get("std_tau",  float("nan"))
        tau1_rev  = curve_rev.get(1,  {}).get("mean_tau", float("nan"))
        tau10_rev = curve_rev.get(10, {}).get("mean_tau", float("nan"))
        std10_rev = curve_rev.get(10, {}).get("std_tau",  float("nan"))

        ceiling_fwd = full_data_ceiling(s_fwd, hifi_c, common)
        ceiling_rev = full_data_ceiling(s_rev, hifi_c, common)
        bias_fwd    = 1.0 - ceiling_fwd
        bias_rev    = 1.0 - ceiling_rev

        # Mutual flip N*
        mut = mutual_flip_result(
            source=s_fwd, competitor=s_rev, policies=common,
            n_values=n_scan_fwd, n_bootstrap=n_bootstrap,
            eps=0.01, rng_seed=seed + abs(hash(fwd_name + rev_name)) % 9999,
        )

        # CI overlap at N=10
        lo_fwd = tau10_fwd - std10_fwd
        hi_fwd = tau10_fwd + std10_fwd
        lo_rev = tau10_rev - std10_rev
        hi_rev = tau10_rev + std10_rev
        ci_overlap = not (hi_fwd < lo_rev or hi_rev < lo_fwd)

        # Mirror symmetry verdict
        ceil_diff = abs(ceiling_fwd - ceiling_rev)
        verdict = (
            "SYMMETRIC"
            if ci_overlap and ceil_diff < 0.05
            else "ASYMMETRIC"
        )

        row = {
            "forward":      fwd_name,
            "reverse":      rev_name,
            "tau@N=1_fwd":  round(tau1_fwd,   4),
            "tau@N=1_rev":  round(tau1_rev,   4),
            "tau@N=10_fwd": round(tau10_fwd,  4),
            "tau@N=10_rev": round(tau10_rev,  4),
            "std@N=10_fwd": round(std10_fwd,  4),
            "std@N=10_rev": round(std10_rev,  4),
            "ceiling_fwd":  round(ceiling_fwd, 4),
            "ceiling_rev":  round(ceiling_rev, 4),
            "bias_fwd":     round(bias_fwd,    4),
            "bias_rev":     round(bias_rev,    4),
            "ceil_diff":    round(ceil_diff,   4),
            "ci_overlap":   ci_overlap,
            "mutual_N*":    mut.flip_n,
            "verdict":      verdict,
        }
        results.append(row)

        print(f"    tau@N=1  : fwd={tau1_fwd:.3f}  rev={tau1_rev:.3f}")
        print(f"    tau@N=10 : fwd={tau10_fwd:.3f}±{std10_fwd:.3f}  rev={tau10_rev:.3f}±{std10_rev:.3f}")
        print(f"    ceiling  : fwd={ceiling_fwd:.3f}  rev={ceiling_rev:.3f}  diff={ceil_diff:.3f}")
        print(f"    mutual N*: {mut.flip_n}  CI overlap: {ci_overlap}  => {verdict}")

    # Summarise
    n_sym  = sum(1 for r in results if r["verdict"] == "SYMMETRIC")
    n_asym = len(results) - n_sym
    conclusion = (
        "ALL SYMMETRIC" if n_sym == len(results)
        else f"{n_asym}/{len(results)} ASYMMETRIC"
    )
    print(f"\n  Q1 conclusion: {conclusion}")

    # Write JSON
    out = {
        "experiment": "15_q1_mirror_pair_test",
        "n_bootstrap": n_bootstrap,
        "hifi": hf_name,
        "pairs": results,
        "conclusion": conclusion,
    }
    json_path = _OUT / "q1_mirror_pairs.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[15] Saved {json_path}")

    # Write Markdown
    md_path = _OUT / "q1_mirror_pairs.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q1 Mirror-Pair Symmetry Test\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={hf_name}\n\n")
        f.write("| Pair | tau@N=1 fwd | tau@N=1 rev | tau@N=10 fwd | tau@N=10 rev | ceiling fwd | ceiling rev | ceil diff | CI overlap | mutual N* | Verdict |\n")
        f.write("|------|-------------|-------------|--------------|--------------|-------------|-------------|-----------|------------|-----------|--------|\n")
        for r in results:
            f.write(
                f"| {r['forward']} / {r['reverse']} "
                f"| {r['tau@N=1_fwd']:.3f} | {r['tau@N=1_rev']:.3f} "
                f"| {r['tau@N=10_fwd']:.3f}±{r['std@N=10_fwd']:.3f} "
                f"| {r['tau@N=10_rev']:.3f}±{r['std@N=10_rev']:.3f} "
                f"| {r['ceiling_fwd']:.3f} | {r['ceiling_rev']:.3f} "
                f"| {r['ceil_diff']:.3f} | {r['ci_overlap']} "
                f"| {r['mutual_N*']} | **{r['verdict']}** |\n"
            )
        f.write(f"\n**Conclusion**: {conclusion}\n\n")
        f.write("Mirror symmetry = CI overlap at N=10 AND ceiling difference < 0.05.\n"
                "Asymmetry is physical/coverage-driven when bias floors or ceilings differ significantly.\n")
    print(f"[15] Saved {md_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q1 mirror-pair symmetry test (Exp 15)")
    p.add_argument("--hifi", default="spectral")
    p.add_argument("--hifi-path", default="output/db_1000_spectral")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(hf_pair=(args.hifi, args.hifi_path), n_bootstrap=args.n_bootstrap, seed=args.seed)
