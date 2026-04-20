"""
Experiment 21 -- Q7: TrueSkill2 Multi-Team Rating.

Teams = (source, n_robots).  Match outcome = tau vs HF ranking.
Runs TrueSkill2-style updates by treating each (source, N) pair as a team
and their tau value as a "skill + noise" observation.

Tests whether TrueSkill2 mu recovers the same source taxonomy as TENKi:
  donors high, receivers low, high-ceiling slow-convergers distinct
  from low-ceiling bad donors.

Method:
  For each bootstrap trial, rank all (source, N) teams by observed tau.
  Run top-k pairwise rate() calls to update ratings.
  Aggregate mu per source (mean over N values).
  Compare ranking to TENKi donor taxonomy via Kendall tau.

Requires: pip install trueskill

Outputs
-------
  results/tenki_1000/q7_trueskill2.json
  results/tenki_1000/q7_trueskill2.md
  results/tenki_1000/q7_ts2_mu_vs_tau.png
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
HF = "spectral"

# TENKi canonical donor taxonomy (from Exp 03 + synthesis table)
TENKI_DONOR_ORDER = [
    "spectral", "study_b", "mixbox", "ryb",
    "study_c_reverse", "study_b_reverse", "km", "study_a", "study_c",
]

N_VALUES    = [1, 2, 5, 10, 20, 50]
N_BOOTSTRAP = 150


def run(n_values=None, n_bootstrap=N_BOOTSTRAP, seed=42):
    try:
        import trueskill
    except ImportError:
        print("[21] trueskill package not installed.")
        print("     Install with: pip install trueskill")
        print("     Skipping Q7.")
        return

    n_vals = sorted(set(n_values or N_VALUES))

    print("[21] Loading databases...")
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

    # Build tau@N table for all (source, N) teams
    print("\n  Building tau@N table...")
    team_tau_mean: dict[tuple[str, int], float] = {}
    team_tau_std:  dict[tuple[str, int], float] = {}

    for src_name in frugal:
        src = studies_c[src_name]
        n_scan = [n for n in n_vals if n <= src.max_n] or [1]
        curve = bootstrap_tau_curve(src, hifi_rank, common, n_scan, n_bootstrap,
                                     rng_seed=seed + abs(hash(src_name)) % 9999)
        for n in n_scan:
            stats = curve.get(n, {})
            team_tau_mean[(src_name, n)] = stats.get("mean_tau", float("nan"))
            team_tau_std[(src_name, n)]  = stats.get("std_tau",  0.1)
            print(f"  {src_name:25s} N={n:4d}  tau={team_tau_mean[(src_name,n)]:.4f}"
                  f"±{team_tau_std[(src_name,n)]:.4f}")

    # Initialise TrueSkill environment
    env = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.167, tau=0.0833,
                               draw_probability=0.02, backend="scipy")
    teams = list(team_tau_mean.keys())
    ratings: dict[tuple[str, int], trueskill.Rating] = {
        t: env.create_rating() for t in teams
    }

    # Run matches: sort teams by mean tau; feed sequential pairwise updates
    n_matches = 0
    sorted_teams = sorted(teams, key=lambda t: team_tau_mean[t], reverse=True)

    for i in range(len(sorted_teams)):
        for j in range(i + 1, min(i + 5, len(sorted_teams))):
            ta, tb = sorted_teams[i], sorted_teams[j]
            tau_a = team_tau_mean[ta]
            tau_b = team_tau_mean[tb]
            if np.isnan(tau_a) or np.isnan(tau_b):
                continue
            try:
                if tau_a > tau_b + 0.02:
                    (ratings[ta],), (ratings[tb],) = env.rate(
                        [[ratings[ta]], [ratings[tb]]], ranks=[0, 1]
                    )
                elif tau_b > tau_a + 0.02:
                    (ratings[tb],), (ratings[ta],) = env.rate(
                        [[ratings[tb]], [ratings[ta]]], ranks=[0, 1]
                    )
                else:
                    # Draw
                    (ratings[ta],), (ratings[tb],) = env.rate(
                        [[ratings[ta]], [ratings[tb]]], ranks=[0, 0]
                    )
                n_matches += 1
            except Exception:
                pass

    print(f"\n  Matches run: {n_matches}")

    # Aggregate mu per source (mean and min over N values)
    source_mu_all:  dict[str, list[float]] = {}
    source_tau_all: dict[str, list[float]] = {}
    for (src, n), rating in ratings.items():
        source_mu_all.setdefault(src, []).append(rating.mu)
        t = team_tau_mean.get((src, n), float("nan"))
        if not np.isnan(t):
            source_tau_all.setdefault(src, []).append(t)

    source_summary = {}
    for src in frugal:
        mus  = source_mu_all.get(src, [25.0])
        taus = source_tau_all.get(src, [float("nan")])
        source_summary[src] = {
            "ts2_mu_mean":    round(float(np.mean(mus)),  3),
            "ts2_mu_min":     round(float(np.min(mus)),   3),
            "ts2_mu_max":     round(float(np.max(mus)),   3),
            "tau_mean":       round(float(np.nanmean(taus)), 4),
            "team_ratings":   {
                str(n): round(ratings.get((src, n), env.create_rating()).mu, 3)
                for n in n_vals if (src, n) in ratings
            },
        }

    ts2_ranking  = sorted(source_summary, key=lambda s: -source_summary[s]["ts2_mu_mean"])
    tenki_subset = [n for n in TENKI_DONOR_ORDER if n in set(ts2_ranking)]
    ts2_subset   = [n for n in ts2_ranking if n in set(tenki_subset)]

    tau_to_tenki = kendall_tau(ts2_subset, tenki_subset)
    ts2_useful   = tau_to_tenki > 0.6

    print(f"\n  TrueSkill2 ranking: {ts2_ranking}")
    print(f"  TENKi taxonomy:     {tenki_subset}")
    print(f"  Kendall tau(TS2 vs TENKi): {tau_to_tenki:.4f}")
    print(f"  TS2 useful (tau > 0.6): {ts2_useful}")
    print("\n  Per-source summary:")
    for src in ts2_ranking:
        s = source_summary[src]
        print(f"  {src:25s}  mu={s['ts2_mu_mean']:.2f}  tau_mean={s['tau_mean']:.4f}")

    _plot_mu_vs_tau(source_summary, ts2_ranking, _OUT / "q7_ts2_mu_vs_tau.png")

    out = {
        "experiment": "21_q7_trueskill2",
        "n_bootstrap": n_bootstrap,
        "hifi": HF,
        "n_matches": n_matches,
        "ts2_ranking": ts2_ranking,
        "tenki_donor_order": tenki_subset,
        "tau_ts2_vs_tenki": round(tau_to_tenki, 4),
        "ts2_useful": ts2_useful,
        "source_summary": source_summary,
    }
    json_path = _OUT / "q7_trueskill2.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[21] Saved {json_path}")

    md_path = _OUT / "q7_trueskill2.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Q7 TrueSkill2 Multi-Team Rating\n\n")
        f.write(f"n_bootstrap={n_bootstrap}  hifi={HF}  matches={n_matches}\n\n")
        f.write(f"**Kendall tau(TS2 ranking vs TENKi taxonomy)**: {tau_to_tenki:.4f}\n\n")
        f.write(f"**TS2 useful** (tau > 0.6 threshold): {ts2_useful}\n\n")
        if ts2_useful:
            f.write("TS2 recovers the same source taxonomy as TENKi. It can be used as an "
                    "interpretive add-on but not as a replacement — TENKi bias floor and ceiling "
                    "diagnostics remain required.\n\n")
        else:
            f.write("TS2 does NOT reliably recover the TENKi taxonomy. The mu ranking does not "
                    "distinguish donors from receivers as well as TENKi's direct tau diagnostics. "
                    "Use TENKi as the primary method; TS2 is redundant here.\n\n")
        f.write("## Source mu summary\n\n")
        f.write("| Source | TS2 mu | tau@N=10 | TENKi rank |\n")
        f.write("|--------|--------|----------|------------|\n")
        for src in ts2_ranking:
            s = source_summary[src]
            tenki_pos = tenki_subset.index(src) + 1 if src in tenki_subset else "—"
            f.write(f"| {src} | {s['ts2_mu_mean']:.2f} | {s['tau_mean']:.4f} | {tenki_pos} |\n")
    print(f"[21] Saved {md_path}")


def _plot_mu_vs_tau(source_summary, ts2_ranking, out_path):
    src_names = ts2_ranking
    mus  = [source_summary[s]["ts2_mu_mean"]  for s in src_names]
    taus = [source_summary[s]["tau_mean"]      for s in src_names]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(taus, mus, s=80, zorder=3)
    for src, mu, tau in zip(src_names, mus, taus):
        ax.annotate(src, (tau, mu), textcoords="offset points", xytext=(5, 3), fontsize=7)
    from scipy.stats import pearsonr
    if len(taus) > 2:
        r, _ = pearsonr(taus, mus)
        ax.set_title(f"Q7: TS2 mu vs tau@N (mean over N values)  r={r:.3f}")
    else:
        ax.set_title("Q7: TS2 mu vs tau@N")
    ax.set_xlabel("Mean tau@N (TENKi)")
    ax.set_ylabel("TrueSkill2 mu")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[21] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q7 TrueSkill2 teams (Exp 21)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if args.output_dir:
        _OUT = Path(args.output_dir)
        _OUT.mkdir(parents=True, exist_ok=True)
    run(n_bootstrap=args.n_bootstrap, seed=args.seed)
