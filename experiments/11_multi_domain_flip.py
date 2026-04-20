"""
Experiment 11 -- Multi-Domain Donor-Flip Comparison.

Runs the core donor-flip analysis (Exp 02 transfer matrix + Exp 03 flip test)
across all registered TENKi domains simultaneously and produces a unified
cross-domain comparison report.

Domains (in order of data richness)
-------------------------------------
color_mixing      Primary domain.  Uses live PEGKi databases.  All experiments
                  (02-10) supported.  Required: databases exist under output/.
                  If missing, skipped gracefully.

polymer_hardness  Shore A / Shore D / Rockwell R.  Embedded data, always loads.

jarvis_leaderboard  16 ML models x 7 JARVIS property benchmarks.  Embedded data.

materials_project  GGA / HSE06 / experimental band gaps by electron-type family.
                   Embedded data.

For each domain the script computes:
  1. Directional tau matrix at N=1 (asymmetry: who is the donor?)
  2. Full-data ceiling tau for each (source, reference) pair
  3. External flip feasibility vs the domain's designated HF reference
  4. Net donor score for each engine

Cross-domain comparison
-----------------------
A summary table ranks domains by "ranking stability" -- the mean full-data
ceiling tau across all engine pairs.  This directly answers:
    "Where does my domain sit on the stable-scrambled spectrum?"
    - JARVIS leaderboard -> near-perfectly stable (tau ~= 1.0)
    - Color mixing       -> highly scrambled (tau ~= 0.3-0.7 depending on pair)
    - Polymer hardness   -> partially stable (tau ~= 0.7-0.9)
    - Materials Project  -> stable within paradigm, partially across paradigms

Outputs
-------
results/multi_domain_flip.json     -- all tau matrices and flip results per domain
results/multi_domain_flip.png      -- 2x2 grid of asymmetry heatmaps + summary bar
results/multi_domain_summary.md    -- cross-domain stability ranking table
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime, timezone

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from analysis.flip_data import common_policy_subset, restrict_to_common_policies
from analysis.flip_metrics import bootstrap_tau_curve, full_data_ceiling, kendall_tau
from analysis.flip_reports import plot_tau_matrix, build_meta

# Domain loaders -- embedded data always available; color_mixing needs live DBs
import domains.color_mixing      as _cm
import domains.polymer_hardness  as _ph
import domains.jarvis_leaderboard as _jl
import domains.materials_project  as _mp

_OUT = Path(_HERE).parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

_DOMAIN_LOADERS = {
    "color_mixing":      (_cm.load, _cm.METADATA),
    "polymer_hardness":  (_ph.load, _ph.METADATA),
    "jarvis_leaderboard": (_jl.load, _jl.METADATA),
    "materials_project": (_mp.load, _mp.METADATA),
}

N_BOOTSTRAP = 200
SEED        = 42


# ---------------------------------------------------------------------------
# Per-domain analysis
# ---------------------------------------------------------------------------

def _analyse_domain(
    domain_name: str,
    studies: dict,
    meta: dict,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Run transfer matrix + ceiling analysis for one domain.
    Returns a dict ready for JSON serialisation.
    """
    if not studies:
        return {"error": "no studies loaded"}

    common = common_policy_subset(studies)
    if len(common) < 2:
        return {"error": f"fewer than 2 common policies ({len(common)})"}

    study_names = list(studies.keys())
    hifi_name   = meta.get("hifi")
    K           = len(study_names)

    # Full-data ceiling matrix
    ceiling_mat = np.zeros((K, K))
    for i, sn in enumerate(study_names):
        for j, rn in enumerate(study_names):
            if i == j:
                ceiling_mat[i, j] = 1.0
            else:
                ceiling_mat[i, j] = full_data_ceiling(studies[sn], studies[rn], common)

    # Tau at N=1 matrix (bootstrap mean)
    rng = np.random.default_rng(seed)
    tau_n1_mat = np.zeros((K, K))
    for i, sn in enumerate(study_names):
        for j, rn in enumerate(study_names):
            if i == j:
                tau_n1_mat[i, j] = 1.0
                continue
            ref_rank = [p for p in studies[rn].full_rank if p in common]
            # For single-score external data, bootstrap is deterministic
            curve = bootstrap_tau_curve(
                studies[sn], ref_rank, common, [1], n_bootstrap,
                rng_seed=seed + i * 100 + j,
            )
            tau_n1_mat[i, j] = curve.get(1, {}).get("mean_tau", float("nan"))

    asym_mat = tau_n1_mat - tau_n1_mat.T

    # Per-engine donor score (mean asymmetry as source)
    donor_scores = {
        study_names[i]: float(np.nanmean([asym_mat[i, j] for j in range(K) if j != i]))
        for i in range(K)
    }

    # External ceiling vs HF reference
    ext_ceilings: dict[str, float] = {}
    if hifi_name and hifi_name in studies:
        for sn in study_names:
            ext_ceilings[sn] = (
                1.0 if sn == hifi_name
                else full_data_ceiling(studies[sn], studies[hifi_name], common)
            )

    # Mean ceiling (ranking stability measure for this domain)
    off_diag = [ceiling_mat[i, j] for i in range(K) for j in range(K) if i != j]
    mean_ceiling = float(np.nanmean(off_diag)) if off_diag else float("nan")

    return {
        "domain":       domain_name,
        "study_names":  study_names,
        "common_policies": common,
        "n_common":     len(common),
        "ceiling_matrix":  ceiling_mat.tolist(),
        "tau_n1_matrix":   tau_n1_mat.tolist(),
        "asymmetry_matrix": asym_mat.tolist(),
        "donor_scores":    donor_scores,
        "external_ceilings": ext_ceilings,
        "mean_ceiling":    mean_ceiling,
        "hifi":            hifi_name,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_all(domain_results: dict[str, dict]) -> None:
    """
    2xN grid: one column per domain with loaded studies.
    Row 0: ceiling matrix heatmap.
    Row 1: asymmetry matrix heatmap.
    Plus a final summary bar chart of mean_ceiling per domain.
    """
    loaded = {k: v for k, v in domain_results.items() if "error" not in v}
    if not loaded:
        return

    n_dom  = len(loaded)
    fig    = plt.figure(figsize=(max(10, 5 * n_dom), 12))
    gs     = gridspec.GridSpec(3, n_dom, figure=fig, hspace=0.55, wspace=0.45)

    cmap_ceiling = "RdYlGn"
    cmap_asym    = "RdBu"

    for col, (dom_name, res) in enumerate(loaded.items()):
        names = res["study_names"]
        K     = len(names)

        # Row 0: ceiling matrix
        ax0 = fig.add_subplot(gs[0, col])
        mat = np.array(res["ceiling_matrix"])
        im0 = ax0.imshow(mat, vmin=0, vmax=1, cmap=cmap_ceiling, aspect="auto")
        ax0.set_xticks(range(K))
        ax0.set_yticks(range(K))
        ax0.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax0.set_yticklabels(names, fontsize=7)
        ax0.set_title(f"{dom_name}\nfull-data ceiling", fontsize=8)
        for i in range(K):
            for j in range(K):
                ax0.text(j, i, f"{mat[i,j]:.2f}",
                         ha="center", va="center", fontsize=6)
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Row 1: asymmetry at N=1
        ax1 = fig.add_subplot(gs[1, col])
        amat = np.array(res["asymmetry_matrix"])
        lim  = max(0.01, float(np.abs(amat).max()))
        im1  = ax1.imshow(amat, vmin=-lim, vmax=lim, cmap=cmap_asym, aspect="auto")
        ax1.set_xticks(range(K))
        ax1.set_yticks(range(K))
        ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax1.set_yticklabels(names, fontsize=7)
        ax1.set_title("asymmetry at N=1\n(red=donor, blue=receiver)", fontsize=8)
        for i in range(K):
            for j in range(K):
                if i != j:
                    ax1.text(j, i, f"{amat[i,j]:+.2f}",
                             ha="center", va="center", fontsize=6)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Row 2: cross-domain mean ceiling bar chart (stability ranking)
    ax2 = fig.add_subplot(gs[2, :])
    dom_order = sorted(loaded, key=lambda d: loaded[d]["mean_ceiling"])
    means   = [loaded[d]["mean_ceiling"] for d in dom_order]
    colors  = [
        "salmon"    if m < 0.6 else
        "goldenrod" if m < 0.85 else
        "steelblue"
        for m in means
    ]
    ax2.barh(dom_order, means, color=colors, alpha=0.85)
    ax2.axvline(0.6,  color="salmon",    linestyle="--", linewidth=1, label="scrambled (<0.6)")
    ax2.axvline(0.85, color="steelblue", linestyle="--", linewidth=1, label="stable (>0.85)")
    ax2.set_xlabel("Mean full-data ceiling tau (all off-diagonal pairs)")
    ax2.set_title(
        "Cross-domain ranking stability\n"
        "(higher = rankings more portable across engines)"
    )
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        "TENKi multi-domain donor-flip comparison\n"
        "Color mixing (primary) . Polymer hardness . JARVIS . Materials Project",
        fontsize=10, y=1.01,
    )
    plt.savefig(_OUT / "multi_domain_flip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[11] Saved multi_domain_flip.png")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _write_markdown(domain_results: dict[str, dict]) -> None:
    loaded = {k: v for k, v in domain_results.items() if "error" not in v}

    lines = [
        "# TENKi Multi-Domain Donor-Flip Summary\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}  \n\n",
        "## Cross-domain ranking stability\n",
        "Ordered by mean full-data ceiling tau (ascending = more scrambled).\n",
        "| Domain | Engines (K) | Policies (N) | Mean ceiling tau | Stability class |",
        "|--------|------------|-------------|----------------|----------------|",
    ]
    for dom, res in sorted(loaded.items(), key=lambda x: x[1]["mean_ceiling"]):
        mc = res["mean_ceiling"]
        cls = (
            "SCRAMBLED  (tau < 0.60)" if mc < 0.60 else
            "PARTIAL    (tau 0.60-0.85)" if mc < 0.85 else
            "STABLE     (tau > 0.85)"
        )
        lines.append(
            f"| {dom} | {len(res['study_names'])} | {res['n_common']} "
            f"| {mc:.3f} | {cls} |"
        )
    lines.append("")

    # Per-domain donor ranking
    lines.append("## Net donor score per engine (mean asymmetry at N=1)\n")
    lines.append(
        "Positive = engine is a better knowledge source per experiment.\n"
        "Negative = engine needs more experiments to stabilise its ranking.\n"
    )
    for dom, res in sorted(loaded.items(), key=lambda x: x[1]["mean_ceiling"]):
        lines.append(f"### {dom}  (HF reference: {res['hifi']})\n")
        lines.append("| Engine | Donor score | Role |")
        lines.append("|--------|------------|------|")
        for eng, score in sorted(res["donor_scores"].items(), key=lambda x: -x[1]):
            role = (
                "DONOR"    if score >  0.05 else
                "RECEIVER" if score < -0.05 else
                "NEUTRAL"
            )
            lines.append(f"| {eng} | {score:+.4f} | {role} |")
        lines.append("")

    # External ceilings
    lines.append("## External ceiling (source -> HF reference, full data)\n")
    lines.append(
        "Shows the asymptotic upper bound for each engine as a "
        "donor vs the domain's high-fidelity reference.\n"
    )
    for dom, res in sorted(loaded.items(), key=lambda x: x[1]["mean_ceiling"]):
        hifi = res["hifi"]
        ec   = res.get("external_ceilings", {})
        if not ec:
            continue
        lines.append(f"### {dom}  (HF = {hifi})\n")
        lines.append("| Engine | Ceiling tau vs HF | Limit type |")
        lines.append("|--------|----------------|-----------|")
        for eng, c in sorted(ec.items(), key=lambda x: -x[1]):
            limit = (
                "SELF"            if eng == hifi else
                "NEAR_PERFECT"    if c > 0.90 else
                "PARTIAL_BIAS"    if c > 0.70 else
                "PERMANENT_GAP"
            )
            lines.append(f"| {eng} | {c:.4f} | {limit} |")
        lines.append("")

    lines.append("## Single-score vs multi-replicate domains\n")
    lines.append(
        "**Directional asymmetry** (who is the donor?) requires within-source variance:\n"
        "multiple experiments per policy so that tau_ij(N=1) != tau_ij(N=full).\n\n"
        "- **color_mixing** -- live database, many experiments -> full asymmetry analysis.\n"
        "- **polymer_hardness / jarvis_leaderboard / materials_project** -- single published\n"
        "  score per policy -> asymmetry collapses to zero (every bootstrap draw is identical).\n"
        "  Only **ceiling-level** analysis is available: which engine is structurally the\n"
        "  better proxy, regardless of how many samples are taken.\n\n"
        "To enable sampling-efficiency analysis on external domains, either:\n"
        "  1. Collect multiple experimental/simulation replicates and pass as `[s1, s2, ...]`\n"
        "     lists to `load_from_dict()`, or\n"
        "  2. Add calibrated synthetic noise: `scores[p] = [score + rng.normal(0, sigma)` "
        "for _ in range(N)]`.\n\n"
    )
    lines.append("## What this means for TENKi experiments\n")
    lines.append(
        "| Experiment | Color mixing | Polymer hardness | JARVIS | Materials Project |\n"
        "|-----------|-------------|-----------------|--------|------------------|\n"
        "| 02 Transfer matrix | Full (4+ engines) | Full (3 scales) | Full (7 props) | Full (3 fidelities) |\n"
        "| 03 Flip test | Full (live data) | Ceiling only | Ceiling only | Ceiling only |\n"
        "| 07 Convergence | Full | N/A (no replicates) | N/A | N/A |\n"
        "| 09 Mixed-source | Full | N/A | N/A | N/A |\n"
        "| 10 Aggregation MAE | **Full** (trial-level) | N/A | See note | N/A |\n\n"
        "> **Note on Exp 10 for JARVIS**: The trial-level version requires individual "
        "structure predictions, not aggregate leaderboard MAEs.  Download the JARVIS "
        "dataset and re-run `load_trial_records()` with a custom loader to enable this.\n"
    )

    md_path = _OUT / "multi_domain_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[11] Saved {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    domains: list[str] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    color_mixing_study_map: dict | None = None,
) -> None:
    requested = domains or list(_DOMAIN_LOADERS.keys())
    all_results: dict[str, dict] = {}

    print("[11] Loading domains...")
    for dom_name in requested:
        if dom_name not in _DOMAIN_LOADERS:
            print(f"  {dom_name}: UNKNOWN -- skipped")
            continue
        loader, meta = _DOMAIN_LOADERS[dom_name]
        try:
            if dom_name == "color_mixing" and color_mixing_study_map:
                studies = loader(study_map=color_mixing_study_map)
            else:
                studies = loader()
        except Exception as exc:
            print(f"  {dom_name}: LOAD ERROR -- {exc}")
            all_results[dom_name] = {"error": str(exc)}
            continue

        if not studies:
            print(f"  {dom_name}: no studies loaded (databases missing?)")
            all_results[dom_name] = {"error": "no studies loaded"}
            continue

        print(f"  {dom_name}: {len(studies)} engines loaded")
        for sn, st in studies.items():
            print(f"    {sn}: {st.n_policies} policies, max_n={st.max_n}")

        print(f"  Analysing {dom_name}...")
        result = _analyse_domain(dom_name, studies, meta, n_bootstrap, seed)
        all_results[dom_name] = result

        if "error" not in result:
            mc = result["mean_ceiling"]
            max_n = max((studies[s].max_n for s in studies), default=0)
        asym_note = "" if max_n > 1 else "  [asymmetry N/A: single-score domain]"
        print(f"  -> mean ceiling tau = {mc:.3f}  common policies = {result['n_common']}{asym_note}")

    # Print cross-domain stability ranking
    print("\n=== Cross-domain ranking stability ===")
    loaded = {k: v for k, v in all_results.items() if "error" not in v}
    if loaded:
        for dom, res in sorted(loaded.items(), key=lambda x: x[1]["mean_ceiling"]):
            mc  = res["mean_ceiling"]
            cls = "SCRAMBLED" if mc < 0.6 else "PARTIAL" if mc < 0.85 else "STABLE"
            print(f"  {dom:<22}: tau_mean={mc:.3f}  {cls}")

        print("\n=== Net donor scores (N=1) ===")
        for dom, res in sorted(loaded.items(), key=lambda x: x[1]["mean_ceiling"]):
            print(f"  {dom}:")
            for eng, score in sorted(res["donor_scores"].items(), key=lambda x: -x[1]):
                role = "DONOR" if score > 0.05 else "RECEIVER" if score < -0.05 else "neutral"
                print(f"    {eng:<20}: {score:+.4f}  {role}")

    # Outputs
    _plot_all(all_results)
    _write_markdown(all_results)

    out_data = {
        "meta": {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "n_bootstrap": n_bootstrap,
            "seed":        seed,
            "domains":     requested,
        },
        "domain_results": all_results,
    }
    out_path = _OUT / "multi_domain_flip.json"
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"[11] Saved {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-domain donor-flip comparison (Exp 11)")
    p.add_argument(
        "--domains", nargs="+",
        choices=list(_DOMAIN_LOADERS.keys()),
        default=list(_DOMAIN_LOADERS.keys()),
        help="Domains to include (default: all)",
    )
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed",        type=int, default=SEED)
    p.add_argument("--studies", nargs="+", metavar="NAME=PATH",
                   help="Color-mixing study map as name=path pairs (overrides color_mixing defaults)")
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
    run(domains=args.domains, n_bootstrap=args.n_bootstrap, seed=args.seed,
        color_mixing_study_map=study_map)
