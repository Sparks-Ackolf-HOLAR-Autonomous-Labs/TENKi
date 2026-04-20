"""
Experiment 09 -- Mixed-Source Donor Flip.

Question (PEGKi Conjecture 4.2.4):
    Can a weighted mixture of low-fidelity sources flip against a stronger donor
    when no single low-fidelity source can do so alone?

Scenario
--------
For each pair (HF reference H, frugal pool F = {f1, f2, ...}):

  Single-source flip: can any single frugal source fi achieve
      tau_fi(N) > ceiling(B -> H) + eps   for some competitor B?

  Mixed-source flip:  can a weighted ensemble of frugal sources achieve this
      when no single source can?

  N*_mix = min total_N such that the ensemble surpasses the target ceiling.

Outputs
-------
results/mixed_source_flip.json
results/mixed_source_flip.png   -- tau vs N_total for single best vs best mixture
results/mixed_source_flip.md    -- summary report
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import dataclass
from itertools import combinations

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.flip_data import (
    load_many_studies,
    common_policy_subset,
    restrict_to_common_policies,
    StudyScores,
)
from analysis.flip_metrics import (
    bootstrap_tau_curve,
    full_data_ceiling,
    kendall_tau,
    rank_from_sample,
)
from analysis.flip_reports import build_meta

_OUT = Path(_HERE).parent / "results"
_OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_STUDIES = {
    "spectral": "output/db_spectral",
    "mixbox":   "output/db_mixbox",
    "km":       "output/db_km",
    "ryb":      "output/db_ryb",
    "study_a":  "output/db_study_a_artist_consensus",
    "study_b":  "output/db_study_b_physics_vs_artist",
    "study_c":  "output/db_study_c_oilpaint_vs_fooddye",
}

DEFAULT_HI_FI  = "spectral"
DEFAULT_N_VALS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
DEFAULT_EPS    = 0.01


# ---------------------------------------------------------------------------
# Mixed-source ranking
# ---------------------------------------------------------------------------

@dataclass
class MixedFlipResult:
    pool: list[str]           # frugal source names in the mixture
    reference: str            # HF study name
    competitor: str | None    # competitor being compared against (or None)
    ceiling_competitor: float # tau(competitor full -> HF full), or HF self-ceiling=1.0
    flip_n_single: int | None # N*_single: cheapest single source flip
    best_single_source: str | None
    flip_n_mix: int | None    # N*_mix: ensemble flip
    best_mix_combo: list[str] | None
    mix_beats_single: bool    # True if mix achieves a flip that no single source can


def _ensemble_rank(
    pool_studies: list[StudyScores],
    n_total: int,
    policies: list[str],
    rng: np.random.Generator,
    allocation: str = "equal",
) -> list[str]:
    """
    Draw a total of ``n_total`` experiments from the pool and return a ranked list.

    allocation = "equal" -- split n_total equally, rounding down
    """
    n_each = max(1, n_total // len(pool_studies))
    combined: dict[str, list[float]] = {p: [] for p in policies}
    for study in pool_studies:
        for pol in policies:
            if pol not in study.policy_scores:
                continue
            sample = rng.choice(
                study.policy_scores[pol],
                size=min(n_each, len(study.policy_scores[pol])),
                replace=True,
            )
            combined[pol].extend(sample.tolist())
    agg = {p: float(np.mean(vals)) for p, vals in combined.items() if vals}
    return sorted(agg, key=agg.__getitem__)


def _find_flip_n(
    tau_curve: dict[int, float],
    n_values: list[int],
    target: float,
    eps: float,
) -> int | None:
    for n in n_values:
        if tau_curve.get(n, float("nan")) > target + eps:
            return n
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    studies: dict[str, str] | None = None,
    hifi: str = DEFAULT_HI_FI,
    frugal_pool: list[str] | None = None,
    n_values: list[int] | None = None,
    n_bootstrap: int = 200,
    eps: float = DEFAULT_EPS,
    score_key: str = "best_color_distance_mean",
    seed: int = 42,
) -> None:
    study_map = studies or DEFAULT_STUDIES
    n_vals    = sorted(set(n_values or DEFAULT_N_VALS))

    print("[09] Loading study databases...")
    all_studies = load_many_studies(study_map, score_key=score_key)
    if not all_studies:
        print("[09] No studies found.")
        return

    for name, study in all_studies.items():
        print(f"  {name}: {study.n_policies} policies, {study.max_n} max experiments")

    common = common_policy_subset(all_studies)
    print(f"  Common policies: {len(common)}")
    if len(common) < 2:
        print("[09] Need at least 2 common policies -- aborting.")
        return

    studies_common = {
        name: restrict_to_common_policies(study, common)
        for name, study in all_studies.items()
    }

    if hifi not in studies_common:
        print(f"[09] HF study '{hifi}' not loaded.")
        return

    hifi_study = studies_common[hifi]
    hifi_rank  = [p for p in hifi_study.full_rank if p in common]

    _pool = frugal_pool or [s for s in studies_common if s != hifi]
    available_pool = [s for s in _pool if s in studies_common]
    if not available_pool:
        print(f"[09] No frugal pool studies found: {_pool}")
        return

    print(f"\n  HF source: {hifi}")
    print(f"  Frugal pool: {available_pool}")

    # Full-data ceilings of each frugal source vs HF
    ceilings: dict[str, float] = {}
    for name in available_pool:
        ceilings[name] = full_data_ceiling(studies_common[name], hifi_study, common)

    print("\n  Ceilings vs HF:")
    for name, c in sorted(ceilings.items(), key=lambda x: -x[1]):
        print(f"    {name}: {c:.4f}")

    # For each competitor (every frugal source), test single vs mixed flip
    rng = np.random.default_rng(seed)
    mixed_results: list[MixedFlipResult] = []
    single_curves: dict[str, dict[int, float]] = {}
    mixed_curves:  dict[tuple, dict[int, float]] = {}

    # Build single-source tau curves
    for src_name in available_pool:
        src    = studies_common[src_name]
        max_n  = src.max_n
        n_scan = [n for n in n_vals if n <= max_n]
        if max_n not in n_scan:
            n_scan.append(max_n)
        curve  = bootstrap_tau_curve(
            src, hifi_rank, common, n_scan, n_bootstrap,
            rng_seed=seed + abs(hash(src_name)) % 10000,
        )
        single_curves[src_name] = {n: stats["mean_tau"] for n, stats in curve.items()}

    # For each possible mixture size k, compute ensemble curves
    for k in range(2, len(available_pool) + 1):
        for combo in combinations(available_pool, k):
            pool_studies = [studies_common[c] for c in combo]
            max_n_pool   = min(s.max_n for s in pool_studies)
            n_scan = [n for n in n_vals if n <= max_n_pool * k]
            if not n_scan:
                n_scan = [1]

            combo_taus: dict[int, list[float]] = {n: [] for n in n_scan}
            for _ in range(n_bootstrap):
                for n_total in n_scan:
                    rank = _ensemble_rank(pool_studies, n_total, common, rng)
                    combo_taus[n_total].append(kendall_tau(rank, hifi_rank))

            mixed_curves[combo] = {
                n: float(np.nanmean(taus)) for n, taus in combo_taus.items()
            }

    # For each frugal competitor: does mixing help?
    print("\n" + "=" * 60)
    print("Mixed-source flip analysis (can a pool beat single-source?)")
    print("=" * 60)

    for comp_name in available_pool:
        ceiling_comp = ceilings[comp_name]

        # Single-source flip N*
        flip_n_single: int | None = None
        best_single: str | None   = None
        for src_name in available_pool:
            if src_name == comp_name:
                continue
            fn = _find_flip_n(single_curves[src_name], n_vals, ceiling_comp, eps)
            if fn is not None and (flip_n_single is None or fn < flip_n_single):
                flip_n_single = fn
                best_single   = src_name

        # Mixed-source flip N*
        flip_n_mix: int | None = None
        best_mix: list[str] | None = None
        for combo, curve in mixed_curves.items():
            if comp_name in combo:
                continue
            fn = _find_flip_n(curve, sorted(curve.keys()), ceiling_comp, eps)
            if fn is not None and (flip_n_mix is None or fn < flip_n_mix):
                flip_n_mix = fn
                best_mix   = list(combo)

        mix_beats = (
            flip_n_single is None and flip_n_mix is not None
        ) or (
            flip_n_single is not None and flip_n_mix is not None
            and flip_n_mix < flip_n_single
        )

        result = MixedFlipResult(
            pool=available_pool,
            reference=hifi,
            competitor=comp_name,
            ceiling_competitor=ceiling_comp,
            flip_n_single=flip_n_single,
            best_single_source=best_single,
            flip_n_mix=flip_n_mix,
            best_mix_combo=best_mix,
            mix_beats_single=mix_beats,
        )
        mixed_results.append(result)

        single_str = f"N*_single={flip_n_single} (via {best_single})" if flip_n_single else "none in range"
        mix_str    = f"N*_mix={flip_n_mix} (via {best_mix})" if flip_n_mix else "none in range"
        beats_str  = " *** MIXING HELPS ***" if mix_beats else ""
        print(f"\n  vs competitor={comp_name}  ceiling={ceiling_comp:.4f}")
        print(f"    {single_str}")
        print(f"    {mix_str}{beats_str}")

    # -- Plot: best single vs best mixture for first competitor ---------------
    if available_pool and single_curves:
        best_comp   = max(available_pool, key=lambda n: ceilings[n])
        ceiling_bc  = ceilings[best_comp]
        best_single_src = max(
            (s for s in available_pool if s != best_comp),
            key=lambda n: single_curves[n].get(max(single_curves[n]), 0),
            default=None,
        )
        best_combo_key = max(
            (c for c in mixed_curves if best_comp not in c),
            key=lambda c: mixed_curves[c].get(max(mixed_curves[c]), 0),
            default=None,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        if best_single_src:
            ns = sorted(single_curves[best_single_src].keys())
            ax.plot(ns, [single_curves[best_single_src][n] for n in ns],
                    "b-o", linewidth=2, label=f"Single best: {best_single_src}")
        if best_combo_key:
            ns = sorted(mixed_curves[best_combo_key].keys())
            ax.plot(ns, [mixed_curves[best_combo_key][n] for n in ns],
                    "g-s", linewidth=2, label=f"Best mix: {list(best_combo_key)}")
        ax.axhline(ceiling_bc, color="red", linestyle="--",
                   label=f"competitor ceiling ({best_comp}) = {ceiling_bc:.3f}")
        ax.set_xlabel("N total experiments")
        ax.set_ylabel(f"Kendall tau vs {hifi} (full)")
        ax.set_title(
            f"Mixed-source flip: single best vs best mixture\n"
            f"Target: surpass ceiling of '{best_comp}' = {ceiling_bc:.3f}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(_OUT / "mixed_source_flip.png", dpi=150)
        plt.close()
        print("\n[09] Saved mixed_source_flip.png")

    # -- JSON -----------------------------------------------------------------
    meta = build_meta(
        config=dict(hifi=hifi, n_values=n_vals, n_bootstrap=n_bootstrap,
                    flip_eps=eps, score_key=score_key, seed=seed),
        studies=list(studies_common.keys()),
        common_policies=common,
        max_n_per_study={n: studies_common[n].max_n for n in studies_common},
    )
    out_data = {
        "meta": meta,
        "ceilings": ceilings,
        "results": [
            {
                "competitor":          r.competitor,
                "ceiling_competitor":  r.ceiling_competitor,
                "flip_n_single":       r.flip_n_single,
                "best_single_source":  r.best_single_source,
                "flip_n_mix":          r.flip_n_mix,
                "best_mix_combo":      r.best_mix_combo,
                "mix_beats_single":    r.mix_beats_single,
            }
            for r in mixed_results
        ],
    }
    out_path = _OUT / "mixed_source_flip.json"
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"[09] Saved {out_path}")

    # -- Markdown report -------------------------------------------------------
    lines = [
        "# Mixed-Source Donor Flip (Experiment 09)\n",
        f"HF reference: **{hifi}**  \n",
        f"Frugal pool: {available_pool}  \n\n",
        "## Summary\n",
        "| Competitor | Ceiling | N*_single | Best single | N*_mix | Best mix | Mix helps? |",
        "|-----------|---------|----------|------------|--------|---------|-----------|",
    ]
    for r in mixed_results:
        lines.append(
            f"| {r.competitor} | {r.ceiling_competitor:.4f} | "
            f"{r.flip_n_single or '>'} | {r.best_single_source or '--'} | "
            f"{r.flip_n_mix or '>'} | {r.best_mix_combo or '--'} | "
            f"{'YES' if r.mix_beats_single else 'no'} |"
        )
    lines.append("")
    lines.append(
        "> PEGKi Conjecture 4.2.4: mixed-source solutions can be optimal under shift. "
        "Rows marked **YES** are evidence that ensemble frugality can overcome "
        "individual source limits.\n"
    )
    md_path = _OUT / "mixed_source_flip.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[09] Saved {md_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed-source donor flip (Exp 09)")
    p.add_argument("--hifi",        default=DEFAULT_HI_FI)
    p.add_argument("--pool",        nargs="+", dest="frugal_pool",
                   help="Names of frugal studies to include in the mixture pool")
    p.add_argument("--n-values",    nargs="+", type=int, default=DEFAULT_N_VALS)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--flip-eps",    type=float, default=DEFAULT_EPS)
    p.add_argument("--score-key",   default="best_color_distance_mean")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--studies",     nargs="+", metavar="NAME=PATH")
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
    run(
        studies=study_map,
        hifi=args.hifi,
        frugal_pool=args.frugal_pool,
        n_values=args.n_values,
        n_bootstrap=args.n_bootstrap,
        eps=args.flip_eps,
        score_key=args.score_key,
        seed=args.seed,
    )
