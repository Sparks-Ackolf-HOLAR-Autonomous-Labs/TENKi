"""
Experiment 14 -- Open Questions Matrix.

Runs all six still-open questions against the expanded source pool in order,
skipping gracefully when required databases are missing.

  Phase 0  Validate source manifest and database availability
  Phase 1  Q1 mirror-pair symmetry  Q2 epsilon-symmetry threshold  Q3 Nash equilibrium
  Phase 2  Q4 quality-aware diversity allocation  Q10 swarm with true specialists
  Phase 3  Q5 per-robot info weighting  Q6 per-policy rho for MFMC  Q9 flip vs difficulty
  Phase 4  Q7 TrueSkill2 multi-team rating (requires: pip install trueskill)

Outputs
-------
  results/open_questions/open_questions_summary.md    -- consolidated human-readable report
  results/open_questions/open_questions_summary.json  -- machine-readable results per question

Usage
-----
  # Run all phases (skips missing databases):
  uv run python experiments/14_open_questions_matrix.py

  # Run specific phases only:
  uv run python experiments/14_open_questions_matrix.py --phases 0 1

  # Point at TENKi-1000 databases:
  uv run python experiments/14_open_questions_matrix.py --db-prefix output/db_1000_

  # Override individual database paths:
  uv run python experiments/14_open_questions_matrix.py --study spectral=output/db_1000_spectral
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

import numpy as np

from analysis.flip_data import (
    StudyScores,
    load_many_studies,
    common_policy_subset,
    restrict_to_common_policies,
    load_paired_data,
)
from analysis.flip_metrics import (
    bootstrap_tau_curve,
    full_data_ceiling,
    kendall_tau,
    rank_from_sample,
    spearman_rho,
)
from analysis.flip_models import external_flip_result, mutual_flip_result

_MANIFEST_PATH = Path(_PKG) / "sources_manifest.json"
_OUT_DEFAULT   = Path(_PKG) / "results" / "open_questions"


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        return {}
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


def build_study_map(
    manifest: dict,
    db_prefix: str = "",
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build name -> db_path dict from the manifest.
    db_prefix is prepended to every db_path (useful to redirect all paths at once).
    overrides replace individual entries.
    """
    overrides = overrides or {}
    result: dict[str, str] = {}
    for section in ("baseline_9", "mirror_8"):
        for entry in manifest.get(section, []):
            name = entry["name"]
            if name in overrides:
                result[name] = overrides[name]
            elif db_prefix:
                # Use the basename of db_path but with the given prefix
                result[name] = db_prefix + os.path.basename(entry["db_path"])
            else:
                result[name] = entry["db_path"]
    return result


def get_mirror_pairs(manifest: dict) -> list[tuple[str, str]]:
    return [tuple(p) for p in manifest.get("mirror_pair_definitions", {}).get("pairs", [])]


# ── Phase 0: Substrate validation ────────────────────────────────────────────

def phase0_validate(
    manifest: dict,
    study_map: dict[str, str],
    out: Path,
) -> dict:
    """
    Check which databases exist. Report missing ones with generate commands.
    Returns a dict: {name: {"exists": bool, "n_policies": int, "max_n": int, ...}}
    """
    print("\n" + "=" * 60)
    print("PHASE 0 -- Source substrate validation")
    print("=" * 60)

    _repo_root = _ROOT
    status: dict[str, dict] = {}
    all_entries = {
        e["name"]: e
        for section in ("baseline_9", "mirror_8")
        for e in manifest.get(section, [])
    }

    for name, db_path in study_map.items():
        full = os.path.join(_repo_root, db_path)
        exists = os.path.isdir(full)
        entry = all_entries.get(name, {})
        row = {
            "exists":   exists,
            "db_path":  db_path,
            "set_op":   entry.get("set_op", "unknown"),
            "ks_type":  entry.get("ks_type", "?"),
            "paired":   entry.get("paired_with_spectral", False),
            "rel_cost": entry.get("rel_cost", None),
            "is_atomic": entry.get("is_atomic_venn_region", False),
            "generate_cmd": entry.get("generate_cmd", ""),
        }
        status[name] = row

    present  = [n for n, s in status.items() if s["exists"]]
    missing  = [n for n, s in status.items() if not s["exists"]]
    baseline = [n for n in present if n in {e["name"] for e in manifest.get("baseline_9", [])}]
    mirrors  = [n for n in present if n in {e["name"] for e in manifest.get("mirror_8",  [])}]

    print(f"\n  Present  : {len(present):2d}  ({len(baseline)} baseline, {len(mirrors)} mirror)")
    print(f"  Missing  : {len(missing):2d}")

    if missing:
        print("\n  -- Missing databases (generate to unlock downstream phases) --")
        for name in missing:
            cmd = status[name]["generate_cmd"]
            print(f"  {name}")
            if cmd:
                print(f"    {cmd}")

    mirror_pairs = get_mirror_pairs(manifest)
    complete_pairs = [(a, b) for a, b in mirror_pairs
                      if status.get(a, {}).get("exists") and status.get(b, {}).get("exists")]
    incomplete_pairs = [(a, b) for a, b in mirror_pairs
                        if not (status.get(a, {}).get("exists") and status.get(b, {}).get("exists"))]

    print(f"\n  Mirror pairs complete : {len(complete_pairs)}/{len(mirror_pairs)}")
    for a, b in complete_pairs:
        print(f"    READY  : {a} <-> {b}")
    for a, b in incomplete_pairs:
        ha = "ok" if status.get(a, {}).get("exists") else "MISSING"
        hb = "ok" if status.get(b, {}).get("exists") else "MISSING"
        print(f"    BLOCKED: {a}({ha}) <-> {b}({hb})")

    return {
        "status": status,
        "present": present,
        "missing": missing,
        "baseline_present": baseline,
        "mirrors_present": mirrors,
        "complete_mirror_pairs": complete_pairs,
        "incomplete_mirror_pairs": incomplete_pairs,
    }


# ── Phase 1 helpers ───────────────────────────────────────────────────────────

def _tau_and_ci(
    source: StudyScores,
    reference: StudyScores,
    policies: list[str],
    n: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Return (mean_tau, std_tau) at N experiments."""
    curve = bootstrap_tau_curve(
        source, reference.full_rank, policies, [n], n_bootstrap, rng_seed=seed
    )
    stats = curve.get(n, {})
    return stats.get("mean_tau", float("nan")), stats.get("std_tau", 0.0)


def q1_mirror_pairs(
    all_studies: dict[str, StudyScores],
    mirror_pairs: list[tuple[str, str]],
    hifi: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Q1: For each mirror pair (G_A\G_B, G_B\G_A), compare tau@N=1, tau@N=10,
    ceiling, bias floor, donor score, and N*.
    Decision: symmetric if bootstrap CIs overlap at N=10.
    """
    results = []
    available = {a: b for a, b in mirror_pairs
                 if a in all_studies and b in all_studies}

    if not available:
        print("  [Q1] No complete mirror pairs available -- skipping.")
        return {"skipped": True, "reason": "no complete mirror pairs found", "pairs": []}

    hifi_study = all_studies.get(hifi)
    common_all = common_policy_subset(all_studies)

    for fwd, rev in available.items():
        s_fwd = restrict_to_common_policies(all_studies[fwd], common_all)
        s_rev = restrict_to_common_policies(all_studies[rev], common_all)

        tau1_fwd, std1_fwd = _tau_and_ci(s_fwd, s_rev, common_all, 1, n_bootstrap,
                                          seed + abs(hash(fwd)) % 9999)
        tau1_rev, std1_rev = _tau_and_ci(s_rev, s_fwd, common_all, 1, n_bootstrap,
                                          seed + abs(hash(rev)) % 9999)
        tau10_fwd, std10_fwd = _tau_and_ci(s_fwd, s_rev, common_all, 10, n_bootstrap,
                                            seed + abs(hash(fwd + "10")) % 9999)
        tau10_rev, std10_rev = _tau_and_ci(s_rev, s_fwd, common_all, 10, n_bootstrap,
                                            seed + abs(hash(rev + "10")) % 9999)

        # Ceilings vs HF if available
        ceiling_fwd = full_data_ceiling(s_fwd, hifi_study, common_all) if hifi_study else float("nan")
        ceiling_rev = full_data_ceiling(s_rev, hifi_study, common_all) if hifi_study else float("nan")

        # CI overlap at N=10 (1-sigma intervals)
        lo_fwd = tau10_fwd - std10_fwd
        hi_fwd = tau10_fwd + std10_fwd
        lo_rev = tau10_rev - std10_rev
        hi_rev = tau10_rev + std10_rev
        ci_overlap = not (hi_fwd < lo_rev or hi_rev < lo_fwd)

        verdict = "SYMMETRIC" if ci_overlap else "ASYMMETRIC"

        row = {
            "forward": fwd,
            "reverse": rev,
            "tau@N=1_fwd":  round(tau1_fwd, 4),
            "tau@N=1_rev":  round(tau1_rev, 4),
            "tau@N=10_fwd": round(tau10_fwd, 4),
            "tau@N=10_rev": round(tau10_rev, 4),
            "std@N=10_fwd": round(std10_fwd, 4),
            "std@N=10_rev": round(std10_rev, 4),
            "ceiling_fwd":  round(ceiling_fwd, 4),
            "ceiling_rev":  round(ceiling_rev, 4),
            "ci_overlap":   ci_overlap,
            "verdict":      verdict,
        }
        results.append(row)
        print(f"  [{fwd} <-> {rev}] tau10=({tau10_fwd:.3f},{tau10_rev:.3f})  "
              f"ceil=({ceiling_fwd:.3f},{ceiling_rev:.3f})  {verdict}")

    n_sym  = sum(1 for r in results if r["verdict"] == "SYMMETRIC")
    n_asym = len(results) - n_sym
    conclusion = (
        "Mirror symmetry SUPPORTED" if n_sym == len(results) and results
        else f"Mirror symmetry PARTIAL ({n_sym}/{len(results)} symmetric)"
        if n_sym > 0 else "Mirror symmetry NOT SUPPORTED"
    )
    print(f"  Q1 conclusion: {conclusion}")
    return {"pairs": results, "conclusion": conclusion}


def q2_epsilon_symmetry(
    all_studies: dict[str, StudyScores],
    manifest: dict,
    out: Path,
) -> dict:
    """
    Q2: Compute epsilon_symmetry for the available source pool.
    epsilon = max over all mirror-able pairs of |vol_fwd - vol_rev| / (vol_fwd + vol_rev).
    Report minimum k to achieve eps <= 0.25, 0.10, 0.05.
    """
    import glob

    print("\n  [Q2] Computing epsilon-symmetry thresholds...")

    _repo_root = _ROOT
    all_entries = {
        e["name"]: e
        for section in ("baseline_9", "mirror_8")
        for e in manifest.get(section, [])
    }

    def count_targets(name: str) -> int:
        entry = all_entries.get(name, {})
        db_path = entry.get("db_path", "")
        db = Path(_repo_root) / db_path
        if not db.exists():
            return 0
        n = 0
        for p in sorted(glob.glob(str(db / "targets" / "*.json")))[:50]:
            try:
                with open(p) as f:
                    d = json.load(f)
                targets = d.get("targets", d if isinstance(d, list) else [])
                n += len(targets)
            except Exception:
                pass
        return n

    mirror_pairs = get_mirror_pairs(manifest)
    available_pairs = [(a, b) for a, b in mirror_pairs
                       if a in all_studies and b in all_studies]

    if not available_pairs:
        return {"skipped": True, "reason": "no mirror pairs available"}

    epsilons = []
    details = []
    for fwd, rev in available_pairs:
        n_fwd = count_targets(fwd)
        n_rev = count_targets(rev)
        if n_fwd + n_rev == 0:
            continue
        eps = abs(n_fwd - n_rev) / (n_fwd + n_rev)
        epsilons.append(eps)
        details.append({"pair": f"{fwd}|{rev}", "n_fwd": n_fwd, "n_rev": n_rev, "epsilon": round(eps, 4)})
        print(f"    {fwd} ({n_fwd} targets) <-> {rev} ({n_rev} targets)  eps={eps:.4f}")

    if not epsilons:
        return {"skipped": True, "reason": "could not count targets in mirror DBs"}

    eps_max = max(epsilons)
    thresholds = {}
    for thresh in [0.25, 0.10, 0.05]:
        passing = [d for d in details if d["epsilon"] <= thresh]
        thresholds[thresh] = {"n_pairs": len(passing), "fraction": len(passing) / len(details)}

    print(f"  Q2 max epsilon = {eps_max:.4f}")
    for t, v in thresholds.items():
        print(f"    eps <= {t}: {v['n_pairs']}/{len(details)} pairs pass")

    return {"epsilon_max": round(eps_max, 4), "pair_details": details, "thresholds": thresholds}


def q3_nash_equilibrium(
    all_studies: dict[str, StudyScores],
    hifi: str,
    n_total: int,
    n_bootstrap: int,
    seed: int,
    max_pool_size: int = 12,
) -> dict:
    """
    Q3: Fixed-budget source-selection Nash equilibrium.
    For each non-empty subset S of non-HF sources (up to max_pool_size sources),
    allocate N_total/|S| experiments per source and compute tau vs HF ranking.
    Report which subset maximises tau (Nash-stable best response at equal allocation).
    """
    print("\n  [Q3] Nash equilibrium under equal allocation...")

    hifi_study = all_studies.get(hifi)
    if not hifi_study:
        return {"skipped": True, "reason": f"HF source '{hifi}' not loaded"}

    frugal = [n for n in all_studies if n != hifi]
    common = common_policy_subset(all_studies)
    if not common or len(common) < 2:
        return {"skipped": True, "reason": "fewer than 2 common policies"}

    # Restrict to common policies
    studies_c = {n: restrict_to_common_policies(all_studies[n], common) for n in all_studies}
    hifi_c    = studies_c[hifi]
    hifi_rank = [p for p in hifi_c.full_rank if p in common]

    rng = np.random.default_rng(seed)
    results: list[dict] = []

    pool = frugal[:max_pool_size]

    for k in range(1, len(pool) + 1):
        for subset in combinations(pool, k):
            n_per_source = max(1, n_total // k)
            taus = []
            for _ in range(n_bootstrap):
                # Equal allocation: draw n_per_source from each source in subset
                combined_scores: dict[str, list[float]] = {p: [] for p in common}
                for src_name in subset:
                    src = studies_c[src_name]
                    for p in common:
                        if p in src.policy_scores:
                            scores = src.policy_scores[p]
                            draw = rng.choice(scores, size=min(n_per_source, len(scores)), replace=True)
                            combined_scores[p].extend(draw.tolist())
                combined_rank = sorted(
                    [p for p in common if combined_scores[p]],
                    key=lambda p: float(np.mean(combined_scores[p])),
                )
                taus.append(kendall_tau(combined_rank, hifi_rank))

            valid = [t for t in taus if not np.isnan(t)]
            mean_tau = float(np.mean(valid)) if valid else float("nan")
            results.append({
                "k": k,
                "sources": list(subset),
                "n_per_source": n_per_source,
                "tau_mean": round(mean_tau, 4),
            })

    results.sort(key=lambda r: -r["tau_mean"])
    best = results[0] if results else {}

    # Concentration check: does the best 1-source solution beat all multi-source solutions?
    best_k1  = max((r for r in results if r["k"] == 1), key=lambda r: r["tau_mean"], default=None)
    best_all = results[0] if results else None

    concentration = (best_k1 and best_all and best_k1["tau_mean"] >= best_all["tau_mean"] - 0.01)

    print(f"  Q3 best allocation: {best.get('sources')} k={best.get('k')}  tau={best.get('tau_mean', 'n/a')}")
    print(f"  Concentration holds: {concentration}")

    return {
        "best": best,
        "top_10": results[:10],
        "concentration_holds": concentration,
        "n_total": n_total,
        "note": "Equal allocation only. Rho-proportional requires set-op sources for spatial variation.",
    }


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def q4_diversity_allocation(
    all_studies: dict[str, StudyScores],
    hifi: str,
    n_total: int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Q4: Compare allocation modes at fixed N_total.
    Modes: equal, global_rho, inverse_variance, oracle_upper_bound.
    """
    print("\n  [Q4] Quality-aware diversity allocation...")

    hifi_study = all_studies.get(hifi)
    if not hifi_study:
        return {"skipped": True, "reason": f"HF source '{hifi}' not loaded"}

    frugal = [n for n in all_studies if n != hifi]
    if len(frugal) < 2:
        return {"skipped": True, "reason": "fewer than 2 frugal sources"}

    common = common_policy_subset(all_studies)
    if len(common) < 2:
        return {"skipped": True, "reason": "fewer than 2 common policies"}

    studies_c = {n: restrict_to_common_policies(all_studies[n], common) for n in all_studies}
    hifi_c    = studies_c[hifi]
    hifi_rank = [p for p in hifi_c.full_rank if p in common]
    rng = np.random.default_rng(seed)

    # Compute global rho for each source (Spearman rho of full-data ranking vs HF)
    global_rhos = {}
    for name in frugal:
        s = studies_c[name]
        full_r = [p for p in s.full_rank if p in common]
        global_rhos[name] = spearman_rho(full_r, hifi_rank)

    # Compute bootstrap variance at N=1 for each source
    var_n1 = {}
    for name in frugal:
        s = studies_c[name]
        curve = bootstrap_tau_curve(s, hifi_rank, common, [1], min(50, n_bootstrap),
                                    rng_seed=seed + abs(hash(name)) % 9999)
        var_n1[name] = curve.get(1, {}).get("std_tau", 1.0) ** 2

    def make_weights(mode: str) -> dict[str, float]:
        if mode == "equal":
            return {n: 1.0 / len(frugal) for n in frugal}
        elif mode == "global_rho":
            rhos = {n: max(0.01, global_rhos.get(n, 0.5)) for n in frugal}
            total = sum(rhos.values())
            return {n: v / total for n, v in rhos.items()}
        elif mode == "inverse_variance":
            inv = {n: 1.0 / max(1e-6, var_n1.get(n, 1.0)) for n in frugal}
            total = sum(inv.values())
            return {n: v / total for n, v in inv.items()}
        elif mode == "oracle_upper_bound":
            # Oracle: give all budget to the single source with highest full-data ceiling
            best = max(frugal, key=lambda n: global_rhos.get(n, 0))
            return {n: (1.0 if n == best else 0.0) for n in frugal}
        return {n: 1.0 / len(frugal) for n in frugal}

    mode_results = {}
    for mode in ["equal", "global_rho", "inverse_variance", "oracle_upper_bound"]:
        weights = make_weights(mode)
        taus = []
        for _ in range(n_bootstrap):
            combined_scores: dict[str, list[float]] = {p: [] for p in common}
            for src_name in frugal:
                w = weights[src_name]
                if w < 1e-9:
                    continue
                n_from = max(1, int(round(n_total * w)))
                src = studies_c[src_name]
                for p in common:
                    if p in src.policy_scores:
                        scores = src.policy_scores[p]
                        draw = rng.choice(scores, size=min(n_from, len(scores)), replace=True)
                        combined_scores[p].extend(draw.tolist())
            combined_rank = sorted(
                [p for p in common if combined_scores[p]],
                key=lambda p: float(np.mean(combined_scores[p])),
            )
            taus.append(kendall_tau(combined_rank, hifi_rank))
        valid = [t for t in taus if not np.isnan(t)]
        mean_tau = float(np.mean(valid)) if valid else float("nan")
        mode_results[mode] = {"tau_mean": round(mean_tau, 4), "weights": {n: round(w, 4) for n, w in weights.items()}}
        print(f"    {mode:25s} tau={mean_tau:.4f}  weights={dict(list(weights.items())[:3])}")

    oracle_tau = mode_results.get("oracle_upper_bound", {}).get("tau_mean", float("nan"))
    equal_tau  = mode_results.get("equal", {}).get("tau_mean", float("nan"))
    quality_aware_helps = (
        mode_results.get("inverse_variance", {}).get("tau_mean", 0)
        > equal_tau + 0.01
        or mode_results.get("global_rho", {}).get("tau_mean", 0)
        > equal_tau + 0.01
    )
    print(f"  Q4 quality-aware improves on equal: {quality_aware_helps}")

    return {
        "modes": mode_results,
        "n_total": n_total,
        "global_rhos": {n: round(v, 4) for n, v in global_rhos.items()},
        "quality_aware_helps": quality_aware_helps,
        "note": "Quality-aware advantage is expected only when sources are spatially heterogeneous.",
    }


def q10_swarm_specialists(
    all_studies: dict[str, StudyScores],
    hifi: str,
    manifest: dict,
    n_values: list[int],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Q10: Swarm vs ensemble on three pools:
      - 4 generalist engines only (spectral, mixbox, km, ryb)
      - all available non-HF sources
      - set-op sources only (is_atomic_venn_region = True)
    """
    print("\n  [Q10] Swarm advantage with specialist pools...")

    hifi_study = all_studies.get(hifi)
    if not hifi_study:
        return {"skipped": True, "reason": f"HF source '{hifi}' not loaded"}

    common = common_policy_subset(all_studies)
    if len(common) < 2:
        return {"skipped": True, "reason": "fewer than 2 common policies"}

    studies_c = {n: restrict_to_common_policies(all_studies[n], common) for n in all_studies}
    hifi_c    = studies_c[hifi]
    hifi_rank = [p for p in hifi_c.full_rank if p in common]

    all_entries = {
        e["name"]: e
        for section in ("baseline_9", "mirror_8")
        for e in manifest.get(section, [])
    }

    generalists = [n for n in ["mixbox", "km", "ryb"] if n in all_studies and n != hifi]
    set_op_names = [n for n, e in all_entries.items() if e.get("is_atomic_venn_region") and n in all_studies]
    all_frugal  = [n for n in all_studies if n != hifi]

    pools = {
        "generalists_only": generalists,
        "all_sources":      all_frugal,
        "specialists_only": set_op_names,
    }

    rng = np.random.default_rng(seed)
    results = {}
    for pool_name, pool in pools.items():
        if len(pool) < 2:
            results[pool_name] = {"skipped": True, "reason": f"pool has {len(pool)} sources"}
            continue
        pool_studies = {n: studies_c[n] for n in pool}

        pool_results = {}
        for n in n_values:
            # Ensemble: equal weight
            ens_taus, swarm_taus = [], []
            for _ in range(n_bootstrap):
                combined_ens: dict[str, list[float]]   = {p: [] for p in common}
                combined_swarm: dict[str, list[float]] = {p: [] for p in common}

                n_per = max(1, n // len(pool))
                source_samples: dict[str, dict[str, np.ndarray]] = {}
                for src_name, src in pool_studies.items():
                    source_samples[src_name] = {}
                    for p in common:
                        if p in src.policy_scores:
                            source_samples[src_name][p] = rng.choice(
                                src.policy_scores[p], size=min(n_per, len(src.policy_scores[p])), replace=True
                            )

                # Ensemble scores
                for src_name in pool:
                    for p in common:
                        if p in source_samples.get(src_name, {}):
                            combined_ens[p].extend(source_samples[src_name][p].tolist())

                # Swarm/local-router: weight source by inverse mean error (proxy: 1/std of its sample)
                source_quality: dict[str, float] = {}
                for src_name in pool:
                    stds = [float(np.std(source_samples[src_name].get(p, [0])))
                            for p in common if p in source_samples.get(src_name, {})]
                    source_quality[src_name] = 1.0 / (float(np.mean(stds)) + 1e-6)
                total_q = sum(source_quality.values())
                for src_name in pool:
                    w = source_quality[src_name] / total_q
                    n_swarm = max(1, int(round(n * w)))
                    src = pool_studies[src_name]
                    for p in common:
                        if p in src.policy_scores:
                            draw = rng.choice(src.policy_scores[p], size=min(n_swarm, len(src.policy_scores[p])), replace=True)
                            combined_swarm[p].extend(draw.tolist())

                rank_ens   = sorted([p for p in common if combined_ens[p]],   key=lambda p: float(np.mean(combined_ens[p])))
                rank_swarm = sorted([p for p in common if combined_swarm[p]], key=lambda p: float(np.mean(combined_swarm[p])))
                ens_taus.append(kendall_tau(rank_ens, hifi_rank))
                swarm_taus.append(kendall_tau(rank_swarm, hifi_rank))

            valid_e = [t for t in ens_taus   if not np.isnan(t)]
            valid_s = [t for t in swarm_taus if not np.isnan(t)]
            pool_results[n] = {
                "ensemble_tau": round(float(np.mean(valid_e)), 4) if valid_e else float("nan"),
                "swarm_tau":    round(float(np.mean(valid_s)), 4) if valid_s else float("nan"),
            }

        swarm_beats = any(
            pool_results[n].get("swarm_tau", 0) > pool_results[n].get("ensemble_tau", 0) + 0.01
            for n in n_values
        )
        print(f"    {pool_name:20s}  K={len(pool)}  swarm_beats_ensemble={swarm_beats}")
        results[pool_name] = {
            "sources": pool,
            "n_results": pool_results,
            "swarm_beats_ensemble": swarm_beats,
        }

    return results


# ── Phase 3 helpers ───────────────────────────────────────────────────────────

def q5_per_robot_difficulty(
    all_studies: dict[str, StudyScores],
    hifi: str,
    n_values: list[int],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Q5: Does target difficulty affect per-robot transferable information?
    Difficulty = best achievable HF score per experiment.
    Split experiments into easy/medium/hard thirds; compute tau@N curve per bin.
    """
    print("\n  [Q5] Per-robot information weighting by target difficulty...")

    hifi_study = all_studies.get(hifi)
    if not hifi_study:
        return {"skipped": True, "reason": f"HF source '{hifi}' not loaded"}

    frugal = [n for n in all_studies if n != hifi]
    if not frugal:
        return {"skipped": True, "reason": "no frugal sources"}

    common = common_policy_subset(all_studies)
    if len(common) < 2:
        return {"skipped": True, "reason": "fewer than 2 common policies"}

    studies_c = {n: restrict_to_common_policies(all_studies[n], common) for n in all_studies}
    hifi_c = studies_c[hifi]

    # Build difficulty score per experiment as the best-policy HF score.
    # Each experiment has one score per policy; use the MINIMUM across policies (best achievable).
    best_policy_scores = hifi_c.policy_scores  # policy -> [exp_score_0, exp_score_1, ...]
    n_exps = hifi_c.max_n
    per_exp_difficulty = []
    for exp_idx in range(n_exps):
        scores = [best_policy_scores[p][exp_idx] for p in common if exp_idx < len(best_policy_scores[p])]
        if scores:
            per_exp_difficulty.append(float(np.min(scores)))

    if not per_exp_difficulty:
        return {"skipped": True, "reason": "no experiment scores in HF study"}

    difficulty_arr = np.array(per_exp_difficulty)
    q33, q67 = np.quantile(difficulty_arr, [0.33, 0.67])
    bins = {
        "easy":   [i for i, d in enumerate(per_exp_difficulty) if d <= q33],
        "medium": [i for i, d in enumerate(per_exp_difficulty) if q33 < d <= q67],
        "hard":   [i for i, d in enumerate(per_exp_difficulty) if d > q67],
    }
    print(f"    Difficulty bins: easy={len(bins['easy'])} medium={len(bins['medium'])} hard={len(bins['hard'])}")

    rng = np.random.default_rng(seed)
    source_results = {}
    for src_name in frugal[:3]:  # limit to 3 sources for speed
        src = studies_c[src_name]
        bin_results = {}
        for bin_name, exp_idxs in bins.items():
            if len(exp_idxs) < 3:
                continue
            # Build a restricted StudyScores using only experiments in this bin
            bin_scores = {
                p: [src.policy_scores[p][i] for i in exp_idxs if i < len(src.policy_scores.get(p, []))]
                for p in common if p in src.policy_scores
            }
            bin_scores = {p: v for p, v in bin_scores.items() if v}
            if not bin_scores:
                continue
            bin_rank_hifi = sorted(
                [p for p in common if p in hifi_c.policy_scores],
                key=lambda p: float(np.mean([hifi_c.policy_scores[p][i]
                                             for i in exp_idxs if i < len(hifi_c.policy_scores.get(p, []))]
                                            or [float("inf")])),
            )

            bin_study = StudyScores(
                name=f"{src_name}_{bin_name}",
                db_path=src.db_path,
                policy_scores=bin_scores,
                full_rank=sorted(bin_scores, key=lambda p: float(np.mean(bin_scores[p]))),
                n_policies=len(bin_scores),
                max_n=min(len(v) for v in bin_scores.values()),
            )
            n_scan = [n for n in n_values if n <= bin_study.max_n] or [1]
            curve = bootstrap_tau_curve(bin_study, bin_rank_hifi, list(bin_scores.keys()),
                                        n_scan, n_bootstrap, rng_seed=seed + abs(hash(src_name + bin_name)) % 9999)
            bin_results[bin_name] = {
                str(n): round(curve.get(n, {}).get("mean_tau", float("nan")), 4) for n in n_scan
            }
        source_results[src_name] = bin_results
        if bin_results:
            easy1  = bin_results.get("easy",   {}).get("1", float("nan"))
            hard1  = bin_results.get("hard",   {}).get("1", float("nan"))
            print(f"    {src_name:20s}  tau@N=1  easy={easy1:.3f}  hard={hard1:.3f}")

    return {"sources": source_results, "difficulty_quantiles": {"q33": round(q33, 4), "q67": round(q67, 4)}}


def q6_per_policy_rho(
    all_studies: dict[str, StudyScores],
    hifi: str,
    paired_sources: list[str],
) -> dict:
    """
    Q6: Compute Spearman rho per policy for each paired LF source.
    Uses the full-data policy_scores (one score per experiment) to estimate
    how well each policy's LF scores correlate with HF scores across targets.
    """
    print("\n  [Q6] Per-policy rho for MFMC...")

    hifi_study = all_studies.get(hifi)
    if not hifi_study:
        return {"skipped": True, "reason": f"HF source '{hifi}' not loaded"}

    available_paired = [n for n in paired_sources if n in all_studies]
    if not available_paired:
        return {"skipped": True, "reason": "no paired sources found"}

    common = common_policy_subset({hifi: all_studies[hifi], **{n: all_studies[n] for n in available_paired}})
    results = {}

    for src_name in available_paired:
        src = all_studies[src_name]
        policy_rhos = {}
        for p in common:
            hf_scores = hifi_study.policy_scores.get(p, [])
            lf_scores = src.policy_scores.get(p, [])
            n = min(len(hf_scores), len(lf_scores))
            if n < 3:
                continue
            from scipy.stats import spearmanr
            rho_val, _ = spearmanr(hf_scores[:n], lf_scores[:n])
            policy_rhos[p] = round(float(rho_val), 4)

        if not policy_rhos:
            continue
        mean_rho = round(float(np.mean(list(policy_rhos.values()))), 4)
        std_rho  = round(float(np.std(list(policy_rhos.values()))), 4)
        robust   = sorted([p for p, r in policy_rhos.items() if r >= mean_rho + 0.05])
        fragile  = sorted([p for p, r in policy_rhos.items() if r <= mean_rho - 0.05])
        print(f"    {src_name:25s}  mean_rho={mean_rho:.3f}  std={std_rho:.3f}  "
              f"robust={len(robust)}  fragile={len(fragile)}")
        results[src_name] = {
            "per_policy_rho": policy_rhos,
            "mean_rho":   mean_rho,
            "std_rho":    std_rho,
            "robust_policies":  robust,
            "fragile_policies": fragile,
        }

    return results


def q9_flip_by_difficulty(
    all_studies: dict[str, StudyScores],
    hifi: str,
    n_values: list[int],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Q9: Compute tau@N and N* separately for easy/medium/hard target bins.
    """
    print("\n  [Q9] Flip N* sensitivity to target difficulty...")

    hifi_study = all_studies.get(hifi)
    if not hifi_study:
        return {"skipped": True, "reason": f"HF source '{hifi}' not loaded"}

    frugal = [n for n in all_studies if n != hifi]
    if not frugal:
        return {"skipped": True, "reason": "no frugal sources"}

    common = common_policy_subset(all_studies)
    if len(common) < 2:
        return {"skipped": True, "reason": "fewer than 2 common policies"}

    studies_c = {n: restrict_to_common_policies(all_studies[n], common) for n in all_studies}
    hifi_c = studies_c[hifi]
    n_exps = hifi_c.max_n

    per_exp_diff = []
    for exp_idx in range(n_exps):
        scores = [hifi_c.policy_scores[p][exp_idx] for p in common
                  if exp_idx < len(hifi_c.policy_scores.get(p, []))]
        if scores:
            per_exp_diff.append(float(np.min(scores)))

    if not per_exp_diff:
        return {"skipped": True, "reason": "no experiment scores"}

    difficulty_arr = np.array(per_exp_diff)
    q33, q67 = np.quantile(difficulty_arr, [0.33, 0.67])
    bins = {
        "easy":   [i for i, d in enumerate(per_exp_diff) if d <= q33],
        "medium": [i for i, d in enumerate(per_exp_diff) if q33 < d <= q67],
        "hard":   [i for i, d in enumerate(per_exp_diff) if d > q67],
    }

    results = {}
    for src_name in frugal[:4]:
        src = studies_c[src_name]
        src_bin_results = {}
        for bin_name, exp_idxs in bins.items():
            if len(exp_idxs) < 5:
                continue
            bin_hf_scores = {
                p: [hifi_c.policy_scores[p][i] for i in exp_idxs if i < len(hifi_c.policy_scores.get(p, []))]
                for p in common if p in hifi_c.policy_scores
            }
            bin_hf_rank = sorted([p for p in common if bin_hf_scores.get(p)],
                                  key=lambda p: float(np.mean(bin_hf_scores[p])))

            bin_lf_scores = {
                p: [src.policy_scores[p][i] for i in exp_idxs if i < len(src.policy_scores.get(p, []))]
                for p in common if p in src.policy_scores
            }
            bin_lf_scores = {p: v for p, v in bin_lf_scores.items() if v}
            if not bin_lf_scores:
                continue

            bin_study = StudyScores(
                name=f"{src_name}_{bin_name}",
                db_path=src.db_path,
                policy_scores=bin_lf_scores,
                full_rank=sorted(bin_lf_scores, key=lambda p: float(np.mean(bin_lf_scores[p]))),
                n_policies=len(bin_lf_scores),
                max_n=min(len(v) for v in bin_lf_scores.values()),
            )
            n_scan = [n for n in n_values if n <= bin_study.max_n] or [1]
            curve = bootstrap_tau_curve(bin_study, bin_hf_rank, list(bin_lf_scores.keys()),
                                        n_scan, n_bootstrap,
                                        rng_seed=seed + abs(hash(src_name + bin_name)) % 9999)
            tau1  = curve.get(min(n_scan), {}).get("mean_tau", float("nan"))
            tau_max = curve.get(max(n_scan), {}).get("mean_tau", float("nan"))
            src_bin_results[bin_name] = {
                "tau@N=1": round(tau1, 4),
                f"tau@N={max(n_scan)}": round(tau_max, 4),
            }
        results[src_name] = src_bin_results
        if src_bin_results:
            easy1 = src_bin_results.get("easy",  {}).get("tau@N=1", float("nan"))
            hard1 = src_bin_results.get("hard",  {}).get("tau@N=1", float("nan"))
            print(f"    {src_name:20s}  tau@N=1 easy={easy1:.3f}  hard={hard1:.3f}")

    return {"sources": results, "difficulty_quantiles": {"q33": round(q33, 4), "q67": round(q67, 4)}}


# ── Phase 4 helper ────────────────────────────────────────────────────────────

def q7_trueskill2_teams(
    all_studies: dict[str, StudyScores],
    hifi: str,
    n_values: list[int],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """
    Q7: Thin TrueSkill2 adapter.
    Teams: (source, n_robots).  Match outcome: tau vs spectral.
    Check if TrueSkill2 mu recovers the same source taxonomy as TENKi.
    """
    print("\n  [Q7] TrueSkill2 multi-team rating...")

    try:
        import trueskill
    except ImportError:
        print("    trueskill package not installed -- skipping Q7.")
        print("    Install with: pip install trueskill")
        return {"skipped": True, "reason": "trueskill package not installed"}

    hifi_study = all_studies.get(hifi)
    if not hifi_study:
        return {"skipped": True, "reason": f"HF source '{hifi}' not loaded"}

    frugal = [n for n in all_studies if n != hifi]
    if not frugal:
        return {"skipped": True, "reason": "no frugal sources"}

    common = common_policy_subset(all_studies)
    if len(common) < 2:
        return {"skipped": True, "reason": "fewer than 2 common policies"}

    studies_c = {n: restrict_to_common_policies(all_studies[n], common) for n in all_studies}
    hifi_c    = studies_c[hifi]
    hifi_rank = [p for p in hifi_c.full_rank if p in common]

    rng = np.random.default_rng(seed)

    # Build a table of (source, n) -> tau@n via bootstrap
    team_taus: dict[tuple[str, int], list[float]] = {}
    for src_name in frugal:
        src = studies_c[src_name]
        for n in n_values:
            if n > src.max_n:
                continue
            taus = []
            for _ in range(n_bootstrap):
                ranked = rank_from_sample(src, common, n, rng)
                taus.append(kendall_tau(ranked, hifi_rank))
            team_taus[(src_name, n)] = taus

    # Run TrueSkill2-style matches: for each pair of teams, compare tau
    env = trueskill.TrueSkill(backend="scipy")
    team_ratings: dict[tuple[str, int], trueskill.Rating] = {
        team: env.create_rating() for team in team_taus
    }

    # Simulate matches: sort all teams by observed tau; feed win/loss pairs top-down
    sorted_teams = sorted(team_taus.keys(), key=lambda t: float(np.mean(team_taus[t])), reverse=True)

    n_matches = 0
    for i in range(len(sorted_teams)):
        for j in range(i + 1, min(i + 4, len(sorted_teams))):
            t_a, t_b = sorted_teams[i], sorted_teams[j]
            tau_a = float(np.mean(team_taus[t_a]))
            tau_b = float(np.mean(team_taus[t_b]))
            try:
                if tau_a > tau_b + 0.01:
                    (team_ratings[t_a],), (team_ratings[t_b],) = env.rate(
                        [[team_ratings[t_a]], [team_ratings[t_b]]], ranks=[0, 1]
                    )
                elif tau_b > tau_a + 0.01:
                    (team_ratings[t_b],), (team_ratings[t_a],) = env.rate(
                        [[team_ratings[t_b]], [team_ratings[t_a]]], ranks=[0, 1]
                    )
                n_matches += 1
            except Exception:
                pass

    # Aggregate per source: mean TrueSkill mu across n_values
    source_mu: dict[str, list[float]] = {}
    for (src, n), rating in team_ratings.items():
        source_mu.setdefault(src, []).append(rating.mu)

    source_summary = {
        src: round(float(np.mean(mus)), 2) for src, mus in source_mu.items()
    }
    ts_ranking = sorted(source_summary, key=source_summary.__getitem__, reverse=True)

    # TENKi canonical ranking (donor score order)
    tenki_donor_order = ["spectral", "study_b", "mixbox", "ryb",
                          "study_c_reverse", "study_b_reverse", "km", "study_a", "study_c"]
    tenki_rank_subset = [n for n in tenki_donor_order if n in ts_ranking]
    ts_rank_subset    = [n for n in ts_ranking if n in set(tenki_rank_subset)]

    tau_to_tenki = kendall_tau(ts_rank_subset, tenki_rank_subset)
    useful = tau_to_tenki > 0.6

    print(f"    Matches run: {n_matches}  tau(TrueSkill2 vs TENKi taxonomy)={tau_to_tenki:.3f}")
    print(f"    TrueSkill2 useful: {useful}")
    for src in ts_ranking:
        print(f"    {src:25s}  mu={source_summary[src]:.2f}")

    return {
        "source_mu": source_summary,
        "ts_ranking": ts_ranking,
        "tau_to_tenki_taxonomy": round(tau_to_tenki, 4),
        "ts2_useful": useful,
        "n_matches": n_matches,
    }


# ── Report writer ─────────────────────────────────────────────────────────────

def write_summary(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "open_questions_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[14] Saved {json_path}")

    # Markdown
    md_path = out / "open_questions_summary.md"
    lines = [
        "# TENKi Open Questions — Experiment 14 Summary\n",
        f"Generated: {summary.get('generated', 'unknown')}\n\n",
    ]

    p0 = summary.get("phase0", {})
    if p0:
        present  = p0.get("present",  [])
        missing  = p0.get("missing",  [])
        cp       = p0.get("complete_mirror_pairs", [])
        ip       = p0.get("incomplete_mirror_pairs", [])
        lines += [
            "## Phase 0 — Substrate Status\n\n",
            f"- Databases present: {len(present)} ({', '.join(present[:6])}{'...' if len(present) > 6 else ''})\n",
            f"- Databases missing: {len(missing)}\n",
            f"- Mirror pairs complete: {len(cp)}/{len(cp)+len(ip)}\n\n",
        ]
        if missing:
            lines.append("**Missing (generate to unlock remaining phases):**\n\n")
            for n in missing:
                cmd = p0.get("status", {}).get(n, {}).get("generate_cmd", "")
                lines.append(f"- `{n}` — `{cmd}`\n")
            lines.append("\n")

    def q_section(key: str, title: str) -> list[str]:
        q = summary.get(key, {})
        sec = [f"## {title}\n\n"]
        if not q or q.get("skipped"):
            sec.append(f"*Skipped — {q.get('reason', 'no data')}*\n\n")
        else:
            sec.append(f"```\n{json.dumps(q, indent=2, default=str)[:2000]}\n```\n\n")
        return sec

    lines += q_section("q1", "Q1 — Mirror-Pair Symmetry")
    lines += q_section("q2", "Q2 — Epsilon-Symmetry Threshold")
    lines += q_section("q3", "Q3 — Nash Equilibrium")
    lines += q_section("q4", "Q4 — Quality-Aware Diversity Allocation")
    lines += q_section("q10", "Q10 — Swarm with True Specialists")
    lines += q_section("q5", "Q5 — Per-Robot Information Weighting")
    lines += q_section("q6", "Q6 — Per-Policy Rho for MFMC")
    lines += q_section("q9", "Q9 — Flip N* vs Target Difficulty")
    lines += q_section("q7", "Q7 — TrueSkill2 Multi-Team Rating")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[14] Saved {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open questions matrix (Exp 14)")
    p.add_argument("--phases", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   metavar="N", help="Which phases to run (0-4, default: all)")
    p.add_argument("--hifi", default="spectral",
                   help="Name of the high-fidelity reference source")
    p.add_argument("--n-values", nargs="+", type=int,
                   default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100],
                   metavar="N")
    p.add_argument("--n-total", type=int, default=10,
                   help="Fixed budget (HF-equivalent experiments) for Q3/Q4")
    p.add_argument("--n-bootstrap", type=int, default=200,
                   help="Bootstrap samples per curve estimate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db-prefix", default="",
                   help="Prepend this string to all db_path values from the manifest")
    p.add_argument("--study", nargs="+", metavar="NAME=PATH",
                   help="Override specific study db paths (name=path pairs)")
    p.add_argument("--output-dir", default=None, metavar="DIR")
    return p.parse_args()


def main() -> None:
    import datetime
    args = _parse_args()

    out = Path(args.output_dir) if args.output_dir else _OUT_DEFAULT
    out.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, str] = {}
    if args.study:
        for item in args.study:
            name, _, path = item.partition("=")
            overrides[name.strip()] = path.strip()

    manifest  = load_manifest()
    study_map = build_study_map(manifest, db_prefix=args.db_prefix, overrides=overrides)

    phases  = set(args.phases)
    summary: dict = {"generated": datetime.datetime.utcnow().isoformat()}

    # Phase 0
    p0 = phase0_validate(manifest, study_map, out)
    summary["phase0"] = p0

    if 0 not in phases:
        print("Phase 0 skipped (not in --phases).")
    else:
        pass  # already ran above unconditionally

    # Load databases once
    print("\n[14] Loading available databases...")
    all_studies = load_many_studies(study_map)
    print(f"  Loaded {len(all_studies)} studies: {list(all_studies.keys())}")

    mirror_pairs = get_mirror_pairs(manifest)
    hifi         = args.hifi
    n_vals       = sorted(set(args.n_values))
    n_bootstrap  = args.n_bootstrap
    seed         = args.seed
    n_total      = args.n_total

    paired_sources = [
        e["name"] for section in ("baseline_9", "mirror_8")
        for e in manifest.get(section, [])
        if e.get("paired_with_spectral") and e["name"] != hifi
    ]

    # Phase 1
    if 1 in phases:
        print("\n" + "=" * 60)
        print("PHASE 1 -- Mirror symmetry  Epsilon threshold  Nash equilibrium")
        print("=" * 60)
        summary["q1"] = q1_mirror_pairs(all_studies, mirror_pairs, hifi, n_bootstrap, seed)
        summary["q2"] = q2_epsilon_symmetry(all_studies, manifest, out)
        summary["q3"] = q3_nash_equilibrium(all_studies, hifi, n_total, n_bootstrap, seed)

    # Phase 2
    if 2 in phases:
        print("\n" + "=" * 60)
        print("PHASE 2 -- Quality-aware diversity  Swarm specialists")
        print("=" * 60)
        summary["q4"]  = q4_diversity_allocation(all_studies, hifi, n_total, n_bootstrap, seed)
        summary["q10"] = q10_swarm_specialists(all_studies, hifi, manifest, n_vals[:6], n_bootstrap, seed)

    # Phase 3
    if 3 in phases:
        print("\n" + "=" * 60)
        print("PHASE 3 -- Per-robot weighting  Per-policy rho  Flip vs difficulty")
        print("=" * 60)
        summary["q5"] = q5_per_robot_difficulty(all_studies, hifi, n_vals[:6], n_bootstrap, seed)
        summary["q6"] = q6_per_policy_rho(all_studies, hifi, paired_sources)
        summary["q9"] = q9_flip_by_difficulty(all_studies, hifi, n_vals[:6], n_bootstrap, seed)

    # Phase 4
    if 4 in phases:
        print("\n" + "=" * 60)
        print("PHASE 4 -- TrueSkill2 multi-team rating")
        print("=" * 60)
        summary["q7"] = q7_trueskill2_teams(all_studies, hifi, n_vals[:5], n_bootstrap, seed)

    write_summary(summary, out)


if __name__ == "__main__":
    main()
