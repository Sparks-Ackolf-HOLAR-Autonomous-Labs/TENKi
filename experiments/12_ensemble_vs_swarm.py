"""
Experiment 12 -- Source aggregation method comparison.

Evaluates five aggregation strategies at matched total budget B = N × K,
measuring Kendall tau vs the HF full-data reference rank.

Methods (ablation ladder):
  ensemble              equal weight 1/K, applied uniformly to every target
  local_router          stateless kNN: weight_S(t) = 1/mean_error(S near t)
  swarm_memory          stateful kNN memory, confidence weights, TenKi priors
  swarm_memory_abstain  as above + low-confidence agents stay silent
  swarm_consensus       full swarm: memory + abstention + one-round message-passing

Budget fairness
  All five methods see EXACTLY the same sampled draw in each bootstrap trial.
  A single rng.choice is made per trial; all modes receive the same list of
  sampled_targets.  Swarm agents load memory from those targets at the start
  of each trial -- no oracle knowledge from the full cube.

Interpretation ladder:
  swarm_memory > local_router      => stateful memory reduces bad upweighting
  swarm_memory_abstain > swarm_memory   => abstention prevents low-ceiling hijack
  swarm_consensus > swarm_memory_abstain => message-passing reduces disagreement

Success criterion: swarm earns the label only if at least one of the above
holds at matched budget.  Otherwise it is a more complex local_router.

Outputs (results/)
  aggregation_tau.png            tau vs N curves, all modes
  aggregation_weights.png        local_router source weight heatmap
  aggregation_comparison.json    tau tables + fairness diagnostics
  aggregation_comparison.md      human-readable summary with fairness notes
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.flip_data import load_study_scores
from analysis.flip_metrics import kendall_tau
from analysis.swarm_agents import (
    TenkiPriors,
    build_swarm,
    load_tenki_priors_from_json,
    BUILTIN_TENKI_PRIORS,
)
from analysis.swarm_consensus import swarm_rank_sampled

_DEFAULT_OUT = Path(_HERE).parent / "results"

DEFAULT_HFI = "spectral"
DEFAULT_LF  = ["mixbox", "km", "ryb"]
DEFAULT_DB  = {
    "spectral": "output/db_spectral",
    "mixbox":   "output/db_mixbox",
    "km":       "output/db_km",
    "ryb":      "output/db_ryb",
    "study_a":  "output/db_study_a_artist_consensus",
    "study_b":  "output/db_study_b_physics_vs_artist",
    "study_c":  "output/db_study_c_oilpaint_vs_fooddye",
}

ALL_MODES = [
    "ensemble",
    "local_router",
    "swarm_memory",
    "swarm_memory_abstain",
    "swarm_consensus",
]

_MODE_COLORS = {
    "ensemble":            "#1f77b4",
    "local_router":        "#2ca02c",
    "swarm_memory":        "#ff7f0e",
    "swarm_memory_abstain":"#d62728",
    "swarm_consensus":     "#9467bd",
}
_MODE_LABELS = {
    "ensemble":            "Ensemble (global equal weight)",
    "local_router":        "Local Router (kNN weights, stateless)",
    "swarm_memory":        "Swarm Memory (stateful, no abstain)",
    "swarm_memory_abstain":"Swarm + Abstention",
    "swarm_consensus":     "Swarm + Abstention + Consensus",
}
_MODE_MARKERS = {
    "ensemble":            "o",
    "local_router":        "s",
    "swarm_memory":        "^",
    "swarm_memory_abstain":"D",
    "swarm_consensus":     "P",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_best_per_target(
    db_path: str,
    policies: list[str],
    max_experiments: int | None = None,
) -> dict[str, dict[tuple, list[float]]]:
    """Load per-target best_color_distance from round JSON files."""
    import glob as _glob
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    policies_dir = os.path.join(_repo_root, db_path, "policies")

    scores: dict[str, dict[tuple, list[float]]] = {p: defaultdict(list) for p in policies}

    for pol in sorted(os.listdir(policies_dir)):
        if pol not in policies:
            continue
        exp_dirs = sorted(_glob.glob(os.path.join(policies_dir, pol, "experiment_*")))
        if max_experiments is not None:
            exp_dirs = exp_dirs[:max_experiments]
        for exp_dir in exp_dirs:
            for rfile in sorted(_glob.glob(os.path.join(exp_dir, "round_*.json"))):
                with open(rfile) as fh:
                    rd = json.load(fh)
                best_cd = rd.get("best_color_distance")
                trials  = rd.get("trials", [])
                if best_cd is None or not trials:
                    continue
                target = tuple(trials[0]["target_rgb"])
                scores[pol][target].append(float(best_cd))

    return {p: dict(scores[p]) for p in policies if scores[p]}


def _build_score_cube(
    db_map: dict[str, str],
    sources: list[str],
    policies: list[str],
    max_experiments: int | None = None,
    require_full_overlap: bool = False,
) -> tuple[dict[str, dict[str, dict[tuple, float]]], list[tuple]]:
    """
    Build cube[source][policy][target_rgb] = best_color_distance.
    Returns (cube, sorted target union).
    """
    cube: dict = {}
    target_union: set[tuple] = set()
    target_per_source: dict[str, set[tuple]] = {}

    for src in sources:
        raw = _load_best_per_target(db_map[src], policies, max_experiments)
        flat: dict[str, dict[tuple, float]] = {}
        src_targets: set[tuple] = set()
        for p in policies:
            flat[p] = {}
            for t, vals in raw.get(p, {}).items():
                if vals:
                    flat[p][t] = vals[0]
                    src_targets.add(t)
        cube[src] = flat
        target_union |= src_targets
        target_per_source[src] = src_targets

    target_set = (
        set.intersection(*target_per_source.values())
        if require_full_overlap and target_per_source
        else target_union
    )
    targets = sorted(target_set)

    for src, src_t in target_per_source.items():
        coverage = len(src_t & target_set) / max(len(target_set), 1)
        print(f"    {src}: {len(src_t)} targets  ({coverage:.0%} of union)")

    return cube, targets


# ---------------------------------------------------------------------------
# Per-method rankers accepting a PRE-SAMPLED list
# (no internal rng.choice -- caller samples once, passes to all methods)
# ---------------------------------------------------------------------------

def _ensemble_rank_sampled(
    cube: dict,
    sources: list[str],
    policies: list[str],
    sampled: list[tuple],
) -> list[str]:
    agg = {}
    for p in policies:
        vals = [
            cube[src][p][t]
            for src in sources
            for t in sampled
            if t in cube[src].get(p, {})
        ]
        agg[p] = float(np.mean(vals)) if vals else 1e9
    return sorted(policies, key=lambda p: agg[p])


def _local_router_weights(
    cube: dict,
    sources: list[str],
    policies: list[str],
    query_target: tuple,
    sampled: list[tuple],
    k: int,
) -> tuple[np.ndarray, int]:
    """
    Per-target source weights using LOO kNN: query_target is excluded from
    its own competence neighbourhood.  All copies of query_target in sampled
    (bootstrap duplicates) are removed together.

    Returns (weights, n_neighbors_used).
    n_neighbors_used=0 means pool was empty after exclusion (fallback applied).
    """
    # LOO: exclude all occurrences of the query target from the reference pool
    pool = [t for t in sampled if t != query_target]

    if not pool:
        # Fully starved (only target in sample is the query itself)
        return np.ones(len(sources)) / len(sources), 0

    tgt_arr = np.array(pool, dtype=float)
    q       = np.array(query_target, dtype=float)
    dists   = np.linalg.norm(tgt_arr - q, axis=1)
    k_eff   = min(k, len(pool))
    nn_targets = [pool[i] for i in np.argsort(dists)[:k_eff]]

    raw = np.array([
        float(np.mean([cube[src][p].get(t, np.nan)
                        for t in nn_targets
                        for p in policies
                        if t in cube[src].get(p, {})] or [1e6]))
        for src in sources
    ])
    inv = 1.0 / (raw + 1e-6)
    return inv / inv.sum(), k_eff


def _local_router_rank_sampled(
    cube: dict,
    sources: list[str],
    policies: list[str],
    sampled: list[tuple],
    k: int,
) -> tuple[list[str], np.ndarray, dict]:
    K = len(sources)
    n = len(sampled)
    wm = np.zeros((n, K))
    agg   = {p: 0.0 for p in policies}
    total = {p: 0.0 for p in policies}

    loo_empty   = 0
    loo_n_sum   = 0

    for ti, t in enumerate(sampled):
        lw, n_neighbors = _local_router_weights(cube, sources, policies, t, sampled, k)
        wm[ti] = lw
        if n_neighbors == 0:
            loo_empty += 1
        loo_n_sum += n_neighbors

        for p in policies:
            score_at_t = 0.0
            w_used = 0.0
            for si, src in enumerate(sources):
                v = cube[src][p].get(t)
                if v is not None:
                    score_at_t += lw[si] * v
                    w_used     += lw[si]
            if w_used > 0:
                agg[p]   += score_at_t / w_used
                total[p] += 1.0

    final = {p: agg[p] / max(total[p], 1e-9) for p in policies}
    diag = {
        "loo_empty_rate":    loo_empty  / max(n, 1),
        "loo_mean_neighbors": loo_n_sum / max(n, 1),
    }
    return sorted(policies, key=lambda p: final[p]), wm, diag


# ---------------------------------------------------------------------------
# Multi-mode tau vs N bootstrap (shared sample per trial)
# ---------------------------------------------------------------------------

def _tau_vs_n(
    cube: dict,
    sources: list[str],
    policies: list[str],
    all_targets: list[tuple],
    hf_rank: list[str],
    n_values: list[int],
    n_bootstrap: int,
    k_neighbors: int,
    rng: np.random.Generator,
    modes: list[str],
    tenki_priors: dict[str, TenkiPriors] | None = None,
    abstain_threshold: float = 0.25,
) -> dict[str, dict[int, dict]]:
    """
    Bootstrap tau curves for all requested modes.

    Single rng.choice per trial ensures all modes see the same sampled draw.
    Swarm agents rebuild memory from that draw at the start of each trial.

    Returns dict[mode][N] = {mean, std, q25, q75, diag}.
    """
    # Build swarm agents once (lightweight, no memory yet)
    agents = None
    if any("swarm" in m for m in modes):
        print("  Building SwarmAgents (lightweight, no memory yet)...")
        agents = build_swarm(sources, tenki_priors, abstain_threshold)
        for a in agents:
            p = a.priors
            print(
                f"    {a.name}: donor={p.donor_score:+.3f}, "
                f"bias_floor={p.bias_floor:.3f}, ceiling={p.ceiling:.3f}, "
                f"abstain_thresh={a.abstain_threshold}"
            )

    results: dict[str, dict[int, dict]] = {m: {} for m in modes}

    for n in n_values:
        print(f"    N={n} targets/source  (B={n*len(sources)} total)...", flush=True)
        mode_taus: dict[str, list[float]] = {m: [] for m in modes}
        # Diagnostic lists per mode: list of per-trial dicts
        mode_diags: dict[str, list[dict]] = {m: [] for m in modes}

        for _ in range(n_bootstrap):
            # --- Single draw, shared by all modes ---
            replace = n > len(all_targets)
            size = n if replace else min(n, len(all_targets))
            idx = rng.choice(len(all_targets), size=size, replace=replace)
            sampled = [all_targets[i] for i in idx]

            for mode in modes:
                if mode == "ensemble":
                    rank = _ensemble_rank_sampled(cube, sources, policies, sampled)
                    mode_taus[mode].append(kendall_tau(rank, hf_rank))

                elif mode == "local_router":
                    rank, _, d = _local_router_rank_sampled(
                        cube, sources, policies, sampled, k_neighbors,
                    )
                    mode_taus[mode].append(kendall_tau(rank, hf_rank))
                    mode_diags[mode].append(d)

                elif mode == "swarm_memory":
                    rank, _, d = swarm_rank_sampled(
                        cube, agents, policies, sampled,
                        k=k_neighbors, allow_abstain=False, message_rounds=0,
                    )
                    mode_taus[mode].append(kendall_tau(rank, hf_rank))
                    mode_diags[mode].append(d)

                elif mode == "swarm_memory_abstain":
                    rank, _, d = swarm_rank_sampled(
                        cube, agents, policies, sampled,
                        k=k_neighbors, allow_abstain=True, message_rounds=0,
                    )
                    mode_taus[mode].append(kendall_tau(rank, hf_rank))
                    mode_diags[mode].append(d)

                elif mode == "swarm_consensus":
                    rank, _, d = swarm_rank_sampled(
                        cube, agents, policies, sampled,
                        k=k_neighbors, allow_abstain=True, message_rounds=1,
                    )
                    mode_taus[mode].append(kendall_tau(rank, hf_rank))
                    mode_diags[mode].append(d)

        for mode in modes:
            taus = mode_taus[mode]
            diags = mode_diags[mode]
            entry: dict = {
                "mean": float(np.mean(taus)),
                "std":  float(np.std(taus)),
                "q25":  float(np.percentile(taus, 25)),
                "q75":  float(np.percentile(taus, 75)),
            }
            if diags:
                entry["diag"] = _aggregate_diags(diags)
            results[mode][n] = entry

    return results


def _aggregate_diags(diags: list[dict]) -> dict:
    """Average per-trial diagnostic dicts into one summary."""
    if not diags:
        return {}
    keys_float = [
        "abstention_rate", "fallback_rate", "downweight_rate",
        "mean_active_agents", "mean_memory_size", "mean_memory_coverage",
    ]
    out: dict = {}
    for k in keys_float:
        vals = [d[k] for d in diags if k in d]
        if vals:
            out[k] = float(np.mean(vals))
    # Fallback type counts (most common)
    all_ft: dict[str, int] = {}
    for d in diags:
        for ft, cnt in d.get("fallback_types", {}).items():
            all_ft[ft] = all_ft.get(ft, 0) + cnt
    if all_ft:
        out["fallback_types"] = all_ft
    # Memory sizes: mean per agent across trials
    all_ms = [d["memory_sizes"] for d in diags if "memory_sizes" in d]
    if all_ms:
        arr = np.array(all_ms, dtype=float)
        out["mean_memory_sizes_per_agent"] = arr.mean(axis=0).tolist()
    return out


# ---------------------------------------------------------------------------
# Weight profile for local_router diagnostic plot
# ---------------------------------------------------------------------------

def _local_router_weight_profile(
    cube: dict,
    sources: list[str],
    policies: list[str],
    all_targets: list[tuple],
    k: int,
    rng: np.random.Generator,
    n_targets: int | None = None,
) -> np.ndarray:
    n = n_targets or len(all_targets)
    replace = n > len(all_targets)
    size = n if replace else min(n, len(all_targets))
    idx = rng.choice(len(all_targets), size=size, replace=replace)
    sampled = [all_targets[i] for i in idx]
    _, wm, _ = _local_router_rank_sampled(cube, sources, policies, sampled, k)
    return wm, sampled


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_tau_curves(
    tau_results: dict[str, dict[int, dict]],
    sources: list[str],
    n_values: list[int],
    modes: list[str],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for mode in modes:
        if mode not in tau_results or not tau_results[mode]:
            continue
        ns    = sorted(tau_results[mode].keys())
        means = [tau_results[mode][n]["mean"] for n in ns]
        q25   = [tau_results[mode][n]["q25"]  for n in ns]
        q75   = [tau_results[mode][n]["q75"]  for n in ns]

        ax.plot(
            ns, means,
            marker=_MODE_MARKERS.get(mode, "o"),
            color=_MODE_COLORS.get(mode, "gray"),
            label=_MODE_LABELS.get(mode, mode),
            linewidth=2, markersize=6,
        )
        ax.fill_between(ns, q25, q75, alpha=0.12,
                        color=_MODE_COLORS.get(mode, "gray"))

    ax.set_xlabel(f"N targets sampled per source  (K={len(sources)}, total B = N × K)")
    ax.set_ylabel("Kendall tau vs HF full-data rank")
    ax.set_title(
        "Source aggregation: Kendall tau at matched budget\n"
        "(all modes see same sampled draw per trial)"
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[12] Saved {save_path.name}")


def _plot_weights(
    weight_matrix: np.ndarray,
    sources: list[str],
    targets: list[tuple],
    save_path: Path,
    mode_label: str = "Local Router",
) -> None:
    brightness = [sum(t) for t in targets]
    order = np.argsort(brightness)
    wm = weight_matrix[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(targets) * 0.3 + 2)))

    im = ax1.imshow(wm, aspect="auto", cmap="YlOrRd", vmin=0)
    ax1.set_xticks(range(len(sources)))
    ax1.set_xticklabels(sources, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(targets)))
    ax1.set_yticklabels([f"t{i}" for i in range(len(targets))], fontsize=6)
    ax1.set_xlabel("Source")
    ax1.set_ylabel("Target (sorted by brightness)")
    ax1.set_title(f"{mode_label}: source weights per target")
    plt.colorbar(im, ax=ax1, label="weight")

    mean_w = wm.mean(axis=0)
    ax2.barh(range(len(sources)), mean_w, color="#2ca02c", alpha=0.7)
    ax2.set_yticks(range(len(sources)))
    ax2.set_yticklabels(sources)
    ax2.set_xlabel("Mean weight across all targets")
    ax2.set_title(f"{mode_label}: average source attention")
    ax2.axvline(1.0 / len(sources), color="gray", linestyle="--",
                linewidth=1.2, label="Equal weight baseline")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[12] Saved {save_path.name}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown(
    tau_results: dict[str, dict[int, dict]],
    sources: list[str],
    hf_name: str,
    modes: list[str],
    mean_w: np.ndarray,
    save_path: Path,
) -> None:
    lines = [
        "# Experiment 12 -- Source Aggregation Method Comparison",
        "",
        "## Fairness contract",
        "",
        "- All methods receive the **same sampled draw** per bootstrap trial.",
        "- Swarm agent memory is built from that draw only (no oracle knowledge).",
        "- Total budget is identical: B = N x K for all methods.",
        "- Memory model: Version A (source-level, policy-agnostic mean error per target).",
        "",
        "## Setup",
        f"- HF reference: `{hf_name}`",
        f"- LF sources ({len(sources)}): " + ", ".join(f"`{s}`" for s in sources),
        "",
        "## Method definitions",
        "",
        "**ensemble**: 1/K weight applied uniformly. Blind to local source reliability.",
        "",
        "**local_router**: stateless kNN. weight_S(t) = 1/mean_error(S near t). "
        "No state, no memory between targets.",
        "",
        "**swarm_memory**: stateful agents with kNN memory loaded from sampled targets. "
        "Weights = confidence/predicted_error. TenKi priors gate authority.",
        "",
        "**swarm_memory_abstain**: as above + agents with low neighbourhood "
        "confidence stay silent. Fallback: TenKi-reputation weights (not equal).",
        "",
        "**swarm_consensus**: full swarm: memory + abstention + one-round "
        "message-passing. Agents deviating >1.5 sigma from consensus downweighted.",
        "",
        "## Tau vs N (mean, vs ensemble delta)",
        "",
    ]

    header = "| N/source |"
    sep    = "|---|"
    for m in modes:
        if m in tau_results and tau_results[m]:
            header += f" {m.replace('_',' ')} |"
            sep    += "---|"
    lines += [header, sep]

    all_n = sorted({n for m in modes for n in tau_results.get(m, {}).keys()})
    for n in all_n:
        row = f"| {n} |"
        ref = tau_results.get("ensemble", {}).get(n, {}).get("mean")
        for m in modes:
            if m not in tau_results or not tau_results[m]:
                continue
            e = tau_results[m].get(n)
            if e:
                t = e["mean"]
                if m != "ensemble" and ref is not None:
                    adv = t - ref
                    row += f" {t:.3f} ({'+' if adv>=0 else ''}{adv:.3f}) |"
                else:
                    row += f" {t:.3f} |"
            else:
                row += " — |"
        lines.append(row)

    # Diagnostics per mode
    swarm_modes = [m for m in modes if "swarm" in m]
    if swarm_modes:
        lines += ["", "## Swarm diagnostics (mean across all N and bootstrap trials)", ""]
        lines += [
            "| Mode | mem/agent | coverage | abstention | fallback | downweight | active |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in swarm_modes:
            all_diags = [
                v["diag"] for v in tau_results.get(m, {}).values()
                if "diag" in v and v["diag"]
            ]
            if not all_diags:
                continue

            def _mean_key(key):
                vals = [d.get(key) for d in all_diags if d.get(key) is not None]
                return float(np.mean(vals)) if vals else float("nan")

            lines.append(
                f"| {m} "
                f"| {_mean_key('mean_memory_size'):.1f} "
                f"| {_mean_key('mean_memory_coverage'):.2f} "
                f"| {_mean_key('abstention_rate'):.3f} "
                f"| {_mean_key('fallback_rate'):.3f} "
                f"| {_mean_key('downweight_rate'):.3f} "
                f"| {_mean_key('mean_active_agents'):.2f} |"
            )

    # Success criterion
    lines += ["", "## Swarm success criterion", ""]
    ens_at_max = None
    swarm_results_at_max = {}
    if all_n:
        max_n = max(all_n)
        ens_at_max = tau_results.get("ensemble", {}).get(max_n, {}).get("mean")
        for m in swarm_modes:
            v = tau_results.get(m, {}).get(max_n, {})
            if v:
                swarm_results_at_max[m] = v.get("mean")

    lr_at_max = tau_results.get("local_router", {}).get(max(all_n, default=0), {}).get("mean") if all_n else None
    for m in swarm_modes:
        t_val = swarm_results_at_max.get(m)
        if t_val is not None and lr_at_max is not None:
            diff = t_val - lr_at_max
            verdict = "BEATS local_router" if diff > 0.005 else (
                "TIES local_router" if abs(diff) <= 0.005 else "LOSES to local_router"
            )
            lines.append(f"- **{m}** vs local_router at N={max(all_n)}: "
                         f"delta={diff:+.3f}  → {verdict}")

    # TenKi source trust
    lines += ["", "## TenKi structural trust (built-in priors)", ""]
    lines += ["| Source | donor_score | bias_floor | ceiling |", "|---|---|---|---|"]
    for src in sources:
        p = BUILTIN_TENKI_PRIORS.get(src)
        if p:
            lines.append(
                f"| {src} | {p.donor_score:+.3f} | {p.bias_floor:.3f} | {p.ceiling:.3f} |"
            )

    # Local router source attention
    lines += [
        "", "## Local Router: source attention", "",
        "| Source | Mean weight | vs equal |",
        "|---|---|---|",
    ]
    eq = 1.0 / len(sources)
    for si, src in enumerate(sources):
        d = mean_w[si] - eq
        lines.append(f"| {src} | {mean_w[si]:.3f} | {'+' if d>=0 else ''}{d:.3f} |")

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[12] Saved {save_path.name}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    hf: str = DEFAULT_HFI,
    lf_sources: list[str] | None = None,
    db_map: dict[str, str] | None = None,
    n_values: list[int] | None = None,
    n_bootstrap: int = 200,
    k_neighbors: int = 5,
    max_experiments: int | None = None,
    seed: int = 42,
    output_dir: Path | str | None = None,
    modes: list[str] | None = None,
    tenki_priors_path: str | None = None,
    abstain_threshold: float = 0.25,
) -> None:
    out = Path(output_dir) if output_dir else _DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)

    sources    = lf_sources or DEFAULT_LF
    dbm        = db_map or DEFAULT_DB
    n_vals     = n_values or [1, 5, 10, 50, 100, 500, 1000]
    act_modes  = modes or ALL_MODES
    rng        = np.random.default_rng(seed)

    # TenKi priors
    if tenki_priors_path:
        tenki_priors: dict[str, TenkiPriors] | None = load_tenki_priors_from_json(tenki_priors_path)
        print(f"[12] Loaded TenKi priors from {tenki_priors_path}")
    else:
        tenki_priors = {src: BUILTIN_TENKI_PRIORS[src]
                        for src in sources if src in BUILTIN_TENKI_PRIORS}
        print("[12] Using built-in TenKi priors (TENKi-1000 canonical values)")

    # HF reference rank
    print("[12] Loading HF reference ranking...")
    hf_study = load_study_scores(dbm[hf])
    hf_rank  = hf_study.full_rank
    policies = hf_rank
    print(f"  HF: {hf}  |  {len(policies)} policies  |  rank: {policies}")

    # Validate LF sources
    valid_sources = []
    for src in sources:
        if src not in dbm:
            print(f"  [skip] {src}: not in db_map")
            continue
        st = load_study_scores(dbm[src])
        if st.n_policies == 0:
            print(f"  [skip] {src}: database not found")
            continue
        common = [p for p in policies if p in st.policy_scores]
        if len(common) < len(policies):
            print(f"  [warn] {src}: missing {set(policies) - set(common)}")
        print(f"  LF: {src}  ({st.n_policies} policies, {st.max_n} experiments)")
        valid_sources.append(src)

    if not valid_sources:
        print("[12] No valid LF sources -- abort.")
        return
    sources = valid_sources

    # Build score cube
    all_sources = [hf] + sources
    print(f"\n[12] Loading per-target scores for {len(all_sources)} sources...")
    cube, targets = _build_score_cube(dbm, all_sources, policies, max_experiments)
    print(f"  Union targets: {len(targets)}")

    if len(targets) < 3:
        print("[12] Too few targets for kNN (need >= 3). Abort.")
        return
    k = min(k_neighbors, len(targets))

    # HF cube rank consistency check
    hf_agg = {p: float(np.mean(list(cube[hf][p].values()))) if cube[hf].get(p) else 1e9
              for p in policies}
    hf_cube_rank = sorted(policies, key=lambda p: hf_agg[p])
    tau_check = kendall_tau(hf_cube_rank, hf_rank)
    print(f"  HF cube rank:     {hf_cube_rank}")
    print(f"  HF DB rank:       {hf_rank}")
    print(f"  Consistency tau:  {tau_check:.3f}")
    reference_rank = hf_cube_rank

    print(f"\n[12] Modes: {act_modes}")
    print(f"  k_neighbors={k}, abstain_threshold={abstain_threshold}, "
          f"n_bootstrap={n_bootstrap}")
    print(f"  N values: {n_vals}  (bootstrap replaces when N > {len(targets)})")

    # Tau curves
    print(f"\n[12] Computing tau vs N  ({n_bootstrap} bootstraps per N)...")
    tau_results = _tau_vs_n(
        cube, sources, policies, targets,
        reference_rank, n_vals, n_bootstrap, k, rng,
        modes=act_modes,
        tenki_priors=tenki_priors,
        abstain_threshold=abstain_threshold,
    )

    # Console summary
    print("\n=== Aggregation comparison: Kendall tau (matched budget B = N x K) ===")
    print(f"  K={len(sources)} sources, k_neighbors={k}")
    hdr = f"  {'N':>6}"
    for m in act_modes:
        hdr += f"  {m[:16]:>16}"
    print(hdr)
    for n in n_vals:
        row = f"  {n:>6}"
        for m in act_modes:
            e = tau_results.get(m, {}).get(n)
            row += f"  {e['mean']:>16.3f}" if e else f"  {'—':>16}"
        print(row)

    # Weight profile for local_router plot
    print("\n[12] Computing Local Router weight profile...")
    wm, sampled_for_plot = _local_router_weight_profile(
        cube, sources, policies, targets, k, rng, n_targets=len(targets),
    )
    mean_w = wm.mean(axis=0)
    wm_std = wm.std(axis=0)
    eq = 1.0 / len(sources)
    print("  Mean weights per source:")
    for si, src in enumerate(sources):
        bar = "#" * int(mean_w[si] * 40)
        print(f"    {src:>12}: {mean_w[si]:.3f}  (equal={eq:.3f})  std={wm_std[si]:.3f}  {bar}")

    # Plots
    _plot_tau_curves(tau_results, sources, n_vals, act_modes, out / "aggregation_tau.png")
    _plot_weights(wm, sources, sampled_for_plot, out / "aggregation_weights.png",
                  mode_label="Local Router")

    # JSON output
    result_dict = {
        "experiment": "12_aggregation_comparison",
        "fairness_note": (
            "All methods share the same sampled draw per bootstrap trial. "
            "Swarm memory loaded from sampled targets only (Version A: source-level)."
        ),
        "hf": hf,
        "sources": sources,
        "k_neighbors": k,
        "n_values": n_vals,
        "n_bootstrap": n_bootstrap,
        "abstain_threshold": abstain_threshold,
        "modes": act_modes,
        "tau_results": {
            mode: {
                str(n): {
                    k2: v2 for k2, v2 in entry.items()
                    if k2 != "diag"  # keep tau stats compact at top level
                }
                for n, entry in data.items()
            }
            for mode, data in tau_results.items()
        },
        "diagnostics_by_mode_and_n": {
            mode: {
                str(n): entry.get("diag", {})
                for n, entry in data.items()
            }
            for mode, data in tau_results.items()
        },
        "local_router_mean_weights": {src: float(mean_w[si]) for si, src in enumerate(sources)},
        "local_router_weight_std":   {src: float(wm_std[si]) for si, src in enumerate(sources)},
        "tenki_priors": {
            src: {
                "donor_score": p.donor_score,
                "bias_floor":  p.bias_floor,
                "ceiling":     p.ceiling,
            }
            for src, p in (tenki_priors or {}).items()
        },
        "hf_cube_rank":    reference_rank,
        "hf_db_rank":      hf_rank,
        "consistency_tau": float(tau_check),
    }
    json_path = out / "aggregation_comparison.json"
    json_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    print(f"[12] Saved {json_path.name}")

    _write_markdown(tau_results, sources, hf, act_modes, mean_w, out / "aggregation_comparison.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Source aggregation comparison (Exp 12)")
    p.add_argument("--hfi", default=DEFAULT_HFI)
    p.add_argument("--lf", nargs="+", default=DEFAULT_LF)
    p.add_argument("--studies", nargs="+", metavar="NAME=PATH",
                   help="Override database paths, e.g. spectral=../../output/db_spectral_1k")
    p.add_argument("--n-values", nargs="+", type=int,
                   default=[1, 5, 10, 50, 100, 500, 1000])
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--max-experiments", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode", nargs="+", dest="modes",
        choices=ALL_MODES, default=ALL_MODES,
        help="Aggregation modes to run (default: all five)",
    )
    p.add_argument(
        "--tenki-priors", default=None,
        help="JSON file with donor_score/bias_floor/ceiling per source. "
             "Falls back to built-in canonical values (TENKi-1000) if omitted.",
    )
    p.add_argument(
        "--abstain-threshold", type=float, default=0.25,
        help="Confidence threshold below which swarm agents abstain (default: 0.25)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    db_map = dict(DEFAULT_DB)
    if args.studies:
        for item in args.studies:
            name, _, path = item.partition("=")
            db_map[name.strip()] = path.strip()
    run(
        hf=args.hfi,
        lf_sources=args.lf,
        db_map=db_map,
        n_values=args.n_values,
        n_bootstrap=args.n_bootstrap,
        k_neighbors=args.k_neighbors,
        max_experiments=args.max_experiments,
        seed=args.seed,
        output_dir=args.output_dir,
        modes=args.modes,
        tenki_priors_path=args.tenki_priors,
        abstain_threshold=args.abstain_threshold,
    )
