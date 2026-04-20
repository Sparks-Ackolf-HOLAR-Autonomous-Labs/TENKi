"""
experiments/13_async_mf_optimizer.py -- Async MF optimizer entry point.

Runs the AsyncMFScheduler over PEGKi study databases using the offline
simulation executor (no real async, no sleeping). Produces:

  results/async_mf_optimizer_summary.json
  results/async_mf_optimizer_trace.jsonl
  results/async_mf_optimizer_tau_vs_time.png
  results/async_mf_optimizer_allocations.png

Usage examples:
  # Minimal (uses defaults -- spectral as HF, mixbox+km as LF)
  python experiments/13_async_mf_optimizer.py

  # Choose policy mode
  python experiments/13_async_mf_optimizer.py --policy-mode swarm_mf

  # Custom sources
  python experiments/13_async_mf_optimizer.py \\
      --hifi spectral \\
      --lf-sources mixbox km ryb \\
      --fidelity-levels 1 3 5 10 \\
      --budget-total 200 \\
      --max-workers 4 \\
      --save-dir runs/exp13_swarm

  # With studies provided explicitly
  python experiments/13_async_mf_optimizer.py \\
      --study spectral=../../../output/db_spectral \\
      --study mixbox=../../../output/db_mixbox \\
      --study km=../../../output/db_km

  # With fidelity-DB map (multi-fidelity databases per source)
  python experiments/13_async_mf_optimizer.py \\
      --policy-mode mfbo \\
      --fidelity-db SPECTRAL:3=output/db_spectral_r3 \\
      --fidelity-db SPECTRAL:12=output/db_spectral_r12

  # SMAC policy
  python experiments/13_async_mf_optimizer.py --policy-mode smac

  # Hyperband policy
  python experiments/13_async_mf_optimizer.py --policy-mode hyperband

  # MFMC policy
  python experiments/13_async_mf_optimizer.py --policy-mode mfmc
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
sys.path.insert(0, str(_ROOT.parent.parent))

_DEFAULT_RESULTS = _ROOT / "results"

# Default study databases (relative to repo root)
_DEFAULT_STUDIES = {
    "spectral": str(_ROOT.parent.parent / "output" / "db_spectral"),
    "mixbox":   str(_ROOT.parent.parent / "output" / "db_mixbox"),
    "km":       str(_ROOT.parent.parent / "output" / "db_km"),
}


# ---------------------------------------------------------------------------
# Imports from optimizer package
# ---------------------------------------------------------------------------
from optimizer.types import Config, RunState, WorkerState, BeliefState
from optimizer.runtime_model import ConstantRuntimeModel, SourceFidelityRuntimeModel
from optimizer.objective import OfflineReplayObjective
from optimizer.executor import LocalSimExecutor
from optimizer.store import RunStore
from optimizer.policies import (
    SingleSourceMFPolicy, EnsembleMFPolicy, SwarmMFPolicy,
    MFBOPolicy, MFMCPolicy, HyperbandPolicy, SMACPolicy,
)
from optimizer.scheduler import AsyncMFScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime_model(mode: str, sources: list[str], fidelity_levels: list[int | float]):
    """Build a runtime model from CLI mode string."""
    if mode == "constant":
        return ConstantRuntimeModel(runtime=1.0)
    # source_fidelity: spectral costs 3x, others 1x; scale so fid=1->0.1, fid=10->1.0
    multipliers = {s: (3.0 if "spectral" in s else 1.0) for s in sources}
    return SourceFidelityRuntimeModel(
        source_multipliers=multipliers,
        fidelity_scale=0.1,
        noise_std=0.0,
    )


def _make_policy(
    mode: str,
    hifi: str,
    lf_sources: list[str],
    all_sources: list[str],
    policies: list[str],
    fidelity_levels: list[int | float],
    runtime_model,
    explore_weight: float,
    rng: np.random.Generator,
    budget_total: float = 100.0,
):
    if mode == "single_source_mf":
        lf = lf_sources[0] if lf_sources else all_sources[-1]
        return SingleSourceMFPolicy(
            hifi_source=hifi,
            lf_source=lf,
            policies=policies,
            fidelity_levels=fidelity_levels,
            runtime_model=runtime_model,
            rng=rng,
        )
    if mode == "swarm_mf":
        return SwarmMFPolicy(
            sources=all_sources,
            policies=policies,
            fidelity_levels=fidelity_levels,
            runtime_model=runtime_model,
            explore_weight=explore_weight,
            rng=rng,
        )
    if mode == "mfbo":
        return MFBOPolicy(
            sources=all_sources,
            policies=policies,
            fidelity_levels=fidelity_levels,
            runtime_model=runtime_model,
            rng=rng,
        )
    if mode == "smac":
        return SMACPolicy(
            sources=all_sources,
            policies=policies,
            fidelity_levels=fidelity_levels,
            runtime_model=runtime_model,
            rng=rng,
        )
    if mode == "hyperband":
        return HyperbandPolicy(
            sources=all_sources,
            policies=policies,
            fidelity_levels=fidelity_levels,
            runtime_model=runtime_model,
        )
    if mode == "mfmc":
        return MFMCPolicy(
            hifi_source=hifi,
            lf_sources=lf_sources,
            fidelity_levels=fidelity_levels,
            runtime_model=runtime_model,
            budget_total=budget_total,
        )
    # default: ensemble_mf
    return EnsembleMFPolicy(
        sources=all_sources,
        policies=policies,
        fidelity_levels=fidelity_levels,
        runtime_model=runtime_model,
        rng=rng,
    )


def _initial_beliefs(sources: list[str]) -> list[BeliefState]:
    """Uniform prior beliefs for all sources."""
    return [
        BeliefState(
            source_name=s,
            fidelity=None,
            tau_mean=0.5,
            tau_std=0.2,
            rho_mean=0.5,
            bias_floor=0.0,
            flip_probability=None,
            donor_score=0.5,
            effective_fidelity=0.5,
            quality_score=0.5,
            last_updated_time=0.0,
            n_observations=0,
        )
        for s in sources
    ]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_tau_vs_time(trace: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping tau vs time plot")
        return

    if not trace:
        return

    times = [e.get("sim_time", 0) for e in trace if e.get("event_type") == "belief_update"]
    taus  = [e.get("best_tau", 0) for e in trace if e.get("event_type") == "belief_update"]
    if not times:
        # fall back: any event with best_tau
        times = [e.get("sim_time", 0) for e in trace if "best_tau" in e]
        taus  = [e.get("best_tau", 0) for e in trace if "best_tau" in e]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, taus, marker="o", ms=4, lw=1.5, color="steelblue")
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Best tau (Kendall)")
    ax.set_title("Tau vs simulated time")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  tau plot -> {out_path}")


def _plot_allocations(completed: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping allocations plot")
        return

    if not completed:
        return

    from collections import Counter
    counts = Counter((c.get("source", "?"), c.get("fidelity", 0)) for c in completed)

    # Group by source
    sources = sorted({k[0] for k in counts})
    fids    = sorted({k[1] for k in counts})
    x = range(len(sources))
    width = 0.8 / max(len(fids), 1)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, fid in enumerate(fids):
        vals = [counts.get((s, fid), 0) for s in sources]
        offset = (i - len(fids) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width * 0.9,
                      label=f"fid={fid}", color=cmap(i))

    ax.set_xticks(list(x))
    ax.set_xticklabels(sources, rotation=15, ha="right")
    ax.set_ylabel("Jobs completed")
    ax.set_title("Allocation by source and fidelity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  allocations plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    hifi: str,
    lf_sources: list[str],
    study_map: dict[str, str],
    policy_mode: str,
    executor_mode: str,
    runtime_mode: str,
    fidelity_levels: list[int | float],
    budget_total: float,
    max_workers: int,
    allow_resume: bool,
    target_tau: float,
    n_bootstrap: int,
    explore_weight: float,
    seed: int,
    save_dir: str | None,
    output_dir: Path | None = None,
    snapshot_interval: int = 5,
    fidelity_db_map: dict | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    all_sources = [hifi] + [s for s in lf_sources if s != hifi]
    out = output_dir or _DEFAULT_RESULTS
    fidelity_db_map = fidelity_db_map or {}

    # Build objective
    objective = OfflineReplayObjective(
        study_map=study_map,
        hifi_source=hifi,
        score_key="best_color_distance_mean",
        lower_is_better=True,
        allow_resume=allow_resume,
        rng=np.random.default_rng(seed + 1),
        fidelity_db_map=fidelity_db_map,
    )

    # Discover available policies from HF study
    policies = objective.all_policies()
    hf_rank  = objective.hf_rank()
    if not policies:
        print(f"  WARNING: no policies found for hifi source '{hifi}'.")
        print(f"  Check that study_map['{hifi}'] = {study_map.get(hifi)} exists and is valid.")
        policies = ["grid_search", "ucb1_bandit", "bayesian_ei"]
        hf_rank  = policies

    print(f"  policies ({len(policies)}): {policies[:5]}{'...' if len(policies) > 5 else ''}")
    print(f"  hf_rank:  {hf_rank}")

    # Runtime model
    rt_model = _make_runtime_model(runtime_mode, all_sources, fidelity_levels)

    # Policy
    policy = _make_policy(
        mode=policy_mode,
        hifi=hifi,
        lf_sources=lf_sources,
        all_sources=all_sources,
        policies=policies,
        fidelity_levels=fidelity_levels,
        runtime_model=rt_model,
        explore_weight=explore_weight,
        rng=rng,
        budget_total=budget_total,
    )

    # Config
    cfg = Config(
        run_name="exp13_async_mf",
        seed=seed,
        hifi_source=hifi,
        lf_sources=lf_sources,
        policy_mode=policy_mode,
        executor_mode=executor_mode,
        runtime_mode=runtime_mode,
        max_workers=max_workers,
        budget_total=budget_total,
        budget_units="runtime",
        fidelity_axis="n_experiments",
        fidelity_levels=fidelity_levels,
        n_bootstrap=n_bootstrap,
        allow_resume=allow_resume,
        save_dir=save_dir or "runs/exp13",
        fidelity_db_map=fidelity_db_map,
    )

    # RunState
    workers = [WorkerState(worker_id=f"w{i:02d}") for i in range(max_workers)]
    run_state = RunState(
        workers=workers,
        beliefs=_initial_beliefs(all_sources),
    )

    # Store
    actual_save_dir = save_dir or str(out / "runs")
    store = RunStore(save_dir=actual_save_dir, run_name="exp13_async_mf")

    # Executor
    executor = LocalSimExecutor(objective=objective)

    # Scheduler
    scheduler = AsyncMFScheduler(
        config=cfg,
        run_state=run_state,
        executor=executor,
        objective=objective,
        policy=policy,
        store=store,
        hf_rank=hf_rank,
        lower_is_better=True,
    )

    print(f"  Running scheduler (budget={budget_total}, mode={policy_mode}, workers={max_workers})...")
    scheduler.run_until_budget(snapshot_interval=snapshot_interval)
    store.close()

    # Collect results from run_state
    completed_dicts = [
        {
            "job_id": c.job_id,
            "source": c.source,
            "policy_name": c.policy_name,
            "fidelity": c.fidelity,
            "score": c.score,
            "runtime_simulated_end": c.runtime_simulated_end,
        }
        for c in run_state.completed_jobs
    ]

    # Build trace from coordination events file
    events_path = Path(actual_save_dir) / "exp13_async_mf" / "coordination" / "events.jsonl"
    trace = []
    if events_path.exists():
        with open(str(events_path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trace.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    metrics = dict(run_state.metrics)
    beliefs_summary = [
        {
            "source": b.source_name,
            "fidelity": b.fidelity,
            "tau_mean": b.tau_mean,
            "rho_mean": b.rho_mean,
            "bias_floor": b.bias_floor,
            "donor_score": b.donor_score,
            "effective_fidelity": b.effective_fidelity,
            "quality_score": b.quality_score,
            "n_observations": b.n_observations,
        }
        for b in run_state.beliefs
    ]

    summary = {
        "run_name": cfg.run_name,
        "hifi_source": hifi,
        "lf_sources": lf_sources,
        "policy_mode": policy_mode,
        "fidelity_levels": fidelity_levels,
        "budget_total": budget_total,
        "sim_time_final": run_state.sim_time,
        "n_completed_jobs": len(completed_dicts),
        "metrics": metrics,
        "beliefs": beliefs_summary,
        "hf_rank": hf_rank,
    }

    # Write outputs
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "async_mf_optimizer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  summary -> {summary_path}")

    trace_path = out / "async_mf_optimizer_trace.jsonl"
    with open(str(trace_path), "w", encoding="utf-8") as f:
        for ev in trace:
            f.write(json.dumps(ev) + "\n")
    print(f"  trace   -> {trace_path} ({len(trace)} events)")

    # Plots
    _plot_tau_vs_time(trace, out / "async_mf_optimizer_tau_vs_time.png")
    _plot_allocations(completed_dicts, out / "async_mf_optimizer_allocations.png")

    print(f"\n  sim_time={run_state.sim_time:.1f}  jobs={len(completed_dicts)}  "
          f"best_tau={metrics.get('best_tau', 0):.3f}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_fidelity_db(entries: list[str]) -> dict[str, dict[str, str]]:
    """
    Parse --fidelity-db SOURCE:FIDELITY=PATH triples into a nested dict.

    Example:
        --fidelity-db SPECTRAL:3=output/db_spectral_r3
        --fidelity-db SPECTRAL:12=output/db_spectral_r12
    Returns:
        {"SPECTRAL": {"3": "output/db_spectral_r3", "12": "output/db_spectral_r12"}}
    """
    result: dict[str, dict[str, str]] = {}
    for entry in entries:
        if "=" not in entry or ":" not in entry.split("=", 1)[0]:
            print(f"ERROR: --fidelity-db must be SOURCE:FIDELITY=PATH, got: {entry}")
            sys.exit(1)
        src_fid, path = entry.split("=", 1)
        source, fid_str = src_fid.split(":", 1)
        if source not in result:
            result[source] = {}
        result[source][fid_str] = path
    return result


def _parse_args():
    p = argparse.ArgumentParser(
        description="Async MF optimizer experiment (offline sim)"
    )
    p.add_argument("--hifi", default="spectral",
                   help="Name of the high-fidelity source (default: spectral)")
    p.add_argument("--lf-sources", nargs="+", default=["mixbox", "km"],
                   help="Low-fidelity source names (default: mixbox km)")
    p.add_argument("--study", action="append", default=[], metavar="NAME=PATH",
                   help="Override study DB path: NAME=PATH (repeatable)")
    p.add_argument("--policy-mode", default="ensemble_mf",
                   choices=["single_source_mf", "ensemble_mf", "swarm_mf",
                            "mfbo", "smac", "hyperband", "mfmc"],
                   help="Allocation policy (default: ensemble_mf)")
    p.add_argument("--executor-mode", default="local_sim",
                   choices=["local_sim"],
                   help="Executor backend (default: local_sim)")
    p.add_argument("--runtime-mode", default="source_fidelity",
                   choices=["constant", "source_fidelity"],
                   help="Runtime model (default: source_fidelity)")
    p.add_argument("--fidelity-axis", default="n_experiments",
                   help="Fidelity axis name (default: n_experiments)")
    p.add_argument("--fidelity-levels", nargs="+", type=int, default=[1, 3, 5],
                   help="Fidelity values to sweep (default: 1 3 5)")
    p.add_argument("--budget-total", type=float, default=50.0,
                   help="Budget in runtime units (default: 50)")
    p.add_argument("--max-workers", type=int, default=2,
                   help="Simulated parallel workers (default: 2)")
    p.add_argument("--allow-resume", action="store_true", default=True,
                   help="Allow fidelity promotions via resume tokens")
    p.add_argument("--no-resume", dest="allow_resume", action="store_false",
                   help="Disable fidelity promotion resume")
    p.add_argument("--target-tau", type=float, default=0.8,
                   help="Early-stop when best tau >= this (default: 0.8; not yet implemented)")
    p.add_argument("--n-bootstrap", type=int, default=100,
                   help="Bootstrap samples for tau confidence (default: 100)")
    p.add_argument("--explore-weight", type=float, default=0.2,
                   help="Exploration weight for swarm UCB (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--save-dir", default=None,
                   help="Directory for run snapshots (default: results/runs)")
    p.add_argument("--output-dir", default=None,
                   help="Directory for result outputs (default: results/)")
    p.add_argument("--snapshot-interval", type=int, default=5,
                   help="Save snapshot every N scheduler steps (default: 5)")
    p.add_argument("--fidelity-db", action="append", default=[],
                   metavar="SOURCE:FIDELITY=PATH",
                   help=(
                       "Map a fidelity level to a pre-generated PEGKi database. "
                       "Format: SOURCE:FIDELITY=PATH (repeatable). "
                       "Example: --fidelity-db SPECTRAL:3=output/db_spectral_r3"
                   ))
    return p.parse_args()


def main():
    args = _parse_args()

    # Build study map: defaults, then overrides from --study NAME=PATH
    study_map = dict(_DEFAULT_STUDIES)
    for entry in args.study:
        if "=" not in entry:
            print(f"ERROR: --study must be NAME=PATH, got: {entry}")
            sys.exit(1)
        name, path = entry.split("=", 1)
        study_map[name] = path

    # Keep only sources that are referenced
    all_source_names = [args.hifi] + [s for s in args.lf_sources if s != args.hifi]
    study_map = {k: v for k, v in study_map.items() if k in all_source_names}

    # Parse fidelity-db map
    fidelity_db_map = _parse_fidelity_db(args.fidelity_db)

    print(f"=== Experiment 13: Async MF Optimizer ===")
    print(f"  hifi:         {args.hifi}")
    print(f"  lf_sources:   {args.lf_sources}")
    print(f"  policy_mode:  {args.policy_mode}")
    print(f"  runtime_mode: {args.runtime_mode}")
    print(f"  fidelities:   {args.fidelity_levels}")
    print(f"  budget:       {args.budget_total}")
    print(f"  workers:      {args.max_workers}")
    print(f"  study_map:")
    for k, v in study_map.items():
        exists = Path(v).exists()
        print(f"    {k}: {v}  {'[OK]' if exists else '[NOT FOUND]'}")
    if fidelity_db_map:
        print(f"  fidelity_db_map: {fidelity_db_map}")

    out_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_RESULTS

    summary = run(
        hifi=args.hifi,
        lf_sources=args.lf_sources,
        study_map=study_map,
        policy_mode=args.policy_mode,
        executor_mode=args.executor_mode,
        runtime_mode=args.runtime_mode,
        fidelity_levels=args.fidelity_levels,
        budget_total=args.budget_total,
        max_workers=args.max_workers,
        allow_resume=args.allow_resume,
        target_tau=args.target_tau,
        n_bootstrap=args.n_bootstrap,
        explore_weight=args.explore_weight,
        seed=args.seed,
        save_dir=args.save_dir,
        output_dir=out_dir,
        snapshot_interval=args.snapshot_interval,
        fidelity_db_map=fidelity_db_map,
    )

    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
