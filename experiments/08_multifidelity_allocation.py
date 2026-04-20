"""
Experiment 08 — Multi-Fidelity Optimal Acquisition: Classical vs Hybrid.

Two branches are implemented and compared:

  classical (paired_cv)
  ─────────────────────
  Estimand:   per-policy HF mean score  mu_p = E_target[best_cd(HF, target, p)]
  Estimator:  control-variate (Peherstorfer et al. 2016)
                  mu_hat_p = mean(Y_hf, n_hf paired)
                           - alpha_p * (mean(Y_lf, n_hf paired)
                                        - mean(Y_lf, n_hf+n_lf all))
  alpha_p:    oracle (Cov(Y_hf,Y_lf) / Var(Y_lf) from all paired data)
  Ranking:    sort policies by mu_hat_p ascending (lower = better)
  Label:      "classical" iff databases share targets by experiment index;
              "paired_cv" otherwise (label stored in output JSON).
  Valid for:  LF sources generated with the same shared-targets file as HF
              (mixbox, km, ryb in TENKi-1000).

  hybrid
  ──────
  Estimand:   ranking quality (Kendall tau vs full HF ranking)
  Estimator:  ranking-oriented score fusion (heuristic, NOT textbook MFMC)
                  z_cv = z_hf + rho * z_lf
              where z_hf, z_lf are z-scored means from independent HF/LF samples.
  Ranking:    sort policies by z_cv ascending.
  Valid for:  any LF source (paired or unpaired); no pairing required.

Budget model (identical for both branches):
    B = n_HF + n_LF / r     (HF-equivalent units)
    n_LF = int((B - n_HF) * r)
    Optimal HF fraction from MFMC theory: n_HF ≈ B / (1 + ratio/r)
    where ratio = sqrt(r) * |rho| / sqrt(1 - rho^2)

Decision criteria (recorded in output JSON)
────────────────────────────────────────────
  If classical and hybrid agree on tau → result is robust.
  If hybrid > classical on tau but not on scalar MSE → ranking vs estimation tradeoff.
  If classical >= hybrid on tau → simplify narrative; hybrid is redundant.
"""

import sys
import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kendalltau, pearsonr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)

from extended.gamut_symmetry.analysis.flip_data import load_paired_data, PairedTargetData

_OUT_DEFAULT = Path(__file__).parent.parent / "results"

# ── Default study map ────────────────────────────────────────────────────────
# Paired LF sources: share targets with spectral by experiment index.
# Unpaired LF sources: different target spaces → hybrid branch only.

PAIRED_LF_SOURCES = {
    "mixbox": "output/db_mixbox",
    "km":     "output/db_km",
    "ryb":    "output/db_ryb",
}

HI_FI_DB    = "output/db_spectral"
HI_FI_NAME  = "spectral"
COST_RATIOS = [5, 10, 20, 50, 100, 200, 500, 1000]


# ── Shared utilities ─────────────────────────────────────────────────────────

def load_exp_scores(db_path: str) -> dict[str, list[float]]:
    """Load per-policy list of best_color_distance_mean (one per experiment)."""
    scores: dict[str, list[float]] = {}
    pd_ = os.path.join(_ROOT, db_path, "policies")
    if not os.path.exists(pd_):
        return scores
    for pol in sorted(os.listdir(pd_)):
        vals: list[float] = []
        for ed in sorted(glob.glob(os.path.join(pd_, pol, "experiment_*"))):
            s = os.path.join(ed, "summary.json")
            if os.path.exists(s):
                with open(s) as f:
                    d = json.load(f)
                v = d.get("policy_stats", {}).get("best_color_distance_mean")
                if v is not None:
                    vals.append(float(v))
        if vals:
            scores[pol] = vals
    return scores


def ktau(r1: list[str], r2: list[str]) -> float:
    common = [p for p in r1 if p in r2]
    if len(common) < 2:
        return float("nan")
    pa = {p: i for i, p in enumerate(r1)}
    pb = {p: i for i, p in enumerate(r2)}
    t, _ = kendalltau([pa[p] for p in common], [pb[p] for p in common])
    return float(t)


def pearson_rho(sc_a: dict[str, list[float]], sc_b: dict[str, list[float]]) -> float:
    common = sorted(set(sc_a) & set(sc_b))
    if len(common) < 2:
        return float("nan")
    a = [float(np.mean(sc_a[p])) for p in common]
    b = [float(np.mean(sc_b[p])) for p in common]
    r, _ = pearsonr(a, b)
    return float(r)


def mfmc_optimal_ratio(rho: float, r: float) -> float:
    """Optimal n_LF / n_HF from MFMC theory."""
    if abs(rho) >= 1.0:
        return float("inf")
    return float(np.sqrt(r) * abs(rho) / np.sqrt(1.0 - rho**2))


def budget_split(B: float, opt_ratio: float, r: float) -> tuple[int, int]:
    """
    Given total budget B and optimal ratio, compute (n_hf, n_lf).
    Returns (0, 0) if n_hf would be zero (budget too small for even one HF obs).
    """
    n_hf = int(B / (1.0 + opt_ratio / r))
    if n_hf == 0:
        return 0, 0
    n_lf = int((B - n_hf) * r)
    return n_hf, n_lf


def actual_cost(n_hf: int, n_lf: int, r: float) -> float:
    return float(n_hf + n_lf / r)


# ── Classical branch ─────────────────────────────────────────────────────────

def compute_oracle_alphas(paired: PairedTargetData) -> dict[str, float]:
    """
    Estimate alpha_p = Cov(Y_hf, Y_lf) / Var(Y_lf) from ALL paired observations.

    Using all available data for alpha estimation (oracle alpha) avoids the
    instability of estimating alpha from a small n_hf sample, which is standard
    practice in MFMC simulation studies.
    """
    alphas: dict[str, float] = {}
    for pol, arr in paired.records.items():
        hf = arr[:, 1]
        lf = arr[:, 2]
        var_lf = float(np.var(lf, ddof=1))
        if var_lf < 1e-12:
            alphas[pol] = 0.0
        else:
            cov = float(np.cov(hf, lf, ddof=1)[0, 1])
            alphas[pol] = cov / var_lf
    return alphas


def compute_true_mu_hf(paired: PairedTargetData) -> dict[str, float]:
    """Ground truth per-policy HF mean from all available paired observations."""
    return {pol: float(np.mean(arr[:, 1])) for pol, arr in paired.records.items()}


def compute_true_lf_mean(paired: PairedTargetData) -> dict[str, float]:
    """Full-pool LF mean per policy (used as the 'known' E[Y_lf] anchor)."""
    return {pol: float(np.mean(arr[:, 2])) for pol, arr in paired.records.items()}


def classical_step(
    paired: PairedTargetData,
    oracle_alphas: dict[str, float],
    n_hf: int,
    n_lf: int,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, float]]:
    """
    One bootstrap draw of the classical control-variate estimator.

    Sampling model
    ──────────────
    - n_hf target indices sampled WITH replacement → paired (Y_hf, Y_lf) observations.
    - n_lf additional target indices sampled WITH replacement → LF-only observations.
    - E[Y_lf] estimated from all n_hf + n_lf LF observations (paired + extra).

    Estimator (per policy p)
    ────────────────────────
        mu_hat_p = mean(Y_hf_paired)
                 - alpha_p * (mean(Y_lf_paired) - mean(Y_lf_all))

    Returns
    -------
    ranking : list[str]
        Policies sorted best→worst by mu_hat_p (ascending).
    mu_hats : dict[str, float]
        Per-policy point estimate of mu_HF.
    """
    common = sorted(set(paired.records) & set(oracle_alphas))
    if not common:
        return [], {}

    pool_size = paired.n_targets
    paired_idx = rng.integers(0, pool_size, size=n_hf)
    extra_idx   = rng.integers(0, pool_size, size=n_lf)

    mu_hats: dict[str, float] = {}
    for pol in common:
        arr = paired.records[pol]
        alpha = oracle_alphas[pol]

        hf_paired  = arr[paired_idx, 1]
        lf_paired  = arr[paired_idx, 2]
        lf_extra   = arr[extra_idx,  2] if n_lf > 0 else np.array([])

        mean_hf_paired = float(np.mean(hf_paired))
        mean_lf_paired = float(np.mean(lf_paired))
        if n_lf > 0:
            mean_lf_all = float(np.mean(np.concatenate([lf_paired, lf_extra])))
        else:
            mean_lf_all = mean_lf_paired

        mu_hats[pol] = mean_hf_paired - alpha * (mean_lf_paired - mean_lf_all)

    ranking = sorted(common, key=lambda p: mu_hats[p])   # ascending = better
    return ranking, mu_hats


# ── Hybrid branch ─────────────────────────────────────────────────────────────

def hybrid_step(
    hf_scores: dict[str, list[float]],
    lf_scores: dict[str, list[float]],
    rho: float,
    n_hf: int,
    n_lf: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    Ranking-oriented score fusion (heuristic — NOT textbook MFMC).

    Independent HF and LF samples are z-scored, then fused:
        z_cv = z_hf + rho * z_lf

    This is NOT an unbiased estimator of any scalar quantity; it is purely a
    ranking heuristic calibrated to maximize Kendall tau vs full HF ranking.

    Sign convention (lower-is-better):
        z_hf < 0 for a good policy; rho > 0 means LF ≈ HF in direction.
        Adding rho*z_lf reinforces the HF signal.
        Sanity: HF == LF, rho = 1 → z_cv = 2*z_hf → ranking preserved.
    """
    common = sorted(set(hf_scores) & set(lf_scores))
    if not common:
        return []

    hf_sample = {p: rng.choice(hf_scores[p], size=n_hf, replace=True) for p in common}
    lf_sample = {p: rng.choice(lf_scores[p], size=n_lf, replace=True) for p in common}

    hf_means = np.array([float(np.mean(hf_sample[p])) for p in common])
    lf_means = np.array([float(np.mean(lf_sample[p])) for p in common])

    def _zscore(v: np.ndarray) -> np.ndarray:
        s = float(v.std())
        return (v - v.mean()) / s if s > 1e-10 else np.zeros_like(v)

    z_cv = _zscore(hf_means) + rho * _zscore(lf_means)
    return [common[i] for i in np.argsort(z_cv)]   # ascending = best first


# ── Budget loop ───────────────────────────────────────────────────────────────

def run_budget_sweep(
    mode: str,
    paired: PairedTargetData | None,
    hf_scores: dict[str, list[float]],
    lf_scores: dict[str, list[float]],
    rho: float,
    hifi_full_rank: list[str],
    budgets: list[int],
    r: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """
    Run one branch (classical or hybrid) over the budget grid.

    Returns a results dict suitable for JSON serialisation.
    """
    opt_ratio = mfmc_optimal_ratio(rho, r)

    # Pre-compute oracle quantities for classical branch
    oracle_alphas: dict[str, float] = {}
    true_mu_hf:   dict[str, float] = {}
    if mode == "classical" and paired is not None:
        oracle_alphas = compute_oracle_alphas(paired)
        true_mu_hf    = compute_true_mu_hf(paired)

    out: dict = {
        "mode":       mode,
        "rho":        rho,
        "opt_ratio":  round(opt_ratio, 3),
        "budget":     budgets,
        "n_hf":       [],
        "n_lf":       [],
        "actual_cost": [],
        "tau_mean":   [],
        "tau_std":    [],
    }
    if mode == "classical":
        # per-policy mean and MSE of mu_hat across bootstrap iterations
        common_pols = sorted(set(paired.records) & set(oracle_alphas)) if paired else []
        out["mu_hat_mean"] = {p: [] for p in common_pols}
        out["mu_hat_mse"]  = {p: [] for p in common_pols}
        out["true_mu_hf"]  = {p: round(true_mu_hf.get(p, float("nan")), 6)
                               for p in common_pols}

    for B in budgets:
        n_hf, n_lf = budget_split(B, opt_ratio, r)
        cost = actual_cost(n_hf, n_lf, r)
        out["n_hf"].append(n_hf)
        out["n_lf"].append(n_lf)
        out["actual_cost"].append(round(cost, 4))

        if n_hf == 0:
            out["tau_mean"].append(float("nan"))
            out["tau_std"].append(float("nan"))
            if mode == "classical":
                for p in common_pols:
                    out["mu_hat_mean"][p].append(float("nan"))
                    out["mu_hat_mse"][p].append(float("nan"))
            continue

        taus: list[float] = []
        mu_hat_runs: dict[str, list[float]] = {p: [] for p in common_pols} \
            if mode == "classical" else {}

        for _ in range(n_bootstrap):
            if mode == "classical" and paired is not None:
                ranking, mu_hats = classical_step(paired, oracle_alphas, n_hf, n_lf, rng)
                for p in common_pols:
                    if p in mu_hats:
                        mu_hat_runs[p].append(mu_hats[p])
            else:
                ranking = hybrid_step(hf_scores, lf_scores, rho, n_hf, n_lf, rng)

            taus.append(ktau(ranking, hifi_full_rank))

        out["tau_mean"].append(round(float(np.mean(taus)), 6))
        out["tau_std"].append(round(float(np.std(taus)), 6))

        if mode == "classical":
            for p in common_pols:
                runs = mu_hat_runs[p]
                true_mu = true_mu_hf.get(p, float("nan"))
                out["mu_hat_mean"][p].append(round(float(np.mean(runs)), 6))
                mse = float(np.mean([(x - true_mu) ** 2 for x in runs]))
                out["mu_hat_mse"][p].append(round(mse, 8))

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    mode: str,
    lf_name: str,
    lf_db: str,
    output_dir: Path,
    budgets: list[int],
    r: float,
    n_bootstrap: int,
) -> None:

    print(f"[08] mode={mode}  lf={lf_name}  r={r}")

    # ── Load HF scores ────────────────────────────────────────────────────────
    print("  Loading HF scores ...")
    hf_scores = load_exp_scores(HI_FI_DB)
    hifi_full_rank = sorted(hf_scores, key=lambda p: float(np.mean(hf_scores[p])))
    print(f"    spectral: {len(hf_scores)} policies, "
          f"{min(len(v) for v in hf_scores.values())} experiments each")

    # ── Load LF scores ────────────────────────────────────────────────────────
    print(f"  Loading LF scores ({lf_name}) ...")
    lf_scores = load_exp_scores(lf_db)
    print(f"    {lf_name}: {len(lf_scores)} policies")

    rho = pearson_rho(hf_scores, lf_scores)
    print(f"  Pearson rho ({lf_name} -> spectral): {rho:.4f}")

    # ── Load paired data (classical branch) ───────────────────────────────────
    # The classical branch is valid ONLY for LF sources that were generated with
    # the same --shared-targets-file as the HF database (mixbox, km, ryb in
    # TENKi-1000).  Coincidental experiment-index overlap for non-shared-target
    # sources (study_*, hardness DBs, etc.) does NOT constitute paired data —
    # those databases have different target RGB values at the same index.
    #
    # Hard-block policy:
    #   --mode classical  with a non-paired source  → sys.exit(1)
    #   --mode compare    with a non-paired source  → skip classical, run hybrid only
    paired: PairedTargetData | None = None
    branch_label = "classical"   # overwritten to "hybrid-only" if classical skipped

    is_paired_source = (lf_db in PAIRED_LF_SOURCES.values()
                        or lf_name in PAIRED_LF_SOURCES)

    if mode in ("classical", "compare"):
        if not is_paired_source:
            msg = (
                f"ERROR: '{lf_name}' (db={lf_db}) is not a known shared-target "
                f"paired source.  The classical branch requires that HF and LF "
                f"databases were generated with the same --shared-targets-file so "
                f"that experiment indices map to identical target RGBs.\n"
                f"Known paired sources: {list(PAIRED_LF_SOURCES)}\n"
                f"To run with this source, use --mode hybrid."
            )
            if mode == "classical":
                print(msg)
                sys.exit(1)
            else:  # compare: skip classical, run hybrid only
                print(f"  INFO: {lf_name} is not a paired source; "
                      f"classical branch skipped (compare -> hybrid only).")
                mode = "hybrid"
        else:
            print(f"  Loading paired data ({HI_FI_NAME} <-> {lf_name}) ...")
            paired = load_paired_data(HI_FI_DB, lf_db,
                                      hf_name=HI_FI_NAME, lf_name=lf_name)
            if paired.n_targets == 0:
                # Shared-target source but no experiments found (missing DB?)
                msg = (
                    f"ERROR: '{lf_name}' is listed as a paired source but "
                    f"load_paired_data returned 0 targets.  Check that both "
                    f"'{HI_FI_DB}' and '{lf_db}' exist and are populated."
                )
                if mode == "classical":
                    print(msg)
                    sys.exit(1)
                else:
                    print(f"  WARNING: {msg}\n  classical branch skipped.")
                    paired = None
                    mode = "hybrid"
            else:
                print(f"    {paired.n_targets} paired targets, "
                      f"{len(paired.policies)} common policies")

    # ── Optimal ratio table ───────────────────────────────────────────────────
    print(f"\n  MFMC Optimal n_LF:n_HF  rho={rho:.4f}")
    print(f"  {'r':>6}  {'ratio':>8}  {'n_HF@B=10':>10}  {'n_LF@B=10':>10}")
    for r_ in COST_RATIOS:
        ratio_ = mfmc_optimal_ratio(rho, r_)
        nh, nl = budget_split(10, ratio_, r_)
        print(f"  {r_:>6}  {ratio_:>8.1f}  {nh:>10}  {nl:>10}")

    rng = np.random.default_rng(42)

    # ── Run branches ──────────────────────────────────────────────────────────
    results: dict = {
        "hifi":       HI_FI_NAME,
        "lf_source":  lf_name,
        "lf_db":      lf_db,
        "r":          r,
        "rho":        round(rho, 6),
        "opt_ratio":  round(mfmc_optimal_ratio(rho, r), 3),
        "hifi_full_rank": hifi_full_rank,
        "n_bootstrap": n_bootstrap,
    }

    if mode in ("classical", "compare") and paired is not None:
        oracle_alphas = compute_oracle_alphas(paired)
        results["oracle_alpha"] = {p: round(a, 6) for p, a in oracle_alphas.items()}
        results["true_mu_hf"] = {
            p: round(float(np.mean(paired.records[p][:, 1])), 6)
            for p in paired.policies
        }
        results["branch_label"] = branch_label

        print(f"\n  Oracle alpha per policy ({branch_label}):")
        for p, a in sorted(oracle_alphas.items()):
            true_mu = results["true_mu_hf"].get(p, float("nan"))
            print(f"    {p:<25}: alpha={a:.4f}  true_mu_hf={true_mu:.4f}")

    # Build the list of branches to execute.  "compare" runs both, but classical
    # is only included when paired data is actually available — paired is None
    # iff the code above already downgraded mode to "hybrid".
    run_modes: list[str] = []
    if mode == "compare":
        if paired is not None:
            run_modes = ["classical", "hybrid"]
        else:
            # Should not reach here (mode is already "hybrid" above), but guard
            # defensively so results["classical"] is never populated with hybrid data.
            run_modes = ["hybrid"]
    else:
        run_modes = [mode]

    print(f"\n  Budget sweep  n_bootstrap={n_bootstrap}  budgets={budgets}")

    for branch in run_modes:
        print(f"\n  === Branch: {branch} ===")
        print(f"  {'B':>5}  {'n_HF':>5}  {'n_LF':>6}  {'cost':>6}  "
              f"{'tau_mean':>9}  {'tau_std':>8}")

        br_paired = paired if branch == "classical" else None
        br_result = run_budget_sweep(
            mode=branch,
            paired=br_paired,
            hf_scores=hf_scores,
            lf_scores=lf_scores,
            rho=rho,
            hifi_full_rank=hifi_full_rank,
            budgets=budgets,
            r=r,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )

        for i, B in enumerate(budgets):
            print(f"  {B:>5}  {br_result['n_hf'][i]:>5}  {br_result['n_lf'][i]:>6}  "
                  f"{br_result['actual_cost'][i]:>6.2f}  "
                  f"{br_result['tau_mean'][i]:>9.3f}  "
                  f"{br_result['tau_std'][i]:>8.3f}")

        results[branch] = br_result

    # ── Comparison table (only when both branches actually ran) ──────────────
    if "classical" in results and "hybrid" in results:
        c_tau = results["classical"]["tau_mean"]
        h_tau = results["hybrid"]["tau_mean"]
        verdicts: list[str] = []
        tau_adv: list[float] = []   # classical - hybrid

        for c, h in zip(c_tau, h_tau):
            if any(np.isnan([c, h])):
                verdicts.append("insufficient_data")
                tau_adv.append(float("nan"))
            elif abs(c - h) < 0.01:
                verdicts.append("tied")
                tau_adv.append(round(c - h, 6))
            elif c > h:
                verdicts.append("classical_wins")
                tau_adv.append(round(c - h, 6))
            else:
                verdicts.append("hybrid_wins")
                tau_adv.append(round(c - h, 6))

        results["comparison"] = {
            "tau_advantage_classical_minus_hybrid": tau_adv,
            "verdict": verdicts,
        }

        print("\n  === Comparison (classical - hybrid tau) ===")
        print(f"  {'B':>5}  {'classical':>10}  {'hybrid':>8}  {'delta':>7}  verdict")
        for i, B in enumerate(budgets):
            print(f"  {B:>5}  {c_tau[i]:>10.3f}  {h_tau[i]:>8.3f}  "
                  f"{tau_adv[i]:>+7.3f}  {verdicts[i]}")

    # ── HF-only baseline ──────────────────────────────────────────────────────
    print("\n  === HF-only baseline ===")
    hf_only_taus: list[float] = []
    hf_only_nhf: list[int] = []

    for B in budgets:
        n_hf = B   # all budget on HF
        taus = [
            ktau(
                sorted(hf_scores, key=lambda p: float(
                    np.mean(rng.choice(hf_scores[p], n_hf, replace=True))
                )),
                hifi_full_rank,
            )
            for _ in range(n_bootstrap)
        ]
        hf_only_taus.append(round(float(np.mean(taus)), 6))
        hf_only_nhf.append(n_hf)
        print(f"  B={B:>3}: tau={hf_only_taus[-1]:.3f}")

    results["hf_only"] = {
        "budget":   budgets,
        "n_hf":     hf_only_nhf,
        "tau_mean": hf_only_taus,
    }

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Exp 08: Multi-Fidelity Allocation  (LF={lf_name}, rho={rho:.3f}, r={int(r)})",
                 fontsize=11)

    # Left: MFMC optimal ratio vs cost ratio
    ax = axes[0]
    r_range = np.logspace(0.5, 3, 120)
    ratios_ = [mfmc_optimal_ratio(rho, r_) for r_ in r_range]
    ax.plot(r_range, ratios_, "b-", linewidth=2, label=f"{lf_name} (rho={rho:.2f})")
    ax.axhline(100 / 3, color="black", linestyle=":", linewidth=1, label="100:3 reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Cost ratio  r = cost_HF / cost_LF")
    ax.set_ylabel("Optimal n_LF / n_HF")
    ax.set_title("MFMC: Optimal allocation ratio")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    # Right: tau vs budget
    ax2 = axes[1]
    ax2.plot(budgets, results["hf_only"]["tau_mean"], "b-o", linewidth=2, label="HF-only")

    branch_colors  = {"classical": "green",  "hybrid": "darkorange"}
    branch_markers = {"classical": "s",       "hybrid": "^"}
    branch_lines   = {"classical": "--",      "hybrid": ":"}
    for branch in run_modes:
        if branch in results:
            br = results[branch]
            label = f"{branch_label if branch == 'classical' else 'hybrid'} (LF={lf_name})"
            ax2.plot(budgets, br["tau_mean"],
                     color=branch_colors[branch],
                     linestyle=branch_lines[branch],
                     marker=branch_markers[branch],
                     label=label)

    ax2.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="tau=0.80")
    ax2.set_xlabel(f"Total budget B (HF-equiv units, r={int(r)})")
    ax2.set_ylabel("Kendall tau vs full spectral")
    ax2.set_title("Cost-efficiency: branches vs HF-only")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    png_path = output_dir / "multifidelity_allocation.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"\n[08] Saved {png_path}")

    # If classical branch ran, plot per-policy MSE
    if "classical" in results and "mu_hat_mse" in results["classical"]:
        fig2, ax3 = plt.subplots(figsize=(9, 5))
        mse_data = results["classical"]["mu_hat_mse"]
        for pol, vals in sorted(mse_data.items()):
            clean = [v if not np.isnan(v) else None for v in vals]
            ax3.plot(budgets, clean, marker="o", label=pol, markersize=4)
        ax3.set_xlabel(f"Budget B (r={int(r)})")
        ax3.set_ylabel("MSE of mu_hat vs true mu_HF")
        ax3.set_yscale("log")
        ax3.set_title(f"Classical branch: per-policy estimation MSE  (LF={lf_name})")
        ax3.legend(fontsize=7, ncol=2); ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        mse_png = output_dir / "multifidelity_mse.png"
        plt.savefig(mse_png, dpi=150)
        plt.close()
        print(f"[08] Saved {mse_png}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = output_dir / "multifidelity_allocation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[08] Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-fidelity allocation (Exp 08)")
    p.add_argument(
        "--mode", choices=["classical", "hybrid", "compare"],
        default="compare",
        help="Branch(es) to run. 'compare' runs both and produces a verdict table.",
    )
    p.add_argument(
        "--lf-source", default="mixbox", metavar="NAME",
        help="LF source name (must be a key in PAIRED_LF_SOURCES for classical branch).",
    )
    p.add_argument(
        "--lf-db", default=None, metavar="PATH",
        help="Override LF database path (relative to repo root).",
    )
    p.add_argument(
        "--r", type=float, default=100.0, metavar="FLOAT",
        help="Cost ratio cost_HF / cost_LF (default: 100).",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=300, metavar="INT",
        help="Bootstrap iterations per budget point (default: 300).",
    )
    p.add_argument(
        "--budgets", nargs="+", type=int,
        default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100],
        metavar="B",
        help="Budget grid in HF-equivalent units.",
    )
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Output directory (default: results/).",
    )
    p.add_argument(
        "--studies", nargs="+", metavar="NAME=PATH",
        help="Override study paths, e.g. mixbox=output/db_1000_mixbox.",
    )
    args = p.parse_args()

    if args.studies:
        for s in args.studies:
            name, _, path = s.partition("=")
            name = name.strip(); path = path.strip()
            PAIRED_LF_SOURCES[name] = path
            if name == "spectral":
                HI_FI_DB = path

    lf_db_ = args.lf_db if args.lf_db else PAIRED_LF_SOURCES.get(args.lf_source, "")
    if not lf_db_:
        p.error(f"Unknown --lf-source '{args.lf_source}'. "
                f"Known paired sources: {list(PAIRED_LF_SOURCES)}. "
                "Use --lf-db to specify a path directly.")

    out_dir = Path(args.output_dir) if args.output_dir else _OUT_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        mode=args.mode,
        lf_name=args.lf_source,
        lf_db=lf_db_,
        output_dir=out_dir,
        budgets=args.budgets,
        r=args.r,
        n_bootstrap=args.n_bootstrap,
    )
