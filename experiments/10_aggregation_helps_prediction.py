"""
Experiment 10 -- Does Aggregation Help Prediction?

Direct replication of the d3dd00207a question inside TENKi's framework.

The paper asks: does adding data from a large generic repository to a small
domain-specific training set improve or degrade ML model accuracy?
Key finding: naive concatenation degrades classical ML (Ridge, RF) while
deep learning models are more robust.

TENKi analogue
--------------
- Training data:  (target_rgb, best_action) pairs read from round JSON files.
  best_action = the action that achieved the lowest color_distance for that
  target in that database.
- Model input:    target_rgb  (3D, normalised 0->1)
- Model output:   (red%, yellow%, blue%)  (3D action space)
- Evaluation:     action MAE on held-out SPECTRAL test targets vs the oracle
                  spectral best action.  Also Kendall tau of the implied
                  policy ranking.

Three model archetypes (no imports from parent project):
  Ridge            -- classical linear ML   (expect degradation from paper)
  RandomForest     -- classical nonlinear ML (expect moderate degradation)
  GradientBoosting -- "deep learning" proxy  (expect more robustness)

Four training conditions for each model x LF source combination:
  spectral_only     -- train on spectral training split only
  concat            -- spectral_train + all LF data (naive concatenation)
  weighted          -- spectral_train + LF data weighted by RGB diversity
  lf_only           -- train on LF source only (pure transfer baseline)

Outputs
-------
results/aggregation_helps_prediction.json
results/aggregation_helps_prediction.png   -- MAE grid (model x condition)
results/aggregation_helps_prediction.md    -- summary with verdict per model type
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
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import kendalltau as _scipy_ktau
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from analysis.flip_data import (
    load_trial_records,
    best_action_per_target,
    TrialRecord,
)

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

HI_FI        = "spectral"
TEST_FRAC    = 0.30      # fraction of spectral targets held out for test
DIVERSITY_K  = 5         # nearest neighbours used for diversity weighting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_xy(
    best: dict[tuple, tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert best_action_per_target dict -> (X, y) arrays."""
    targets = sorted(best.keys())
    X = np.array([[t[0] / 255.0, t[1] / 255.0, t[2] / 255.0] for t in targets])
    y = np.array([[best[t][0][0], best[t][0][1], best[t][0][2]] for t in targets])
    return X, y


def _action_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean absolute error over all action components and all test targets."""
    return float(np.mean(np.abs(y_pred - y_true)))


def _ranking_tau(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    cd_true: np.ndarray,
) -> float:
    """
    Kendall tau between:
      - ranking implied by predicted action quality (L1 to oracle)
      - ranking implied by true spectral color_distance (oracle ranking)

    A better predicted action -> lower L1 -> better implied rank.
    We rank n test policies; here n = number of test targets, and each
    target acts as a "policy stand-in" whose score is its color_distance.
    """
    pred_error = np.mean(np.abs(y_pred - y_true), axis=1)  # per-target action error
    # lower pred_error = model thinks this target is easy to solve
    # lower cd_true    = spectral says this target is easy to solve
    if len(pred_error) < 2:
        return float("nan")
    tau, _ = _scipy_ktau(
        np.argsort(pred_error),
        np.argsort(cd_true),
    )
    return float(tau)


def _diversity_weights(
    X_lf: np.ndarray,
    X_ref: np.ndarray,
    k: int = DIVERSITY_K,
) -> np.ndarray:
    """
    For each LF sample compute its mean distance to the k nearest spectral
    training samples.  Samples far from spectral training data get higher
    weight (diversity-prioritised).  Weights are normalised to sum=1.
    """
    if len(X_ref) == 0 or len(X_lf) == 0:
        return np.ones(len(X_lf)) / len(X_lf)
    # pairwise L2 distances: (n_lf, n_ref)
    diff   = X_lf[:, None, :] - X_ref[None, :, :]   # (n_lf, n_ref, 3)
    dists  = np.sqrt((diff ** 2).sum(axis=2))        # (n_lf, n_ref)
    k_eff  = min(k, dists.shape[1])
    knn    = np.sort(dists, axis=1)[:, :k_eff]
    mean_d = knn.mean(axis=1)                        # (n_lf,)
    mean_d = mean_d + 1e-9                           # avoid div-by-zero
    w = mean_d / mean_d.sum()
    return w


def _make_models() -> dict[str, object]:
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                n_jobs=1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
                n_jobs=1,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    studies: dict[str, str] | None = None,
    hifi: str = HI_FI,
    test_frac: float = TEST_FRAC,
    max_experiments: int | None = None,
    seed: int = 42,
) -> None:
    study_map = studies or DEFAULT_STUDIES
    rng = np.random.default_rng(seed)

    # -- Load trial-level data ------------------------------------------------
    print("[10] Loading trial-level data from round JSON files...")
    all_records: dict[str, list[TrialRecord]] = {}
    for name, db_path in study_map.items():
        recs = load_trial_records(db_path, source_name=name,
                                  max_experiments=max_experiments)
        if recs:
            all_records[name] = recs
            print(f"  {name}: {len(recs):,} trials from "
                  f"{len({r.target_rgb for r in recs}):,} unique targets")
        else:
            print(f"  {name}: NOT FOUND")

    if hifi not in all_records:
        print(f"[10] HF source '{hifi}' not found -- aborting.")
        return

    # -- Build per-source oracle datasets ------------------------------------
    print("\n[10] Building oracle (best action per target) datasets...")
    oracle: dict[str, dict] = {}
    for name, recs in all_records.items():
        oracle[name] = best_action_per_target(recs)
        print(f"  {name}: {len(oracle[name]):,} unique targets with oracle action")

    # -- Spectral train / test split ------------------------------------------
    hifi_targets = sorted(oracle[hifi].keys())
    n_test = max(2, int(len(hifi_targets) * test_frac))
    idx    = rng.permutation(len(hifi_targets))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    hifi_train_targets = [hifi_targets[i] for i in sorted(train_idx)]
    hifi_test_targets  = [hifi_targets[i] for i in sorted(test_idx)]

    # Oracle X/y for train and test
    X_hifi_train = np.array([[t[0]/255, t[1]/255, t[2]/255] for t in hifi_train_targets])
    y_hifi_train = np.array([oracle[hifi][t][0] for t in hifi_train_targets])

    X_hifi_test  = np.array([[t[0]/255, t[1]/255, t[2]/255] for t in hifi_test_targets])
    y_hifi_test  = np.array([oracle[hifi][t][0] for t in hifi_test_targets])
    cd_hifi_test = np.array([oracle[hifi][t][1] for t in hifi_test_targets])

    print(f"\n  Spectral train: {len(hifi_train_targets)} targets  "
          f"test: {len(hifi_test_targets)} targets")

    lf_sources = [s for s in all_records if s != hifi]
    if not lf_sources:
        print("[10] No LF sources found -- aborting.")
        return

    # -- Results container ----------------------------------------------------
    # results[model_name][condition][lf_source] = {mae, tau}
    results: dict[str, dict[str, dict[str, dict]]] = {}
    model_names = ["Ridge", "RandomForest", "GradientBoosting"]
    conditions  = ["spectral_only", "concat", "weighted", "lf_only"]

    for model_name in model_names:
        results[model_name] = {c: {} for c in conditions}

    print("\n" + "=" * 65)
    print("Training and evaluating models")
    print("=" * 65)

    for lf_name in lf_sources:
        # LF source X/y
        lf_targets = sorted(oracle[lf_name].keys())
        X_lf = np.array([[t[0]/255, t[1]/255, t[2]/255] for t in lf_targets])
        y_lf = np.array([oracle[lf_name][t][0] for t in lf_targets])

        # Diversity weights for LF samples (relative to spectral train)
        w_lf = _diversity_weights(X_lf, X_hifi_train)

        print(f"\n  LF source: {lf_name}  ({len(lf_targets):,} targets)")

        for model_name in model_names:
            row: dict[str, dict] = {}

            # -- Condition 1: spectral_only -----------------------------------
            m = _make_models()[model_name]
            m.fit(X_hifi_train, y_hifi_train)
            y_pred = m.predict(X_hifi_test)
            row["spectral_only"] = {
                "mae": _action_mae(y_pred, y_hifi_test),
                "tau": _ranking_tau(y_pred, y_hifi_test, cd_hifi_test),
                "n_train": len(X_hifi_train),
            }

            # -- Condition 2: concat (naive concatenation) -----------------
            X_cat = np.vstack([X_hifi_train, X_lf])
            y_cat = np.vstack([y_hifi_train, y_lf])
            m = _make_models()[model_name]
            m.fit(X_cat, y_cat)
            y_pred = m.predict(X_hifi_test)
            row["concat"] = {
                "mae": _action_mae(y_pred, y_hifi_test),
                "tau": _ranking_tau(y_pred, y_hifi_test, cd_hifi_test),
                "n_train": len(X_cat),
            }

            # -- Condition 3: weighted (diversity-aware LF selection) ------
            # Sample LF points proportionally to diversity weight
            n_lf_draw = min(len(X_lf), len(X_hifi_train))
            lf_idx = rng.choice(len(X_lf), size=n_lf_draw, replace=False, p=w_lf)
            X_wlf  = np.vstack([X_hifi_train, X_lf[lf_idx]])
            y_wlf  = np.vstack([y_hifi_train, y_lf[lf_idx]])
            m = _make_models()[model_name]
            m.fit(X_wlf, y_wlf)
            y_pred = m.predict(X_hifi_test)
            row["weighted"] = {
                "mae": _action_mae(y_pred, y_hifi_test),
                "tau": _ranking_tau(y_pred, y_hifi_test, cd_hifi_test),
                "n_train": len(X_wlf),
            }

            # -- Condition 4: lf_only (pure transfer baseline) ------------
            m = _make_models()[model_name]
            m.fit(X_lf, y_lf)
            y_pred = m.predict(X_hifi_test)
            row["lf_only"] = {
                "mae": _action_mae(y_pred, y_hifi_test),
                "tau": _ranking_tau(y_pred, y_hifi_test, cd_hifi_test),
                "n_train": len(X_lf),
            }

            for cond, stats in row.items():
                results[model_name][cond][lf_name] = stats

            # Print row
            spec_mae = row["spectral_only"]["mae"]
            print(
                f"    {model_name:<18}  "
                + "  ".join(
                    f"{c}={row[c]['mae']:.3f}"
                    + ("^" if row[c]["mae"] > spec_mae + 0.5 else
                       "v" if row[c]["mae"] < spec_mae - 0.5 else "~")
                    for c in conditions
                )
            )

    # -- Verdict per model type ------------------------------------------------
    print("\n=== Verdicts (does concatenation help/hurt vs spectral_only?) ===")
    verdicts: dict[str, str] = {}
    for model_name in model_names:
        deltas = []
        for lf_name in lf_sources:
            spec = results[model_name]["spectral_only"].get(lf_name, {}).get("mae")
            cat  = results[model_name]["concat"].get(lf_name, {}).get("mae")
            if spec is not None and cat is not None:
                deltas.append(cat - spec)
        if deltas:
            mean_delta = float(np.mean(deltas))
            verdict = (
                "DEGRADES  (concat hurts)"      if mean_delta >  0.5 else
                "IMPROVES  (concat helps)"       if mean_delta < -0.5 else
                "NEUTRAL   (no clear effect)"
            )
            verdicts[model_name] = verdict
            print(f"  {model_name:<18}: mean Deltamae={mean_delta:+.3f}  {verdict}")

    # -- Plot -----------------------------------------------------------------
    _plot_results(results, model_names, conditions, lf_sources, verdicts)

    # -- JSON -----------------------------------------------------------------
    out_data = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hifi": hifi,
            "test_frac": test_frac,
            "n_hifi_train": len(hifi_train_targets),
            "n_hifi_test":  len(hifi_test_targets),
            "seed": seed,
            "lf_sources": lf_sources,
        },
        "verdicts": verdicts,
        "results": results,
    }
    out_path = _OUT / "aggregation_helps_prediction.json"
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"\n[10] Saved {out_path}")

    _write_markdown(results, model_names, conditions, lf_sources, verdicts, hifi,
                    len(hifi_train_targets), len(hifi_test_targets))


# ---------------------------------------------------------------------------
# Plot and report helpers
# ---------------------------------------------------------------------------

def _plot_results(
    results: dict,
    model_names: list[str],
    conditions: list[str],
    lf_sources: list[str],
    verdicts: dict[str, str],
) -> None:
    """
    One subplot per LF source.  Each subplot shows action MAE for each
    (model x condition) combination.
    """
    n_lf = len(lf_sources)
    if n_lf == 0:
        return

    fig, axes = plt.subplots(
        1, n_lf, figsize=(max(7, 4 * n_lf), 5), squeeze=False
    )

    cond_colors = {
        "spectral_only": "steelblue",
        "concat":        "tomato",
        "weighted":      "goldenrod",
        "lf_only":       "lightgray",
    }
    x = np.arange(len(model_names))
    width = 0.18

    for col, lf_name in enumerate(lf_sources):
        ax = axes[0][col]
        for ci, cond in enumerate(conditions):
            maes = [
                results[m][cond].get(lf_name, {}).get("mae", float("nan"))
                for m in model_names
            ]
            offset = (ci - len(conditions) / 2 + 0.5) * width
            ax.bar(
                x + offset, maes, width,
                label=cond, color=cond_colors[cond], alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Action MAE (%)" if col == 0 else "")
        ax.set_title(f"LF source: {lf_name}", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Aggregation effect on action prediction MAE\n"
        "(lower = better;  red=concat, blue=spectral_only)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(_OUT / "aggregation_helps_prediction.png", dpi=150)
    plt.close()
    print("[10] Saved aggregation_helps_prediction.png")


def _write_markdown(
    results: dict,
    model_names: list[str],
    conditions: list[str],
    lf_sources: list[str],
    verdicts: dict[str, str],
    hifi: str,
    n_train: int,
    n_test: int,
) -> None:
    lines = [
        "# Aggregation Helps Prediction? (Experiment 10)\n",
        f"HF reference: **{hifi}**  \n",
        f"Train/test split: {n_train} / {n_test} spectral targets  \n\n",
        "Replicates the core question from Vriza et al. (2024, d3dd00207a) "
        "inside TENKi's framework:\n",
        "> Does adding low-fidelity training data to a high-fidelity training set "
        "improve or degrade action prediction accuracy on held-out spectral targets?\n\n",
        "## Verdicts by model type\n",
        "| Model | Mean DeltaMAE (concat - spectral_only) | Verdict |",
        "|-------|-----------------------------------|---------|",
    ]
    for model_name in model_names:
        deltas = []
        for lf_name in lf_sources:
            spec = results[model_name]["spectral_only"].get(lf_name, {}).get("mae")
            cat  = results[model_name]["concat"].get(lf_name, {}).get("mae")
            if spec is not None and cat is not None:
                deltas.append(cat - spec)
        if deltas:
            mean_delta = float(np.mean(deltas))
            lines.append(
                f"| {model_name} | {mean_delta:+.3f} | {verdicts.get(model_name, '--')} |"
            )
    lines.append("")

    lines.append("## Per-source MAE table\n")
    for lf_name in lf_sources:
        lines.append(f"### LF source: {lf_name}\n")
        lines.append("| Model | spectral_only | concat | weighted | lf_only | Delta(concat-spec) |")
        lines.append("|-------|--------------|--------|----------|---------|----------------|")
        for model_name in model_names:
            row = results[model_name]
            def _m(c):
                v = row[c].get(lf_name, {}).get("mae")
                return f"{v:.3f}" if v is not None else "--"
            spec_v = row["spectral_only"].get(lf_name, {}).get("mae")
            cat_v  = row["concat"].get(lf_name, {}).get("mae")
            delta  = f"{cat_v - spec_v:+.3f}" if (spec_v and cat_v) else "--"
            lines.append(
                f"| {model_name} | {_m('spectral_only')} | {_m('concat')} | "
                f"{_m('weighted')} | {_m('lf_only')} | {delta} |"
            )
        lines.append("")

    lines.append(
        "## Interpretation\n\n"
        "- **DEGRADES**: Naive concatenation raises action MAE -- the LF source's "
        "action patterns conflict with spectral physics. Matches the paper's finding "
        "for classical ML.\n"
        "- **IMPROVES**: Concatenation lowers action MAE -- the LF source covers "
        "parts of the action space the spectral training set missed.\n"
        "- **NEUTRAL**: No statistically meaningful change. Common for sources with "
        "high Kendall tau ceiling vs spectral (exp 03).\n\n"
        "Cross-reference with `flip_feasibility.json` external_ceilings: sources "
        "with low ceiling tend to DEGRADE; sources with high ceiling tend to NEUTRAL "
        "or IMPROVE.\n"
    )

    md_path = _OUT / "aggregation_helps_prediction.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[10] Saved {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Does aggregation help prediction? (Exp 10)"
    )
    p.add_argument("--hifi",             default=HI_FI)
    p.add_argument("--test-frac",        type=float, default=TEST_FRAC)
    p.add_argument("--max-experiments",  type=int,   default=None,
                   help="Limit experiments loaded per policy per source (speeds up dev runs)")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--studies",          nargs="+",  metavar="NAME=PATH",
                   help="Studies as name=db_path pairs")
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
        test_frac=args.test_frac,
        max_experiments=args.max_experiments,
        seed=args.seed,
    )
