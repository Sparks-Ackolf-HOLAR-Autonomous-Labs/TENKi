"""
analysis/flip_metrics.py — Ranking, Kendall tau, bootstrap CI, ceiling estimation.

Responsibilities
----------------
- Rank policies from a sub-sampled set of experiments.
- Compute Kendall tau (and optionally Spearman rho) between two ranked lists.
- Bootstrap tau curves with full percentile output.
- Compute full-data ceiling tau between two studies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau as _scipy_ktau
from scipy.stats import spearmanr as _scipy_spearman

if TYPE_CHECKING:
    from analysis.flip_data import StudyScores


def rank_from_sample(
    study: "StudyScores",
    policies: list[str],
    n: int,
    rng: np.random.Generator,
    replace: bool = True,
) -> list[str]:
    """
    Sub-sample ``n`` experiments per policy from ``study`` and return policies
    ranked best→worst (lower score = better).

    Capped at ``len(policy_scores[p])`` to avoid over-sampling when
    ``replace=False``.
    """
    means = {
        p: float(
            np.mean(
                rng.choice(
                    study.policy_scores[p],
                    size=min(n, len(study.policy_scores[p])),
                    replace=replace,
                )
            )
        )
        for p in policies
        if p in study.policy_scores
    }
    return sorted(means, key=means.__getitem__)


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    """
    Kendall tau between two ranked lists, restricted to policies in both lists.

    Returns NaN if fewer than 2 common policies exist.
    """
    common = [p for p in rank_a if p in rank_b]
    if len(common) < 2:
        return float("nan")
    pos_a = {p: i for i, p in enumerate(rank_a)}
    pos_b = {p: i for i, p in enumerate(rank_b)}
    tau, _ = _scipy_ktau([pos_a[p] for p in common], [pos_b[p] for p in common])
    return float(tau)


def spearman_rho(rank_a: list[str], rank_b: list[str]) -> float:
    """
    Spearman rho between two ranked lists, restricted to common policies.

    Returns NaN if fewer than 2 common policies exist.
    """
    common = [p for p in rank_a if p in rank_b]
    if len(common) < 2:
        return float("nan")
    pos_a = {p: i for i, p in enumerate(rank_a)}
    pos_b = {p: i for i, p in enumerate(rank_b)}
    rho, _ = _scipy_spearman([pos_a[p] for p in common], [pos_b[p] for p in common])
    return float(rho)


def bootstrap_tau_curve(
    source: "StudyScores",
    reference_rank: list[str],
    policies: list[str],
    n_values: list[int],
    n_bootstrap: int,
    rng_seed: int,
) -> dict[int, dict[str, float]]:
    """
    For each ``N`` in ``n_values``, draw ``n_bootstrap`` random sub-samples
    of ``N`` experiments from ``source`` and compute Kendall tau vs
    ``reference_rank``.

    Parameters
    ----------
    source:
        The study being evaluated.
    reference_rank:
        Fixed reference ranking (the "ground truth" ranking to compare against).
    policies:
        Common policy subset to use for ranking.
    n_values:
        List of N values to evaluate.
    n_bootstrap:
        Number of bootstrap samples per N.
    rng_seed:
        Seed for the random number generator (determinism).

    Returns
    -------
    dict mapping N -> bootstrap statistics dict with keys:
        mean_tau, std_tau, p05_tau, p50_tau, p95_tau, n_bootstrap
    """
    rng = np.random.default_rng(rng_seed)
    results: dict[int, dict[str, float]] = {}
    for n in n_values:
        taus: list[float] = []
        for _ in range(n_bootstrap):
            sampled_rank = rank_from_sample(source, policies, n, rng)
            tau = kendall_tau(sampled_rank, reference_rank)
            if not np.isnan(tau):
                taus.append(tau)
        if taus:
            results[n] = {
                "mean_tau":   float(np.mean(taus)),
                "std_tau":    float(np.std(taus)),
                "p05_tau":    float(np.percentile(taus, 5)),
                "p50_tau":    float(np.percentile(taus, 50)),
                "p95_tau":    float(np.percentile(taus, 95)),
                "n_bootstrap": n_bootstrap,
            }
    return results


def full_data_ceiling(
    source: "StudyScores",
    reference: "StudyScores",
    policies: list[str],
) -> float:
    """
    Kendall tau between source's full-data ranking and reference's full-data
    ranking, restricted to ``policies``.

    This is the asymptotic upper bound: the ranking agreement achievable when
    both sources use all available experiments.
    """
    src_rank = [p for p in source.full_rank if p in policies]
    ref_rank = [p for p in reference.full_rank if p in policies]
    return kendall_tau(src_rank, ref_rank)
