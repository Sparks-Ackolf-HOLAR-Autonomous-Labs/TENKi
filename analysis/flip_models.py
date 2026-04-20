"""
analysis/flip_models.py — Donor-flip logic: asymmetry, feasibility, FlipResult objects.

Responsibilities
----------------
- Compute directional asymmetry at N=1 for all ordered study pairs.
- Compute external flip feasibility and N* (Scenario 1: HF reference).
- Compute mutual flip feasibility and N* (Scenario 2: mutual reference).
- Classify each pair into one of five verdicts.

Verdict semantics
-----------------
ALREADY_DONOR          source is ahead of competitor at N=1 (gap_now > eps)
FLIPPABLE              source starts behind but crosses within tested N_values
PERMANENT_GAP          source full-data ceiling is below competitor ceiling (external mode)
UNRESOLVED_IN_RANGE    ceiling allows a flip but N_values do not reach the crossover
NEAR_SYMMETRIC         absolute gap at ceiling is below eps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from analysis.flip_metrics import (
    bootstrap_tau_curve,
    full_data_ceiling,
    kendall_tau,
    rank_from_sample,
)

if TYPE_CHECKING:
    from analysis.flip_data import StudyScores


@dataclass
class FlipResult:
    source: str
    competitor: str
    reference: str
    mode: str                           # "external" | "mutual"
    tau_source_at_1: float
    tau_competitor_at_1: float | None
    ceiling_source: float
    ceiling_competitor: float
    gap_now: float                      # tau_source_at_1 - tau_competitor_at_1
    gap_ceiling: float                  # ceiling_source  - ceiling_competitor
    flip_possible: bool
    flip_n: int | None                  # N* (first N where source crosses competitor)
    verdict: str


def asymmetry_at_n1(
    studies: dict[str, "StudyScores"],
    policies: list[str],
    n_bootstrap: int,
    rng_seed: int,
) -> dict[tuple[str, str], float]:
    """
    Compute directional asymmetry A[i,j] = tau_ij(N=1) − tau_ji(N=1) for all
    ordered pairs.

    A positive value means ``i`` is the donor at N=1: one experiment from ``i``
    produces a ranking that is more faithful to ``j``'s full-data ranking than
    one experiment from ``j`` is to ``i``'s full-data ranking.

    Returns
    -------
    dict[(src, ref)] -> asymmetry value
    """
    names = list(studies.keys())
    rng = np.random.default_rng(rng_seed)

    # Bootstrap mean tau at N=1 for every ordered pair
    tau_n1: dict[tuple[str, str], float] = {}
    for src_name in names:
        src = studies[src_name]
        for ref_name in names:
            if src_name == ref_name:
                continue
            ref_rank = [p for p in studies[ref_name].full_rank if p in policies]
            taus: list[float] = []
            for _ in range(n_bootstrap):
                sampled = rank_from_sample(src, policies, 1, rng)
                taus.append(kendall_tau(sampled, ref_rank))
            tau_n1[(src_name, ref_name)] = float(np.nanmean(taus))

    asym: dict[tuple[str, str], float] = {}
    for src_name in names:
        for ref_name in names:
            if src_name == ref_name:
                continue
            asym[(src_name, ref_name)] = (
                tau_n1[(src_name, ref_name)] - tau_n1[(ref_name, src_name)]
            )
    return asym


def external_flip_result(
    source: "StudyScores",
    competitor: "StudyScores",
    hifi: "StudyScores",
    policies: list[str],
    n_values: list[int],
    n_bootstrap: int,
    eps: float,
    rng_seed: int,
) -> FlipResult:
    """
    Scenario 1 — external-reference flip.

    Question: can ``source`` (using N experiments) produce a ranking that agrees
    with ``hifi`` better than ``competitor``'s full-data ceiling does?

    N*_ext = min N such that tau_AH(N) > ceiling(B → H) + eps

    If ceiling_source <= ceiling_competitor + eps, the flip is PERMANENTLY
    impossible: a physics / calibration limit prevents ``source`` from ever
    agreeing with ``hifi`` as well as ``competitor`` does.
    """
    hifi_rank = [p for p in hifi.full_rank if p in policies]

    ceiling_src  = full_data_ceiling(source,     hifi, policies)
    ceiling_comp = full_data_ceiling(competitor, hifi, policies)
    gap_ceiling  = ceiling_src - ceiling_comp

    src_curve  = bootstrap_tau_curve(
        source, hifi_rank, policies, n_values, n_bootstrap, rng_seed
    )
    comp_curve = bootstrap_tau_curve(
        competitor, hifi_rank, policies, n_values, n_bootstrap, rng_seed + 1
    )

    tau_src_at_1  = src_curve.get(n_values[0], {}).get("mean_tau", float("nan"))
    tau_comp_at_1 = comp_curve.get(n_values[0], {}).get("mean_tau", float("nan"))
    gap_now = (
        tau_src_at_1 - tau_comp_at_1
        if not (np.isnan(tau_src_at_1) or np.isnan(tau_comp_at_1))
        else float("nan")
    )

    # Find N* (first N where source's mean tau exceeds competitor's ceiling)
    flip_n: int | None = None
    for n in n_values:
        mean_src = src_curve.get(n, {}).get("mean_tau", float("nan"))
        if not np.isnan(mean_src) and mean_src > ceiling_comp + eps:
            flip_n = n
            break

    # Verdict
    if not np.isnan(gap_now) and gap_now > eps:
        verdict = "ALREADY_DONOR"
    elif abs(gap_ceiling) <= eps:
        verdict = "NEAR_SYMMETRIC"
    elif gap_ceiling < -eps:
        verdict = "PERMANENT_GAP"
    elif flip_n is not None:
        verdict = "FLIPPABLE"
    else:
        verdict = "UNRESOLVED_IN_RANGE"

    flip_possible = verdict in ("ALREADY_DONOR", "FLIPPABLE")
    # flip_n is only meaningful for FLIPPABLE; clear it for other verdicts
    if verdict != "FLIPPABLE":
        flip_n = None

    return FlipResult(
        source=source.name,
        competitor=competitor.name,
        reference=hifi.name,
        mode="external",
        tau_source_at_1=tau_src_at_1,
        tau_competitor_at_1=tau_comp_at_1,
        ceiling_source=ceiling_src,
        ceiling_competitor=ceiling_comp,
        gap_now=gap_now,
        gap_ceiling=gap_ceiling,
        flip_possible=flip_possible,
        flip_n=flip_n,
        verdict=verdict,
    )


def mutual_flip_result(
    source: "StudyScores",
    competitor: "StudyScores",
    policies: list[str],
    n_values: list[int],
    n_bootstrap: int,
    eps: float,
    rng_seed: int,
) -> FlipResult:
    """
    Scenario 2 — mutual-reference flip.

    Question: how many experiments from ``source`` are needed so that
    ``source``'s ranking agrees with ``competitor``'s full-data ranking better
    than ``competitor``'s single experiment agrees with ``source``'s full-data
    ranking?

    N*_mut = min N such that tau_AB(N) > tau_BA(1) + eps

    Because the shared ceiling is the same for both sides, a finite N* always
    exists unless the pair is nearly symmetric.
    """
    comp_rank = [p for p in competitor.full_rank if p in policies]
    src_rank  = [p for p in source.full_rank     if p in policies]

    ceiling_src  = full_data_ceiling(source,     competitor, policies)
    ceiling_comp = full_data_ceiling(competitor, source,     policies)
    # Ceilings should be equal (Kendall tau symmetric); average for robustness
    shared_ceiling = (ceiling_src + ceiling_comp) / 2.0

    src_curve = bootstrap_tau_curve(
        source, comp_rank, policies, n_values, n_bootstrap, rng_seed
    )
    comp_curve = bootstrap_tau_curve(
        competitor, src_rank, policies, n_values, n_bootstrap, rng_seed + 1
    )

    tau_src_at_1  = src_curve.get(n_values[0], {}).get("mean_tau", float("nan"))
    tau_comp_at_1 = comp_curve.get(n_values[0], {}).get("mean_tau", float("nan"))
    gap_now = (
        tau_src_at_1 - tau_comp_at_1
        if not (np.isnan(tau_src_at_1) or np.isnan(tau_comp_at_1))
        else float("nan")
    )

    # N* where source's mean tau first surpasses competitor's single-experiment tau
    flip_n: int | None = None
    for n in n_values:
        mean_src = src_curve.get(n, {}).get("mean_tau", float("nan"))
        if not np.isnan(mean_src) and not np.isnan(tau_comp_at_1):
            if mean_src > tau_comp_at_1 + eps:
                flip_n = n
                break

    gap_ceiling = ceiling_src - ceiling_comp  # should be ~0

    if not np.isnan(gap_now) and gap_now > eps:
        verdict = "ALREADY_DONOR"
    elif not np.isnan(gap_now) and abs(gap_now) <= eps:
        verdict = "NEAR_SYMMETRIC"
    elif flip_n is not None:
        verdict = "FLIPPABLE"
    else:
        verdict = "UNRESOLVED_IN_RANGE"

    flip_possible = verdict in ("ALREADY_DONOR", "FLIPPABLE")
    # flip_n is only meaningful for FLIPPABLE; clear it for other verdicts
    if verdict != "FLIPPABLE":
        flip_n = None

    return FlipResult(
        source=source.name,
        competitor=competitor.name,
        reference=competitor.name,
        mode="mutual",
        tau_source_at_1=tau_src_at_1,
        tau_competitor_at_1=tau_comp_at_1,
        ceiling_source=shared_ceiling,
        ceiling_competitor=shared_ceiling,
        gap_now=gap_now,
        gap_ceiling=gap_ceiling,
        flip_possible=flip_possible,
        flip_n=flip_n,
        verdict=verdict,
    )
