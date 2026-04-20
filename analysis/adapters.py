"""
analysis/adapters.py — Generic adapters that convert non-PEGKi data into TENKi's
StudyScores format, enabling experiments 02–09 on any domain.

Background
----------
All nine PEGKi case studies share one abstract structure:

    policies  — the entities being ranked
                (ML optimisers, polymer grades, electron-type families, ML models …)
    engines   — the measurement modalities or fidelity levels
                (physics simulators, hardness scales, DFT functionals, databases …)
    score(policy, engine)  — performance of policy P under engine E

TENKi's StudyScores already captures this:
    study.name           = engine name
    study.policy_scores  = {policy: [score]}   # list of length ≥ 1

For external domains that have a single score per (policy, engine) pair, each
score is stored as a length-1 list.  Bootstrap sampling (for Kendall-tau CI)
will be degenerate (zero variance) but the full-data rankings and tau values
are computed correctly.  This is an acceptable limitation for tabular domain data.

Functions
---------
load_from_dict(name, scores, lower_is_better)
    Simplest entry point: {policy: float} → StudyScores

load_from_score_matrix(score_matrix, lower_is_better)
    {engine_name: {policy_name: float}} → dict[str, StudyScores]

load_from_csv(path, policy_col, engine_col, score_col, lower_is_better)
    Tidy CSV (one row per policy×engine observation) → dict[str, StudyScores]

load_from_wide_csv(path, lower_is_better)
    Wide CSV (rows=policies, cols=engines, values=scores) → dict[str, StudyScores]
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from analysis.flip_data import StudyScores


def load_from_dict(
    name: str,
    scores: dict[str, float],
    lower_is_better: bool = True,
) -> StudyScores:
    """
    Build one StudyScores from a {policy: score} dict.

    Each score is stored as a length-1 list so the bootstrap API treats it as
    a single experiment.  All ranking and tau computations are exact.

    Parameters
    ----------
    name:
        Engine / study name.
    scores:
        {policy_name: numeric_score}
    lower_is_better:
        True for error metrics (MAE, color_distance).
        False for quality metrics (hardness, band gap, efficiency).
    """
    if not scores:
        return StudyScores(
            name=name, db_path="", policy_scores={},
            full_rank=[], n_policies=0, max_n=0,
        )
    policy_scores = {p: [float(v)] for p, v in scores.items()}
    reverse = not lower_is_better
    full_rank = sorted(policy_scores, key=lambda p: policy_scores[p][0], reverse=reverse)
    return StudyScores(
        name=name,
        db_path="",
        policy_scores=policy_scores,
        full_rank=full_rank,
        n_policies=len(policy_scores),
        max_n=1,
    )


def load_from_score_matrix(
    score_matrix: dict[str, dict[str, float]],
    lower_is_better: bool = True,
) -> dict[str, StudyScores]:
    """
    Build one StudyScores per engine from a nested dict.

    Parameters
    ----------
    score_matrix:
        {engine_name: {policy_name: score}}
    lower_is_better:
        Direction convention applied uniformly to all engines.

    Returns
    -------
    {engine_name: StudyScores}
    """
    return {
        engine: load_from_dict(engine, policy_scores, lower_is_better)
        for engine, policy_scores in score_matrix.items()
    }


def load_from_csv(
    path: str,
    policy_col: str = "policy",
    engine_col: str = "engine",
    score_col:  str = "score",
    lower_is_better: bool = True,
) -> dict[str, StudyScores]:
    """
    Load from a tidy CSV with columns (policy, engine, score).

    The CSV may have extra columns; only policy_col, engine_col, score_col are used.

    Example CSV:
        policy,engine,score
        cgcnn,band_gap,0.28
        alignn,band_gap,0.14
        cgcnn,bulk_modulus,14.3
        alignn,bulk_modulus,10.1
    """
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            eng = row[engine_col].strip()
            pol = row[policy_col].strip()
            val = float(row[score_col])
            raw[eng][pol] = val
    return load_from_score_matrix(dict(raw), lower_is_better=lower_is_better)


def load_from_wide_csv(
    path: str,
    lower_is_better: bool = True,
    policy_col: str | None = None,
) -> dict[str, StudyScores]:
    """
    Load from a wide CSV where rows=policies and columns=engines.

    The first column is treated as the policy name (or specify via policy_col).

    Example CSV:
        policy,shore_a,shore_d,rockwell_r
        PTFE,55,55,58
        HDPE,97,60,60
        PC,100,80,118
    """
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        pcol = policy_col or headers[0]
        engine_cols = [h for h in headers if h != pcol]
        for row in reader:
            pol = row[pcol].strip()
            for eng in engine_cols:
                raw[eng][pol] = float(row[eng])
    return load_from_score_matrix(dict(raw), lower_is_better=lower_is_better)


def common_policies_across(studies: dict[str, StudyScores]) -> list[str]:
    """Return sorted list of policies present in ALL studies (re-exported for convenience)."""
    from analysis.flip_data import common_policy_subset
    return common_policy_subset(studies)
