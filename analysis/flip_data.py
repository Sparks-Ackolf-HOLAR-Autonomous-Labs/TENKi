"""
analysis/flip_data.py — Data loading and score normalization for the donor-flip pipeline.

Responsibilities
----------------
- Load per-policy per-experiment scores from a PEGKi database directory.
- Load trial-level (target_rgb, action, color_distance) records from round JSON files.
- Intersect common policy sets across compared sources.
- Build deterministic full-data rankings (lower score = better, configurable).
- Expose metadata: n_policies, max_n, db_path.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


@dataclass
class StudyScores:
    """All score data for one study/database."""

    name: str
    db_path: str
    policy_scores: dict[str, list[float]]   # policy -> per-experiment scores
    full_rank: list[str]                     # policies sorted best→worst (lower = better)
    n_policies: int
    max_n: int                               # min experiments available across all policies


def load_study_scores(
    db_path: str,
    score_key: str = "best_color_distance_mean",
    lower_is_better: bool = True,
) -> StudyScores:
    """
    Load per-policy per-experiment scores from a PEGKi database directory.

    Parameters
    ----------
    db_path:
        Path to the database root, relative to the repo root (three levels up
        from this file: color_mixing_lab/).
    score_key:
        JSON field inside ``policy_stats`` to use as the performance score.
    lower_is_better:
        When True, policies are ranked ascending (lower score = better).

    Returns
    -------
    StudyScores with empty policy_scores if the database does not exist.
    """
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    policy_scores: dict[str, list[float]] = {}
    policies_dir = os.path.join(_repo_root, db_path, "policies")

    if os.path.exists(policies_dir):
        for pol in sorted(os.listdir(policies_dir)):
            exp_scores: list[float] = []
            for exp_dir in sorted(
                glob.glob(os.path.join(policies_dir, pol, "experiment_*"))
            ):
                summary_path = os.path.join(exp_dir, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path) as fh:
                        d = json.load(fh)
                    val = d.get("policy_stats", {}).get(score_key)
                    if val is not None:
                        exp_scores.append(float(val))
            if exp_scores:
                policy_scores[pol] = exp_scores

    if policy_scores:
        reverse = not lower_is_better
        full_rank = sorted(
            policy_scores.keys(),
            key=lambda p: float(np.mean(policy_scores[p])),
            reverse=reverse,
        )
        max_n = min(len(v) for v in policy_scores.values())
    else:
        full_rank = []
        max_n = 0

    # Use the leaf directory name as a fallback study name
    name = os.path.basename(db_path.rstrip("/\\"))

    return StudyScores(
        name=name,
        db_path=db_path,
        policy_scores=policy_scores,
        full_rank=full_rank,
        n_policies=len(policy_scores),
        max_n=max_n,
    )


def load_many_studies(
    study_map: dict[str, str],
    score_key: str = "best_color_distance_mean",
    lower_is_better: bool = True,
) -> dict[str, StudyScores]:
    """
    Load multiple studies from a name→db_path mapping.

    Only non-empty studies (databases that exist and contain at least one
    policy) are returned.
    """
    result: dict[str, StudyScores] = {}
    for name, path in study_map.items():
        raw = load_study_scores(path, score_key=score_key, lower_is_better=lower_is_better)
        # Override the name with the caller-supplied key
        study = StudyScores(
            name=name,
            db_path=raw.db_path,
            policy_scores=raw.policy_scores,
            full_rank=raw.full_rank,
            n_policies=raw.n_policies,
            max_n=raw.max_n,
        )
        if study.n_policies > 0:
            result[name] = study
    return result


def common_policy_subset(
    studies: dict[str, StudyScores],
    selected: list[str] | None = None,
) -> list[str]:
    """
    Return sorted list of policies present in ALL studies.

    Parameters
    ----------
    selected:
        Optional allowlist — only policies in this list are kept even if
        they are common across all studies.
    """
    if not studies:
        return []
    sets = [set(s.policy_scores.keys()) for s in studies.values()]
    common = sorted(set.intersection(*sets))
    if selected is not None:
        selected_set = set(selected)
        common = [p for p in common if p in selected_set]
    return common


def restrict_to_common_policies(
    study: StudyScores,
    common: list[str],
    lower_is_better: bool = True,
) -> StudyScores:
    """Return a new StudyScores containing only policies in ``common``."""
    policy_scores = {
        p: study.policy_scores[p] for p in common if p in study.policy_scores
    }
    if policy_scores:
        reverse = not lower_is_better
        full_rank = sorted(
            policy_scores.keys(),
            key=lambda p: float(np.mean(policy_scores[p])),
            reverse=reverse,
        )
        max_n = min(len(v) for v in policy_scores.values())
    else:
        full_rank = []
        max_n = 0

    return StudyScores(
        name=study.name,
        db_path=study.db_path,
        policy_scores=policy_scores,
        full_rank=full_rank,
        n_policies=len(policy_scores),
        max_n=max_n,
    )


# ---------------------------------------------------------------------------
# Paired HF/LF target-level data (for MFMC experiment 08)
# ---------------------------------------------------------------------------

@dataclass
class PairedTargetData:
    """
    Per-policy paired observations aligned by target_id (experiment index).

    Valid only when HF and LF databases were generated from the same shared-
    targets file, so experiment_NNN corresponds to the same target RGB in both.
    Alignment is by experiment_id (integer), NOT by target_rgb value.
    """

    hf_name: str
    lf_name: str
    policies: list[str]
    # policy -> array shape (n_targets, 3): columns = [target_id, hf_score, lf_score]
    records: dict[str, np.ndarray]
    n_targets: int          # min across all policies


def load_paired_data(
    hf_db: str,
    lf_db: str,
    hf_name: str = "hf",
    lf_name: str = "lf",
) -> PairedTargetData:
    """
    Load per-policy (target_id, hf_score, lf_score) aligned by experiment index.

    Only the intersection of experiment_ids available in BOTH databases is kept.
    If a policy is missing from either database it is excluded.

    Parameters
    ----------
    hf_db, lf_db:
        Paths relative to the repo root (three levels up from this file).
    hf_name, lf_name:
        Human-readable labels stored in the returned struct.

    Returns
    -------
    PairedTargetData with empty records if either database is missing.
    """
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    def _read_by_exp_id(db: str) -> dict[str, dict[int, float]]:
        """policy -> {experiment_id: best_color_distance_mean}"""
        out: dict[str, dict[int, float]] = {}
        pol_dir = os.path.join(_repo_root, db, "policies")
        if not os.path.exists(pol_dir):
            return out
        for pol in sorted(os.listdir(pol_dir)):
            scores: dict[int, float] = {}
            for exp_dir in sorted(
                glob.glob(os.path.join(pol_dir, pol, "experiment_*"))
            ):
                s_path = os.path.join(exp_dir, "summary.json")
                if not os.path.exists(s_path):
                    continue
                with open(s_path) as fh:
                    d = json.load(fh)
                exp_id = d.get("experiment_id")
                val = d.get("policy_stats", {}).get("best_color_distance_mean")
                if exp_id is not None and val is not None:
                    scores[int(exp_id)] = float(val)
            if scores:
                out[pol] = scores
        return out

    hf_by_id = _read_by_exp_id(hf_db)
    lf_by_id = _read_by_exp_id(lf_db)

    common_pols = sorted(set(hf_by_id) & set(lf_by_id))
    records: dict[str, np.ndarray] = {}

    for pol in common_pols:
        common_ids = sorted(set(hf_by_id[pol]) & set(lf_by_id[pol]))
        if not common_ids:
            continue
        arr = np.array(
            [[tid, hf_by_id[pol][tid], lf_by_id[pol][tid]] for tid in common_ids],
            dtype=float,
        )
        records[pol] = arr  # columns: [target_id, hf, lf]

    n_targets = min((len(v) for v in records.values()), default=0)

    return PairedTargetData(
        hf_name=hf_name,
        lf_name=lf_name,
        policies=list(records.keys()),
        records=records,
        n_targets=n_targets,
    )


# ---------------------------------------------------------------------------
# Trial-level data loading (for aggregation experiment 10)
# ---------------------------------------------------------------------------

class TrialRecord(NamedTuple):
    """One observed (target → action → outcome) triple from a round JSON file."""
    target_rgb: tuple[float, float, float]   # raw 0-255
    action: tuple[float, float, float]        # (red%, yellow%, blue%)
    color_distance: float
    policy_type: str
    source: str                               # study/engine name


def load_trial_records(
    db_path: str,
    source_name: str,
    policy_types: list[str] | None = None,
    max_experiments: int | None = None,
) -> list[TrialRecord]:
    """
    Load every trial record from all round_*.json files in a database.

    Each trial contributes one TrialRecord with:
      - target_rgb (raw 0-255 float tuple)
      - action (red%, yellow%, blue%)
      - color_distance (lower = better)
      - policy_type, source

    Parameters
    ----------
    db_path:
        Path relative to the repo root (three levels up from this file).
    source_name:
        Human-readable label for this database (stored in TrialRecord.source).
    policy_types:
        If given, only load records for these policy types.
    max_experiments:
        If given, load at most this many experiments per policy type.
    """
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    policies_dir = os.path.join(_repo_root, db_path, "policies")
    if not os.path.exists(policies_dir):
        return []

    records: list[TrialRecord] = []
    pol_dirs = sorted(os.listdir(policies_dir))

    for pol_name in pol_dirs:
        if policy_types is not None and pol_name not in policy_types:
            continue

        pol_dir = os.path.join(policies_dir, pol_name)
        exp_dirs = sorted(glob.glob(os.path.join(pol_dir, "experiment_*")))
        if max_experiments is not None:
            exp_dirs = exp_dirs[:max_experiments]

        for exp_dir in exp_dirs:
            for round_path in sorted(glob.glob(os.path.join(exp_dir, "round_*.json"))):
                try:
                    with open(round_path) as fh:
                        rdata = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue

                for trial in rdata.get("trials", []):
                    trgb = trial.get("target_rgb")
                    act  = trial.get("action", {})
                    cd   = trial.get("color_distance")
                    if trgb is None or cd is None:
                        continue
                    records.append(TrialRecord(
                        target_rgb=(float(trgb[0]), float(trgb[1]), float(trgb[2])),
                        action=(
                            float(act.get("red_percent",    0.0)),
                            float(act.get("yellow_percent", 0.0)),
                            float(act.get("blue_percent",   0.0)),
                        ),
                        color_distance=float(cd),
                        policy_type=pol_name,
                        source=source_name,
                    ))

    return records


def best_action_per_target(
    records: list[TrialRecord],
) -> dict[tuple[float, float, float], tuple[tuple[float, float, float], float]]:
    """
    For each unique target_rgb, return the action that achieved the lowest
    color_distance and that distance.

    Returns
    -------
    dict: target_rgb → (best_action, best_color_distance)
    """
    best: dict[tuple, tuple] = {}
    for r in records:
        key = r.target_rgb
        if key not in best or r.color_distance < best[key][1]:
            best[key] = (r.action, r.color_distance)
    return best
