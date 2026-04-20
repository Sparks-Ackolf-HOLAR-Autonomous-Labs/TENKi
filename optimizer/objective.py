"""
optimizer/objective.py -- Objective evaluation layer.

Translates a Suggestion into a CompletedEval.

Classes
-------
BaseObjective          Abstract interface
OfflineReplayObjective Replays PEGKi study databases; no live execution
"""

from __future__ import annotations

import os
import sys
import uuid
from abc import ABC, abstractmethod

import numpy as np

from .types import Suggestion, CompletedEval, IntermediateState


class BaseObjective(ABC):
    @abstractmethod
    def evaluate(
        self,
        suggestion: Suggestion,
        current_time: float = 0.0,
    ) -> CompletedEval: ...

    def resume(
        self,
        intermediate: IntermediateState,
        suggestion: Suggestion,
        current_time: float = 0.0,
    ) -> CompletedEval:
        return self.evaluate(suggestion, current_time)

    def supports_resume(self, source: str) -> bool:
        return False


class OfflineReplayObjective(BaseObjective):
    """
    Evaluates suggestions by replaying existing PEGKi study databases.

    Each job:
    - samples `fidelity` experiments from source's policy_scores pool
    - returns mean best_color_distance as the score
    - optionally stores sampled indices for deterministic resume

    Resume semantics:
    - A lower-fidelity result stores its sampled scores as IntermediateState
    - A higher-fidelity promotion extends those samples rather than resampling

    Fidelity-DB mode (when fidelity_db_map is non-empty):
    - Each fidelity level maps to a separate pre-generated PEGKi database path
    - Pool is all data for that fidelity level; score = mean of all experiments
    - No sampling; resume not applicable

    This keeps offline simulation deterministic and resumable without real
    process management, which is all that is needed for Milestone 1.
    """

    def __init__(
        self,
        study_map: dict[str, str],
        hifi_source: str,
        score_key: str = "best_color_distance_mean",
        lower_is_better: bool = True,
        allow_resume: bool = True,
        rng: np.random.Generator | None = None,
        fidelity_db_map: dict | None = None,
    ):
        self.study_map = study_map
        self.hifi_source = hifi_source
        self.score_key = score_key
        self.lower_is_better = lower_is_better
        self.allow_resume = allow_resume
        self.rng = rng or np.random.default_rng(42)
        self._studies: dict | None = None
        self.fidelity_db_map: dict = fidelity_db_map or {}
        self._fidelity_studies: dict = {}

    def _load_studies(self) -> None:
        if self._studies is not None:
            return
        _here = os.path.dirname(__file__)
        sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))
        from analysis.flip_data import load_many_studies
        self._studies = load_many_studies(self.study_map, score_key=self.score_key)

    def _load_fidelity_study(self, source: str, fid_str: str):
        key = (source, fid_str)
        if key not in self._fidelity_studies:
            db_path = self.fidelity_db_map[source][fid_str]
            _here = os.path.dirname(__file__)
            sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))
            from analysis.flip_data import load_study_scores
            self._fidelity_studies[key] = load_study_scores(db_path, self.score_key)
        return self._fidelity_studies.get((source, fid_str))

    def _pool(self, source: str, policy_name: str, fidelity=None) -> list[float]:
        if fidelity is not None and self.fidelity_db_map.get(source):
            fid_str = str(fidelity)
            if fid_str in self.fidelity_db_map[source]:
                study = self._load_fidelity_study(source, fid_str)
                if study:
                    return list(study.policy_scores.get(policy_name, []))
        # fall back to flat study_map
        self._load_studies()
        study = (self._studies or {}).get(source)
        return list(study.policy_scores.get(policy_name, [])) if study else []

    def all_policies(self) -> list[str]:
        # When fidelity_db_map is active, use highest fidelity level for the hifi source
        if self.fidelity_db_map.get(self.hifi_source):
            fid_strs = sorted(self.fidelity_db_map[self.hifi_source].keys(), key=lambda x: float(x))
            if fid_strs:
                study = self._load_fidelity_study(self.hifi_source, fid_strs[-1])
                if study:
                    return list(study.policy_scores.keys())
        self._load_studies()
        hf = (self._studies or {}).get(self.hifi_source)
        return list(hf.full_rank) if hf else []

    def hf_rank(self) -> list[str]:
        # When fidelity_db_map is active, use highest fidelity level for the hifi source
        if self.fidelity_db_map.get(self.hifi_source):
            fid_strs = sorted(self.fidelity_db_map[self.hifi_source].keys(), key=lambda x: float(x))
            if fid_strs:
                study = self._load_fidelity_study(self.hifi_source, fid_strs[-1])
                if study:
                    return list(study.policy_scores.keys())
        self._load_studies()
        hf = (self._studies or {}).get(self.hifi_source)
        return list(hf.full_rank) if hf else []

    def evaluate(
        self,
        suggestion: Suggestion,
        current_time: float = 0.0,
    ) -> CompletedEval:
        pool = self._pool(suggestion.source, suggestion.policy_name, suggestion.fidelity)
        if not pool:
            return CompletedEval(
                job_id=suggestion.job_id,
                source=suggestion.source,
                policy_name=suggestion.policy_name,
                target_id=suggestion.target_id,
                fidelity=suggestion.fidelity,
                score=float("nan"),
                runtime_observed=suggestion.expected_runtime,
                runtime_simulated_end=current_time + suggestion.expected_runtime,
                metadata={"error": "source_not_found"},
            )

        # Fidelity-DB mode: pool is all data for this fidelity level, use all of it
        fid_str = str(suggestion.fidelity)
        using_fidelity_db = (
            bool(self.fidelity_db_map.get(suggestion.source))
            and fid_str in self.fidelity_db_map.get(suggestion.source, {})
        )

        if using_fidelity_db:
            score = float(np.mean(pool))
            n_used = len(pool)
            can_resume = False
            resume_token_out = None
            sampled = pool
        else:
            # Legacy sampling: sample `fidelity` items from pool
            n = int(suggestion.fidelity)
            n_draw = n if n <= len(pool) else len(pool)
            replace = n > len(pool)
            idx = self.rng.choice(len(pool), size=n_draw, replace=replace)
            sampled = [pool[i] for i in idx]
            score = float(np.mean(sampled))
            n_used = n_draw
            can_resume = self.allow_resume and n < len(pool)
            resume_token_out = str(uuid.uuid4()) if can_resume else None

        return CompletedEval(
            job_id=suggestion.job_id,
            source=suggestion.source,
            policy_name=suggestion.policy_name,
            target_id=suggestion.target_id,
            fidelity=suggestion.fidelity,
            score=score,
            runtime_observed=suggestion.expected_runtime,
            runtime_simulated_end=current_time + suggestion.expected_runtime,
            resume_token_out=resume_token_out,
            can_resume=can_resume,
            metadata={"sampled_scores": sampled[:20], "n_available": len(pool), "n_used": n_used},
        )

    def resume(
        self,
        intermediate: IntermediateState,
        suggestion: Suggestion,
        current_time: float = 0.0,
    ) -> CompletedEval:
        pool = self._pool(suggestion.source, suggestion.policy_name, suggestion.fidelity)
        prior = intermediate.sampled_scores
        n_new = int(suggestion.fidelity) - len(prior)
        if n_new <= 0 or not pool:
            score = float(np.mean(prior)) if prior else float("nan")
            sampled = list(prior)
        else:
            n_draw = min(n_new, len(pool))
            idx = self.rng.choice(len(pool), size=n_draw, replace=n_new > len(pool))
            sampled = prior + [pool[i] for i in idx]
            score = float(np.mean(sampled))
        n_total = len(sampled)
        can_resume = self.allow_resume and n_total < len(pool)
        resume_token_out = str(uuid.uuid4()) if can_resume else None
        return CompletedEval(
            job_id=suggestion.job_id,
            source=suggestion.source,
            policy_name=suggestion.policy_name,
            target_id=suggestion.target_id,
            fidelity=n_total,
            score=score,
            runtime_observed=suggestion.expected_runtime,
            runtime_simulated_end=current_time + suggestion.expected_runtime,
            resume_token_out=resume_token_out,
            can_resume=can_resume,
            metadata={
                "sampled_scores": sampled,
                "n_available": len(pool),
                "resumed_from": intermediate.resume_token,
            },
        )

    def supports_resume(self, source: str) -> bool:
        return self.allow_resume
