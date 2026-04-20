"""
analysis/swarm_agents.py -- Stateful swarm agents for source aggregation.

Each LF source becomes a SwarmAgent with:
- A kNN memory bank populated per-bootstrap from the SAME sampled targets
  seen by ensemble and local_router at that budget (no oracle leakage)
- A local competence model via inverse-distance-weighted kNN prediction
- An abstention mechanism driven by neighbourhood confidence
- Optional structural trust priors from TenKi diagnostics
  (donor_score, bias_floor, ceiling)

Memory model (Version A): source-level only.
  memory[target] = mean_best_color_distance across all policies.
  This is policy-agnostic: the same per-target source weight is applied
  to every policy's score at that target.  Limitation: cannot detect
  source × policy interaction (e.g. source A is better for bayesian_ei
  but worse for grid_search).  Version B would store per-policy errors;
  implement if Version A swarm still underperforms local_router.

Budget fairness contract:
  Agents MUST call load_sampled_memory(cube_slice, policies, sampled_targets)
  at the start of every bootstrap trial.  This resets and refills memory
  from ONLY the sampled targets for that trial.  The agent then knows
  exactly what ensemble and local_router know -- no more, no less.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# TenKi structural trust priors
# ---------------------------------------------------------------------------

@dataclass
class TenkiPriors:
    """
    Structural trust from TenKi diagnostics for one source.

    Canonical values (TENKi-1000, flip_test_summary.json):
        source       tau@N=1  ceiling  bias_floor  donor_score
        spectral     1.000    1.000    0.000        +0.233
        study_b      0.808    0.833    0.167        +0.227
        mixbox       0.765    0.889    0.111        +0.142
        ryb          0.677    0.944    0.056        +0.042
        study_c_reverse  0.542    0.833    0.167        -0.054
        study_b_reverse  0.536    0.833    0.167        -0.069
        km           0.464    0.778    0.222        -0.107
        study_a      0.438    0.833    0.167        -0.176
        study_c      0.342    0.722    0.278        -0.238
    """
    donor_score: float = 0.0
    bias_floor:  float = 0.0
    ceiling:     float = 1.0


BUILTIN_TENKI_PRIORS: dict[str, TenkiPriors] = {
    "spectral":    TenkiPriors(donor_score=+0.233, bias_floor=0.000, ceiling=1.000),
    "mixbox":      TenkiPriors(donor_score=+0.142, bias_floor=0.111, ceiling=0.889),
    "ryb":         TenkiPriors(donor_score=+0.042, bias_floor=0.056, ceiling=0.944),
    "km":          TenkiPriors(donor_score=-0.107, bias_floor=0.222, ceiling=0.778),
    "study_a":     TenkiPriors(donor_score=-0.176, bias_floor=0.167, ceiling=0.833),
    "study_b":     TenkiPriors(donor_score=+0.227, bias_floor=0.167, ceiling=0.833),
    # Canonical names match the actual output DB directories (study_b_reverse_*, study_c_reverse_*)
    "study_b_reverse": TenkiPriors(donor_score=-0.069, bias_floor=0.167, ceiling=0.833),
    "study_c":         TenkiPriors(donor_score=-0.238, bias_floor=0.278, ceiling=0.722),
    "study_c_reverse": TenkiPriors(donor_score=-0.054, bias_floor=0.167, ceiling=0.833),
    # Short aliases kept for backward compatibility with any existing callers
    "study_b_rev": TenkiPriors(donor_score=-0.069, bias_floor=0.167, ceiling=0.833),
    "study_c_rev": TenkiPriors(donor_score=-0.054, bias_floor=0.167, ceiling=0.833),
}


def load_tenki_priors_from_json(path: str) -> dict[str, TenkiPriors]:
    """Load TenKi priors from a JSON file (donor_score/bias_floor/ceiling per source)."""
    import json
    with open(path) as fh:
        raw = json.load(fh)
    return {
        src: TenkiPriors(
            donor_score=float(vals.get("donor_score", 0.0)),
            bias_floor=float(vals.get("bias_floor", 0.0)),
            ceiling=float(vals.get("ceiling", 1.0)),
        )
        for src, vals in raw.items()
    }


# ---------------------------------------------------------------------------
# SwarmAgent
# ---------------------------------------------------------------------------

class SwarmAgent:
    """
    A stateful agent representing one LF source in the swarm.

    Lifecycle per bootstrap trial
    -----------------------------
    1. ``load_sampled_memory(cube_slice, policies, sampled_targets)``
          Reset and refill memory from the same sampled targets used by
          ensemble/local_router.  MUST be called before predict().
    2. ``predict(query_rgb, k)``
          Local competence estimate + confidence from kNN recall.
    3. ``should_abstain(confidence)``
          Gate before contributing to consensus.
    4. ``global_weight()``
          Reputation-based weight (TenKi-gated) for abstention fallback.
    """

    def __init__(
        self,
        name: str,
        priors: Optional[TenkiPriors] = None,
        abstain_threshold: float = 0.25,
        min_neighbors: int = 2,
        proximity_scale: float = 50.0,
    ) -> None:
        self.name = name
        self.priors = priors or BUILTIN_TENKI_PRIORS.get(name, TenkiPriors())
        self.abstain_threshold = abstain_threshold
        self.min_neighbors = min_neighbors
        self.proximity_scale = proximity_scale

        # Memory: populated fresh per bootstrap trial, never at construction time
        self._mem_targets: np.ndarray = np.empty((0, 3), dtype=float)
        self._mem_errors:  np.ndarray = np.empty(0, dtype=float)
        self.reputation_error: float = float("inf")
        # Unique target count from last load_sampled_memory call (for coverage)
        self._sampled_unique_n: int = 0

    # ------------------------------------------------------------------
    # Memory — per-bootstrap, budget-consistent
    # ------------------------------------------------------------------

    def load_sampled_memory(
        self,
        source_cube: dict[str, dict[tuple, float]],
        policies: list[str],
        sampled_targets: list[tuple],
    ) -> None:
        """
        Reset and populate memory from ONLY the sampled targets for this trial.

        Version A: policy-agnostic. For each target, memory stores the mean
        best_color_distance across all policies. This gives the same information
        budget as ensemble and local_router at the same N.

        Deduplication: bootstrap sampling with replacement can repeat the same
        target multiple times in sampled_targets. Storing duplicates would inflate
        memory_size and the min_neighbors guard, making the agent appear more
        certain than the unique evidence supports. Only unique targets are stored;
        the raw draw size is kept separately for honest coverage reporting.

        Parameters
        ----------
        source_cube : cube[policy][target_rgb] = best_color_distance (this source only)
        policies    : HF policy set (same policies ranked by all methods)
        sampled_targets : the sampled draw shared with all other methods this trial
        """
        # Deduplicate while preserving first-occurrence order
        unique_targets = list(dict.fromkeys(sampled_targets))
        self._sampled_unique_n = len(unique_targets)

        targets_list: list[np.ndarray] = []
        errors_list:  list[float] = []

        for t in unique_targets:
            errs = [
                source_cube[p][t]
                for p in policies
                if t in source_cube.get(p, {})
            ]
            if errs:
                targets_list.append(np.array(t, dtype=float))
                errors_list.append(float(np.mean(errs)))

        if targets_list:
            self._mem_targets = np.stack(targets_list)
            self._mem_errors  = np.array(errors_list)
            self.reputation_error = float(np.mean(self._mem_errors))
        else:
            self._mem_targets = np.empty((0, 3), dtype=float)
            self._mem_errors  = np.empty(0, dtype=float)
            self.reputation_error = float("inf")

    @property
    def memory_size(self) -> int:
        """Number of unique targets stored (deduplicated)."""
        return len(self._mem_targets)

    @property
    def memory_coverage(self) -> float:
        """
        Fraction of the unique sampled targets this agent has data for.
        Uses unique-target count from the last load_sampled_memory call,
        not the raw bootstrap draw size (which may contain duplicates).
        """
        return self.memory_size / max(self._sampled_unique_n, 1)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        query_rgb: tuple,
        k: int = 5,
        exclude_target: Optional[tuple] = None,
    ) -> tuple[float, float, int]:
        """
        Predict local competence at ``query_rgb`` using kNN memory recall.

        Leave-one-out contract: when ``exclude_target`` is given, all memory
        entries whose stored target equals ``exclude_target`` are excluded from
        the kNN search.  This prevents the method from using the target it is
        currently judging as evidence of its own local competence.  Bootstrap
        sampling may have placed multiple copies of the same target into memory;
        all copies are removed together.

        Parameters
        ----------
        query_rgb      : target color being judged
        k              : kNN neighbourhood size
        exclude_target : target to exclude (should equal query_rgb for LOO)

        Returns
        -------
        predicted_error   : float
        confidence        : float in [0, 1]
        n_eligible        : int  — neighbors available after LOO exclusion
                            (0 means fully starved; caller sees confidence=0)
        """
        q = np.array(query_rgb, dtype=float)

        # LOO exclusion: remove all entries matching exclude_target
        if exclude_target is not None and self.memory_size > 0:
            exc = np.array(exclude_target, dtype=float)
            # Integer-valued RGB targets — exact equality is safe
            keep = ~np.all(self._mem_targets == exc, axis=1)
            mem_t = self._mem_targets[keep]
            mem_e = self._mem_errors[keep]
        else:
            mem_t = self._mem_targets
            mem_e = self._mem_errors

        n_eligible = len(mem_t)
        if n_eligible < self.min_neighbors:
            return self.reputation_error, 0.0, n_eligible

        dists    = np.linalg.norm(mem_t - q, axis=1)
        k_actual = min(k, n_eligible)
        nn_idx   = np.argsort(dists)[:k_actual]
        nn_errs  = mem_e[nn_idx]
        nn_dists = dists[nn_idx]

        # Inverse-distance-weighted prediction
        idw       = 1.0 / (nn_dists + 1e-6)
        predicted = float(np.average(nn_errs, weights=idw))

        # Confidence: count × inverse-variance × proximity
        count_conf = min(1.0, k_actual / max(self.min_neighbors * 2, 1))
        var        = float(np.var(nn_errs)) if k_actual > 1 else (predicted ** 2 + 1.0)
        var_conf   = 1.0 / (1.0 + var / max(predicted ** 2, 1e-6))
        prox_conf  = float(np.exp(-nn_dists.mean() / self.proximity_scale))
        confidence = count_conf * var_conf * prox_conf

        # TenKi structural trust attenuation
        p = self.priors
        if p.donor_score < 0.0:
            confidence *= max(0.05, 1.0 + p.donor_score)
        if p.ceiling < 0.7:
            confidence *= p.ceiling / 0.7
        if p.bias_floor > 0.15:
            confidence *= 1.0 - min((p.bias_floor - 0.15) / 0.85, 0.9)

        return predicted, float(np.clip(confidence, 0.0, 1.0)), k_actual

    def should_abstain(self, confidence: float) -> bool:
        return confidence < self.abstain_threshold

    # ------------------------------------------------------------------
    # Global reputation weight (TenKi-gated) — used as abstention fallback
    # ------------------------------------------------------------------

    def global_weight(self) -> float:
        """
        Scalar trustworthiness based on mean observed error + TenKi penalties.
        This is budget-consistent: reputation_error is set from the sampled data.
        """
        w = 1.0 / max(self.reputation_error, 1e-6)
        p = self.priors
        if p.donor_score < 0.0:
            w *= max(0.05, 1.0 + p.donor_score)
        if p.bias_floor > 0.0:
            w *= max(0.01, 1.0 - p.bias_floor)
        if p.ceiling < 1.0:
            w *= p.ceiling
        return max(w, 1e-9)

    def __repr__(self) -> str:
        return (
            f"SwarmAgent({self.name!r}, "
            f"mem={self.memory_size}, "
            f"rep_err={self.reputation_error:.1f}, "
            f"donor={self.priors.donor_score:+.3f})"
        )


# ---------------------------------------------------------------------------
# Factory (lightweight — no memory loaded at construction)
# ---------------------------------------------------------------------------

def build_swarm(
    sources: list[str],
    tenki_priors: Optional[dict[str, TenkiPriors]] = None,
    abstain_threshold: float = 0.25,
    min_neighbors: int = 2,
) -> list[SwarmAgent]:
    """
    Construct one lightweight SwarmAgent per source.

    Memory is NOT loaded here.  Call agent.load_sampled_memory(...) at the
    start of every bootstrap trial to ensure budget-consistent memory.

    Parameters
    ----------
    sources           : ordered source name list (LF pool)
    tenki_priors      : override map; falls back to BUILTIN_TENKI_PRIORS
    abstain_threshold : confidence below which an agent stays silent
    min_neighbors     : minimum kNN entries before agent can predict
    """
    priors_map = tenki_priors or {}
    return [
        SwarmAgent(
            name=src,
            priors=priors_map.get(src, BUILTIN_TENKI_PRIORS.get(src)),
            abstain_threshold=abstain_threshold,
            min_neighbors=min_neighbors,
        )
        for src in sources
    ]
