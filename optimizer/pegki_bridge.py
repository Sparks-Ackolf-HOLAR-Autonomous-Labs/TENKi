"""
optimizer/pegki_bridge.py -- Belief updates from completed evaluations.

Translates CompletedEval records into PEGKi-style belief estimates:
- tau / rho agreement with HF reference
- bias floor (ceiling tau achievable with infinite data)
- flip probability (ranking instability)
- donor score (low-fidelity transfer strength)
- effective fidelity / usefulness score for policy acquisition
- team rating inputs (reserved for TrueSkill / BladeChest adapters)

Design rule: the optimizer depends only on bridge outputs (BeliefState),
not on the internals of any specific rating model.
"""

from __future__ import annotations

import numpy as np

from .types import CompletedEval, BeliefState


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    if len(rank_a) < 2 or len(rank_b) < 2:
        return 0.0
    pos_a = {p: i for i, p in enumerate(rank_a)}
    pos_b = {p: i for i, p in enumerate(rank_b)}
    common = [p for p in rank_a if p in pos_b]
    if len(common) < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            pi, pj = common[i], common[j]
            da = pos_a[pi] - pos_a[pj]
            db = pos_b[pi] - pos_b[pj]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


def _score_map(
    completed: list[CompletedEval],
    source: str,
    fidelity: int | float | None = None,
) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for ev in completed:
        if ev.source != source or (isinstance(ev.score, float) and np.isnan(ev.score)):
            continue
        if fidelity is not None and ev.fidelity != fidelity:
            continue
        scores.setdefault(ev.policy_name, []).append(ev.score)
    return {p: float(np.mean(v)) for p, v in scores.items()}


def _pearson_rho(scores_a: dict[str, float], scores_b: dict[str, float]) -> float | None:
    common = [p for p in scores_a if p in scores_b]
    if len(common) < 2:
        return None
    a = np.array([scores_a[p] for p in common], dtype=float)
    b = np.array([scores_b[p] for p in common], dtype=float)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    rho = np.corrcoef(a, b)[0, 1]
    if np.isnan(rho):
        return None
    return float(rho)


def _tau_to_unit_interval(tau: float | None) -> float | None:
    if tau is None:
        return None
    return float(np.clip((tau + 1.0) / 2.0, 0.0, 1.0))


def _blend_effective_fidelity(
    tau_mean: float | None,
    rho_mean: float | None,
    bias_floor: float | None,
    flip_probability: float | None,
    donor_score: float | None,
) -> float | None:
    """
    PEGKi-defined effective fidelity.

    Raw execution fidelity remains the budget axis. This score estimates how
    useful that raw fidelity is as a transfer state relative to the HF source.
    """
    components: list[tuple[float, float]] = []
    tau_unit = _tau_to_unit_interval(tau_mean)
    donor_unit = _tau_to_unit_interval(donor_score)
    if tau_unit is not None:
        components.append((0.35, tau_unit))
    if rho_mean is not None:
        components.append((0.25, float(np.clip(rho_mean, 0.0, 1.0))))
    if bias_floor is not None:
        components.append((0.20, float(np.clip(1.0 - bias_floor, 0.0, 1.0))))
    if donor_unit is not None:
        components.append((0.15, donor_unit))
    if flip_probability is not None:
        components.append((0.05, float(np.clip(1.0 - flip_probability, 0.0, 1.0))))
    if not components:
        return None
    weight_sum = sum(weight for weight, _ in components)
    return float(sum(weight * value for weight, value in components) / weight_sum)


def _refresh_effective_fidelity_scores(
    beliefs: list[BeliefState],
    sources: list[str],
) -> list[BeliefState]:
    belief_map: dict[tuple[str, int | float | None], BeliefState] = {
        (b.source_name, b.fidelity): b for b in beliefs
    }
    for src in sources:
        per_fid = [
            b for b in belief_map.values()
            if b.source_name == src and b.fidelity is not None
        ]
        global_key = (src, None)
        global_belief = belief_map.get(global_key)
        if global_belief is None:
            global_belief = BeliefState(source_name=src, fidelity=None)
            belief_map[global_key] = global_belief

        if per_fid:
            tau_vals = [b.tau_mean for b in per_fid if b.tau_mean is not None]
            rho_vals = [b.rho_mean for b in per_fid if b.rho_mean is not None]
            donor_belief = min(per_fid, key=lambda b: float(b.fidelity))
            if tau_vals:
                global_belief.tau_mean = float(np.mean(tau_vals))
                global_belief.tau_std = float(np.std(tau_vals))
            if rho_vals:
                global_belief.rho_mean = float(np.mean(rho_vals))
            if donor_belief.tau_mean is not None:
                global_belief.donor_score = donor_belief.tau_mean

        global_belief.effective_fidelity = _blend_effective_fidelity(
            global_belief.tau_mean,
            global_belief.rho_mean,
            global_belief.bias_floor,
            global_belief.flip_probability,
            global_belief.donor_score,
        )
        if global_belief.effective_fidelity is not None:
            global_belief.quality_score = global_belief.effective_fidelity
        global_belief.metadata["pegki_components"] = {
            "tau_mean": global_belief.tau_mean,
            "rho_mean": global_belief.rho_mean,
            "bias_floor": global_belief.bias_floor,
            "flip_probability": global_belief.flip_probability,
            "donor_score": global_belief.donor_score,
        }

        for belief in per_fid:
            belief.donor_score = global_belief.donor_score
            belief.effective_fidelity = _blend_effective_fidelity(
                belief.tau_mean,
                belief.rho_mean,
                global_belief.bias_floor,
                global_belief.flip_probability,
                global_belief.donor_score,
            )
            if belief.effective_fidelity is not None:
                belief.quality_score = belief.effective_fidelity
            belief.metadata["pegki_components"] = {
                "tau_mean": belief.tau_mean,
                "rho_mean": belief.rho_mean,
                "bias_floor": global_belief.bias_floor,
                "flip_probability": global_belief.flip_probability,
                "donor_score": global_belief.donor_score,
            }

    return list(belief_map.values())


def compute_implied_rank(
    completed: list[CompletedEval],
    source: str,
    lower_is_better: bool = True,
) -> list[str]:
    """
    Aggregate all completed evaluations for a source and compute implied policy rank.
    Uses mean score across all fidelities (higher fidelity overrides lower via mean).
    """
    scores = _score_map(completed, source)
    if not scores:
        return []
    return sorted(scores, key=lambda p: scores[p], reverse=not lower_is_better)


def compute_implied_rank_at_fidelity(
    completed: list[CompletedEval],
    source: str,
    fidelity: int | float,
    lower_is_better: bool = True,
) -> list[str]:
    """Implied rank using only evaluations at a specific fidelity."""
    scores = _score_map(completed, source, fidelity)
    if not scores:
        return []
    return sorted(scores, key=lambda p: scores[p], reverse=not lower_is_better)


# ---------------------------------------------------------------------------
# Belief update functions
# ---------------------------------------------------------------------------

def update_tau_rho_beliefs(
    beliefs: list[BeliefState],
    completed: list[CompletedEval],
    hf_rank: list[str],
    hifi_source: str,
    sources: list[str],
    fidelity_levels: list[int | float],
    lower_is_better: bool = True,
    current_time: float = 0.0,
    alpha: float = 0.3,
) -> list[BeliefState]:
    """
    For each (source, fidelity) compute:
    - implied ranking -> tau vs HF rank
    - score-space agreement -> rho vs HF mean scores

    Updates BeliefState with exponential moving averages, then refreshes the
    PEGKi-defined effective-fidelity score used by the acquisition policies.
    """
    belief_map: dict[tuple[str, int | float | None], BeliefState] = {
        (b.source_name, b.fidelity): b for b in beliefs
    }
    hf_evals = [ev for ev in completed if ev.source == hifi_source]
    hf_scores: dict[str, float] = {}
    hf_fidelity: int | float | None = None
    if hf_evals:
        hf_fidelity = max(ev.fidelity for ev in hf_evals)
        hf_scores = _score_map(hf_evals, hifi_source, hf_fidelity)

    for src in sources:
        for fid in fidelity_levels:
            score_map = _score_map(completed, src, fid)
            if not score_map:
                continue
            implied = compute_implied_rank_at_fidelity(completed, src, fid, lower_is_better)
            if len(implied) < 2:
                continue
            tau = _kendall_tau(implied, hf_rank)
            rho = None
            if src == hifi_source and hf_fidelity == fid and hf_scores:
                rho = 1.0
            elif hf_scores:
                rho = _pearson_rho(score_map, hf_scores)
            key = (src, fid)
            b = belief_map.get(key)
            if b is None:
                b = BeliefState(source_name=src, fidelity=fid)
                belief_map[key] = b
            if b.tau_mean is None:
                b.tau_mean = tau
                b.tau_std = 0.0
            else:
                b.tau_mean = (1 - alpha) * b.tau_mean + alpha * tau
                b.tau_std = (1 - alpha) * (b.tau_std or 0.0) + alpha * abs(tau - b.tau_mean)
            if rho is not None:
                if b.rho_mean is None:
                    b.rho_mean = rho
                else:
                    b.rho_mean = (1 - alpha) * b.rho_mean + alpha * rho
            b.n_observations += 1
            b.last_updated_time = current_time
    return _refresh_effective_fidelity_scores(list(belief_map.values()), sources)


def update_bias_floor_estimates(
    beliefs: list[BeliefState],
    completed: list[CompletedEval],
    hf_rank: list[str],
    sources: list[str],
    lower_is_better: bool = True,
    current_time: float = 0.0,
) -> list[BeliefState]:
    """
    Estimate bias floor: tau at maximum observed fidelity.
    Approximates the ceiling achievable with infinite data from that source.
    bias_floor = 1 - tau_at_max_fidelity
    """
    # Keep existing fidelity-keyed beliefs
    fid_beliefs = {(b.source_name, b.fidelity): b for b in beliefs if b.fidelity is not None}
    # Global (fidelity=None) beliefs for bias floor
    global_beliefs: dict[str, BeliefState] = {
        b.source_name: b for b in beliefs if b.fidelity is None
    }
    for src in sources:
        src_evals = [ev for ev in completed if ev.source == src]
        if not src_evals:
            continue
        max_fid = max(ev.fidelity for ev in src_evals)
        implied = compute_implied_rank_at_fidelity(src_evals, src, max_fid, lower_is_better)
        if len(implied) < 2:
            continue
        tau = _kendall_tau(implied, hf_rank)
        b = global_beliefs.get(src)
        if b is None:
            b = BeliefState(source_name=src, fidelity=None)
            global_beliefs[src] = b
        b.bias_floor = 1.0 - tau
        b.last_updated_time = current_time
    merged = list(fid_beliefs.values()) + list(global_beliefs.values())
    return _refresh_effective_fidelity_scores(merged, sources)


def update_flip_probability(
    beliefs: list[BeliefState],
    completed: list[CompletedEval],
    hf_rank: list[str],
    sources: list[str],
    lower_is_better: bool = True,
    flip_threshold: float = 0.05,
    current_time: float = 0.0,
) -> list[BeliefState]:
    """
    Estimate flip probability by bootstrap-resampling one score per policy at the
    maximum observed fidelity. High flip_probability -> unstable source.
    """
    fid_beliefs = {(b.source_name, b.fidelity): b for b in beliefs if b.fidelity is not None}
    global_beliefs: dict[str, BeliefState] = {
        b.source_name: b for b in beliefs if b.fidelity is None
    }
    rng = np.random.default_rng(0)
    for src in sources:
        src_evals = [ev for ev in completed if ev.source == src]
        if len(src_evals) < 3:
            continue
        max_fid = max(ev.fidelity for ev in src_evals)
        by_policy: dict[str, list[float]] = {}
        for ev in src_evals:
            if ev.fidelity == max_fid and not (isinstance(ev.score, float) and np.isnan(ev.score)):
                by_policy.setdefault(ev.policy_name, []).append(ev.score)
        if len(by_policy) < 2:
            continue
        taus = []
        for _ in range(32):
            sampled_scores = {
                policy: float(rng.choice(scores))
                for policy, scores in by_policy.items()
                if scores
            }
            if len(sampled_scores) < 2:
                continue
            implied = sorted(
                sampled_scores,
                key=lambda p: sampled_scores[p],
                reverse=not lower_is_better,
            )
            taus.append(_kendall_tau(implied, hf_rank))
        if not taus:
            continue
        mean_tau = float(np.mean(taus))
        flip_prob = float(np.mean([abs(t - mean_tau) > flip_threshold for t in taus]))
        b = global_beliefs.get(src)
        if b is None:
            b = BeliefState(source_name=src, fidelity=None)
            global_beliefs[src] = b
        b.flip_probability = flip_prob
        b.last_updated_time = current_time
    merged = list(fid_beliefs.values()) + list(global_beliefs.values())
    return _refresh_effective_fidelity_scores(merged, sources)


def build_team_rating_inputs(
    completed: list[CompletedEval],
    hf_rank: list[str],
    sources: list[str],
    lower_is_better: bool = True,
) -> dict:
    """
    Build normalized inputs for team-level rating models.
    Reserved for future TrueSkill / BladeChest adapter integration.
    """
    team_data: dict = {"sources": {}, "pairwise": {}}
    for src in sources:
        src_evals = [ev for ev in completed if ev.source == src]
        if not src_evals:
            continue
        implied = compute_implied_rank(src_evals, src, lower_is_better)
        tau = _kendall_tau(implied, hf_rank) if len(implied) >= 2 else 0.0
        team_data["sources"][src] = {
            "implied_rank": implied,
            "tau_vs_hf": tau,
            "n_evals": len(src_evals),
        }
    return team_data
