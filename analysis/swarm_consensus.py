"""
analysis/swarm_consensus.py -- Consensus aggregation for swarm agents.

Three swarm modes share ``swarm_rank_sampled``; mode flags select features:

  swarm_memory          allow_abstain=False, message_rounds=0
  swarm_memory_abstain  allow_abstain=True,  message_rounds=0
  swarm_consensus       allow_abstain=True,  message_rounds=1

Budget fairness: this module calls agent.load_sampled_memory() at the top
of every swarm_rank_sampled() call.  Agents' memory is built from the same
sampled draw passed by the experiment driver -- the same draw used by
ensemble and local_router.  No oracle knowledge enters the swarm.

Abstention fallback priority (when all agents abstain at a target):
  1. TenKi-gated reputation weights (global_weight() -- budget-consistent
     because reputation_error is set from the same sampled data)
  2. Best-single-source (lowest reputation_error among agents)
  3. Equal weights -- last resort; logged as diagnostic

Diagnostics collected per target, accumulated across targets, returned
as per-trial averages:
  abstention_rate       fraction of (target × agent) pairs where agent abstained
  fallback_rate         fraction of targets where fallback was triggered
  fallback_type         "reputation" | "best_single" | "equal" (most common)
  downweight_rate       fraction of (target × agent) pairs downweighted by consensus
  mean_active_agents    mean agents speaking per target
  memory_sizes          list of memory_size per agent after load_sampled_memory
  mean_memory_size      mean across agents
  mean_memory_coverage  fraction of sampled targets each agent had data for (mean)
"""

from __future__ import annotations

import numpy as np

from analysis.swarm_agents import SwarmAgent


# ---------------------------------------------------------------------------
# Abstention fallback
# ---------------------------------------------------------------------------

def _fallback_weights(agents: list[SwarmAgent]) -> tuple[np.ndarray, str]:
    """
    Reputation-first fallback weights when all agents abstain.

    Returns (weights, fallback_type) where fallback_type is one of:
    "reputation", "best_single", "equal".
    """
    # 1. TenKi-gated reputation weights
    rep_w = np.array([a.global_weight() for a in agents])
    total = rep_w.sum()
    if total > 1e-9 and not np.all(rep_w == rep_w[0]):
        return rep_w / total, "reputation"

    # 2. Best single source (lowest mean error in the sampled memory)
    rep_errs = np.array([a.reputation_error for a in agents])
    if not np.all(np.isinf(rep_errs)):
        w = np.zeros(len(agents))
        w[int(np.argmin(rep_errs))] = 1.0
        return w, "best_single"

    # 3. Equal weights (all agents have no memory data at all)
    n = len(agents)
    return np.ones(n) / n, "equal"


# ---------------------------------------------------------------------------
# Core: compute consensus weights at a single target
# ---------------------------------------------------------------------------

def compute_consensus_weights(
    agents: list[SwarmAgent],
    query_rgb: tuple,
    k: int = 5,
    allow_abstain: bool = True,
    message_rounds: int = 1,
    disagreement_threshold: float = 1.5,
    exclude_target: tuple | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Compute consensus source weights for one query target.

    Leave-one-out: ``exclude_target`` (should equal ``query_rgb``) is passed
    to every agent.predict() so the judged target cannot be its own evidence.

    Phase 1 -- Local proposals:
        Each agent predicts (error, confidence, n_eligible) for query_rgb,
        excluding exclude_target from its kNN search.
        If allow_abstain and confidence < threshold: mark abstained.

    Phase 2 -- Message passing (message_rounds iterations):
        Active agents share predictions.  Any agent deviating more than
        ``disagreement_threshold`` sigma from the confidence-weighted
        consensus is down-weighted proportionally.

    Phase 3 -- Final weights:
        weight_i = confidence_i / (predicted_error_i + eps)
        Normalised to sum 1.
        If all abstained: fallback via _fallback_weights (reputation first).

    Returns
    -------
    weights    : np.ndarray shape (n_agents,), sums to 1
    diagnostics: dict -- per-target detail for aggregation in swarm_rank_sampled
    """
    n = len(agents)
    preds      = np.zeros(n)
    confs      = np.zeros(n)
    n_eligible = np.zeros(n, dtype=int)
    abstained  = np.zeros(n, dtype=bool)

    # Phase 1
    for i, agent in enumerate(agents):
        pred, conf, n_elig = agent.predict(query_rgb, k=k, exclude_target=exclude_target)
        preds[i]      = pred
        confs[i]      = conf
        n_eligible[i] = n_elig
        if allow_abstain and agent.should_abstain(conf):
            abstained[i] = True
            confs[i] = 0.0

    # Phase 2: message passing
    n_downweighted = 0
    for _round in range(message_rounds):
        active = ~abstained & (confs > 0.0)
        if active.sum() < 2:
            break
        w_sum = confs[active].sum()
        if w_sum < 1e-9:
            break
        consensus_pred = float(np.dot(preds[active], confs[active]) / w_sum)
        consensus_std  = float(
            np.sqrt(np.dot((preds[active] - consensus_pred) ** 2, confs[active]) / w_sum)
        )
        if consensus_std < 1e-6:
            break
        for i in range(n):
            if active[i]:
                deviation = abs(preds[i] - consensus_pred) / consensus_std
                if deviation > disagreement_threshold:
                    confs[i] *= max(0.05, disagreement_threshold / deviation)
                    n_downweighted += 1

    # Phase 3: final weights
    speaking = ~abstained & (confs > 0.0)
    fallback_used = False
    fallback_type = "none"

    if speaking.sum() == 0:
        # All abstained: reputation-first fallback
        weights, fallback_type = _fallback_weights(agents)
        fallback_used = True
    else:
        weights = np.zeros(n)
        for i in range(n):
            if speaking[i]:
                weights[i] = confs[i] / (preds[i] + 1e-6)
        w_sum = weights.sum()
        if w_sum > 1e-9:
            weights /= w_sum
        else:
            weights, fallback_type = _fallback_weights(agents)
            fallback_used = True

    diag = {
        "confidences":      confs.tolist(),
        "abstained":        abstained.tolist(),
        "predicted_errors": preds.tolist(),
        "n_eligible":       n_eligible.tolist(),
        "active_count":     int(speaking.sum()),
        "fallback_used":    fallback_used,
        "fallback_type":    fallback_type,
        "downweight_count": n_downweighted,
        # LOO starvation: agents with 0 eligible neighbors after exclusion
        "loo_starved_count": int((n_eligible == 0).sum()),
        "loo_mean_neighbors": float(n_eligible.mean()),
    }
    return weights, diag


# ---------------------------------------------------------------------------
# Ranking function: budget-consistent, shared-sample
# ---------------------------------------------------------------------------

def swarm_rank_sampled(
    cube: dict[str, dict[str, dict[tuple, float]]],
    agents: list[SwarmAgent],
    policies: list[str],
    sampled: list[tuple],
    k: int = 5,
    allow_abstain: bool = False,
    message_rounds: int = 0,
) -> tuple[list[str], np.ndarray, dict]:
    """
    Rank policies using swarm agents on a pre-sampled target list.

    Budget contract: agents' memory is built from ``sampled`` at the start
    of this call.  This matches the information available to ensemble and
    local_router on the exact same draw.

    Parameters
    ----------
    cube          : cube[source][policy][target_rgb] = best_color_distance
    agents        : pre-built SwarmAgent list (memory will be reset here)
    policies      : policy names to rank
    sampled       : pre-sampled target list (shared with all other methods)
    k             : kNN neighbourhood size
    allow_abstain : enable abstention phase (swarm_memory_abstain / swarm_consensus)
    message_rounds: consensus rounds (swarm_consensus only)

    Returns
    -------
    rank          : list[str], policies sorted best→worst (lower error = better)
    weight_matrix : np.ndarray shape (len(sampled), n_agents)
    diagnostics   : per-trial aggregated diagnostics dict
    """
    sources = [a.name for a in agents]
    K = len(agents)
    n = len(sampled)

    # --- Budget-consistent memory load ---
    for agent in agents:
        agent.load_sampled_memory(cube.get(agent.name, {}), policies, sampled)

    weight_matrix = np.zeros((n, K))
    agg   = {p: 0.0 for p in policies}
    total = {p: 0.0 for p in policies}

    # Diagnostic accumulators
    total_abstentions  = 0
    total_fallbacks    = 0
    fallback_types: dict[str, int] = {}
    total_downweights  = 0
    total_active       = 0

    total_loo_starved   = 0
    total_loo_neighbors = 0.0

    for ti, t in enumerate(sampled):
        weights, diag = compute_consensus_weights(
            agents, t,
            k=k,
            allow_abstain=allow_abstain,
            message_rounds=message_rounds,
            exclude_target=t,          # LOO: exclude query target from its own evidence
        )
        weight_matrix[ti] = weights

        total_abstentions    += sum(diag["abstained"])
        total_active         += diag["active_count"]
        total_downweights    += diag["downweight_count"]
        total_loo_starved    += diag["loo_starved_count"]
        total_loo_neighbors  += diag["loo_mean_neighbors"]
        if diag["fallback_used"]:
            total_fallbacks += 1
            ft = diag["fallback_type"]
            fallback_types[ft] = fallback_types.get(ft, 0) + 1

        for p in policies:
            score_at_t = 0.0
            w_used = 0.0
            for si, src in enumerate(sources):
                v = cube.get(src, {}).get(p, {}).get(t)
                if v is not None:
                    score_at_t += weights[si] * v
                    w_used     += weights[si]
            if w_used > 0.0:
                agg[p]   += score_at_t / w_used
                total[p] += 1.0

    final = {p: agg[p] / max(total[p], 1e-9) for p in policies}
    rank  = sorted(policies, key=lambda p: final[p])

    # Memory diagnostics (after load_sampled_memory)
    # memory_coverage uses unique-target count, not raw n (which may contain bootstrap duplicates)
    memory_sizes    = [a.memory_size     for a in agents]
    memory_coverage = [a.memory_coverage for a in agents]

    total_queries = n * K
    diag_out = {
        "abstention_rate":      total_abstentions  / max(total_queries, 1),
        "fallback_rate":        total_fallbacks    / max(n, 1),
        "fallback_types":       fallback_types,
        "downweight_rate":      total_downweights  / max(total_queries, 1),
        "mean_active_agents":   total_active       / max(n, 1),
        "memory_sizes":         memory_sizes,
        "mean_memory_size":     float(np.mean(memory_sizes)),
        "mean_memory_coverage": float(np.mean(memory_coverage)),
        # LOO diagnostics
        "loo_empty_rate":       total_loo_starved  / max(total_queries, 1),
        "loo_mean_neighbors":   total_loo_neighbors / max(n, 1),
        "sampled_n":            n,
    }
    return rank, weight_matrix, diag_out
