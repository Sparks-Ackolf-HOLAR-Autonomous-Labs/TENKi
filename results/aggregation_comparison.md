# Experiment 12 -- Source Aggregation Method Comparison

## Fairness contract

- All methods receive the **same sampled draw** per bootstrap trial.
- Swarm agent memory is built from that draw only (no oracle knowledge).
- Total budget is identical: B = N x K for all methods.
- Memory model: Version A (source-level, policy-agnostic mean error per target).

## Setup
- HF reference: `spectral`
- LF sources (3): `mixbox`, `km`, `ryb`

## Method definitions

**ensemble**: 1/K weight applied uniformly. Blind to local source reliability.

**local_router**: stateless kNN. weight_S(t) = 1/mean_error(S near t). No state, no memory between targets.

**swarm_memory**: stateful agents with kNN memory loaded from sampled targets. Weights = confidence/predicted_error. TenKi priors gate authority.

**swarm_memory_abstain**: as above + agents with low neighbourhood confidence stay silent. Fallback: TenKi-reputation weights (not equal).

**swarm_consensus**: full swarm: memory + abstention + one-round message-passing. Agents deviating >1.5 sigma from consensus downweighted.

## Tau vs N (mean, vs ensemble delta)

| N/source | ensemble | local router | swarm memory | swarm memory abstain | swarm consensus |
|---|---|---|---|---|---|
| 1 | 0.708 | 0.714 (+0.006) | 0.716 (+0.008) | 0.716 (+0.008) | 0.716 (+0.008) |
| 5 | 0.806 | 0.809 (+0.003) | 0.810 (+0.005) | 0.809 (+0.003) | 0.809 (+0.003) |
| 10 | 0.825 | 0.832 (+0.006) | 0.836 (+0.011) | 0.835 (+0.010) | 0.835 (+0.009) |
| 50 | 0.824 | 0.830 (+0.006) | 0.841 (+0.017) | 0.841 (+0.017) | 0.840 (+0.016) |
| 100 | 0.831 | 0.833 (+0.003) | 0.833 (+0.003) | 0.833 (+0.003) | 0.833 (+0.003) |
| 500 | 0.817 | 0.831 (+0.014) | 0.833 (+0.016) | 0.833 (+0.016) | 0.833 (+0.016) |
| 1000 | 0.817 | 0.833 (+0.015) | 0.833 (+0.016) | 0.833 (+0.016) | 0.833 (+0.016) |

## Swarm diagnostics (mean across all N and bootstrap trials)

| Mode | mem/agent | coverage | abstention | fallback | downweight | active |
|---|---|---|---|---|---|---|
| swarm_memory | 53.3 | 1.00 | 0.000 | 0.143 | 0.000 | 2.57 |
| swarm_memory_abstain | 53.3 | 1.00 | 0.317 | 0.301 | 0.000 | 2.05 |
| swarm_consensus | 53.3 | 1.00 | 0.317 | 0.301 | 0.087 | 2.05 |

## Swarm success criterion

- **swarm_memory** vs local_router at N=1000: delta=+0.001  → TIES local_router
- **swarm_memory_abstain** vs local_router at N=1000: delta=+0.001  → TIES local_router
- **swarm_consensus** vs local_router at N=1000: delta=+0.001  → TIES local_router

## TenKi structural trust (built-in priors)

| Source | donor_score | bias_floor | ceiling |
|---|---|---|---|
| mixbox | +0.142 | 0.111 | 0.889 |
| km | -0.107 | 0.222 | 0.778 |
| ryb | +0.042 | 0.056 | 0.944 |

## Local Router: source attention

| Source | Mean weight | vs equal |
|---|---|---|
| mixbox | 0.362 | +0.029 |
| km | 0.306 | -0.027 |
| ryb | 0.332 | -0.001 |