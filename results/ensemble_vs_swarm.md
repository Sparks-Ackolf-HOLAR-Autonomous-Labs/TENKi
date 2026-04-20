# Experiment 12 -- Ensemble vs Swarm

## Setup
- HF reference: `spectral`
- LF sources (3): `mixbox`, `km`, `ryb`
- Total budget per data point: N x K (same for both strategies)

## Strategy definitions

**Ensemble** (no local awareness): equal weight w_S = 1/K applied uniformly to
every target.  The aggregator cannot distinguish which source is locally reliable.

**Swarm** (local awareness): at each target t, source weight
local_w_S(t) = 1 / mean_error(S near t).  Sources that perform better in the
neighbourhood of t contribute more to the ranking at t.

## Tau vs N results

| N per source | Ensemble tau | Swarm tau | Swarm advantage |
|---|---|---|---|
| 1 | 0.701 | 0.721 | +0.020 |
| 5 | 0.784 | 0.781 | -0.003 |
| 10 | 0.824 | 0.842 | +0.018 |
| 50 | 0.826 | 0.841 | +0.016 |
| 100 | 0.833 | 0.832 | -0.001 |
| 500 | 0.821 | 0.828 | +0.007 |
| 1000 | 0.819 | 0.833 | +0.014 |

## Source attention (swarm mean weights)

| Source | Mean weight | vs equal (1/K) |
|---|---|---|
| mixbox | 0.362 | +0.029 |
| km | 0.306 | -0.027 |
| ryb | 0.332 | -0.001 |

## Interpretation

When swarm advantage > 0: local awareness recovers ranking agreement that the
ensemble loses by assigning equal weight to locally unreliable sources.

When swarm advantage ~= 0: sources have uniform competence across target space --
local routing adds overhead without benefit.

The swarm weight heatmap reveals SOURCE SPECIALISATION: a source with highly
variable weights across targets is a local specialist (good in some regions,
poor in others).  Equal weights (ensemble) dilutes such specialists.