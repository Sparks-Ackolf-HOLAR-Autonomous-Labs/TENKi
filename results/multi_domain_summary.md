# TENKi Multi-Domain Donor-Flip Summary

Generated: 2026-04-06T04:49:49.243780+00:00  


## Cross-domain ranking stability

Ordered by mean full-data ceiling tau (ascending = more scrambled).

| Domain | Engines (K) | Policies (N) | Mean ceiling tau | Stability class |
|--------|------------|-------------|----------------|----------------|
| polymer_hardness | 3 | 10 | 0.585 | SCRAMBLED  (tau < 0.60) |
| jarvis_leaderboard | 7 | 12 | 0.758 | PARTIAL    (tau 0.60-0.85) |
| materials_project | 3 | 4 | 0.778 | PARTIAL    (tau 0.60-0.85) |
| color_mixing | 7 | 9 | 0.815 | PARTIAL    (tau 0.60-0.85) |

## Net donor score per engine (mean asymmetry at N=1)

Positive = engine is a better knowledge source per experiment.
Negative = engine needs more experiments to stabilise its ranking.

### polymer_hardness  (HF reference: rockwell_r)

| Engine | Donor score | Role |
|--------|------------|------|
| shore_d | +0.0889 | DONOR |
| rockwell_r | +0.0889 | DONOR |
| shore_a | -0.1778 | RECEIVER |

### jarvis_leaderboard  (HF reference: bulk_modulus)

| Engine | Donor score | Role |
|--------|------------|------|
| band_gap_opt | +0.0000 | NEUTRAL |
| form_energy | +0.0000 | NEUTRAL |
| bulk_modulus | +0.0000 | NEUTRAL |
| shear_modulus | +0.0000 | NEUTRAL |
| ehull | +0.0000 | NEUTRAL |
| exfol_energy | +0.0000 | NEUTRAL |
| band_gap_mbj | +0.0000 | NEUTRAL |

### materials_project  (HF reference: experimental)

| Engine | Donor score | Role |
|--------|------------|------|
| gga | +0.0000 | NEUTRAL |
| hse06 | +0.0000 | NEUTRAL |
| experimental | +0.0000 | NEUTRAL |

### color_mixing  (HF reference: spectral)

| Engine | Donor score | Role |
|--------|------------|------|
| spectral | +0.2569 | DONOR |
| study_b | +0.2309 | DONOR |
| mixbox | +0.1470 | DONOR |
| ryb | +0.0373 | NEUTRAL |
| km | -0.1102 | RECEIVER |
| study_a | -0.2596 | RECEIVER |
| study_c | -0.3022 | RECEIVER |

## External ceiling (source -> HF reference, full data)

Shows the asymptotic upper bound for each engine as a donor vs the domain's high-fidelity reference.

### polymer_hardness  (HF = rockwell_r)

| Engine | Ceiling tau vs HF | Limit type |
|--------|----------------|-----------|
| rockwell_r | 1.0000 | SELF |
| shore_d | 0.8667 | PARTIAL_BIAS |
| shore_a | 0.4667 | PERMANENT_GAP |

### jarvis_leaderboard  (HF = bulk_modulus)

| Engine | Ceiling tau vs HF | Limit type |
|--------|----------------|-----------|
| bulk_modulus | 1.0000 | SELF |
| shear_modulus | 1.0000 | NEAR_PERFECT |
| ehull | 0.6364 | PERMANENT_GAP |
| form_energy | 0.6061 | PERMANENT_GAP |
| exfol_energy | 0.6061 | PERMANENT_GAP |
| band_gap_mbj | 0.6061 | PERMANENT_GAP |
| band_gap_opt | 0.5152 | PERMANENT_GAP |

### materials_project  (HF = experimental)

| Engine | Ceiling tau vs HF | Limit type |
|--------|----------------|-----------|
| experimental | 1.0000 | SELF |
| gga | 0.6667 | PERMANENT_GAP |
| hse06 | 0.6667 | PERMANENT_GAP |

### color_mixing  (HF = spectral)

| Engine | Ceiling tau vs HF | Limit type |
|--------|----------------|-----------|
| spectral | 1.0000 | SELF |
| study_a | 0.9444 | NEAR_PERFECT |
| study_b | 0.8889 | PARTIAL_BIAS |
| study_c | 0.8889 | PARTIAL_BIAS |
| ryb | 0.8333 | PARTIAL_BIAS |
| mixbox | 0.7778 | PARTIAL_BIAS |
| km | 0.7222 | PARTIAL_BIAS |

## Single-score vs multi-replicate domains

**Directional asymmetry** (who is the donor?) requires within-source variance:
multiple experiments per policy so that tau_ij(N=1) != tau_ij(N=full).

- **color_mixing** -- live database, many experiments -> full asymmetry analysis.
- **polymer_hardness / jarvis_leaderboard / materials_project** -- single published
  score per policy -> asymmetry collapses to zero (every bootstrap draw is identical).
  Only **ceiling-level** analysis is available: which engine is structurally the
  better proxy, regardless of how many samples are taken.

To enable sampling-efficiency analysis on external domains, either:
  1. Collect multiple experimental/simulation replicates and pass as `[s1, s2, ...]`
     lists to `load_from_dict()`, or
  2. Add calibrated synthetic noise: `scores[p] = [score + rng.normal(0, sigma)` for _ in range(N)]`.


## What this means for TENKi experiments

| Experiment | Color mixing | Polymer hardness | JARVIS | Materials Project |
|-----------|-------------|-----------------|--------|------------------|
| 02 Transfer matrix | Full (4+ engines) | Full (3 scales) | Full (7 props) | Full (3 fidelities) |
| 03 Flip test | Full (live data) | Ceiling only | Ceiling only | Ceiling only |
| 07 Convergence | Full | N/A (no replicates) | N/A | N/A |
| 09 Mixed-source | Full | N/A | N/A | N/A |
| 10 Aggregation MAE | **Full** (trial-level) | N/A | See note | N/A |

> **Note on Exp 10 for JARVIS**: The trial-level version requires individual structure predictions, not aggregate leaderboard MAEs.  Download the JARVIS dataset and re-run `load_trial_records()` with a custom loader to enable this.
