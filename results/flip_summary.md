# Donor-Flip Summary

Generated: 2026-04-06T04:48:22.748702+00:00  

n_bootstrap=300  flip_eps=0.01  hifi=spectral  


## 1. Donor/receiver ordering at N=1 (external reference)

| Source | tau@N=1 vs HF | Ceiling | Role |
|--------|--------------|---------|------|
| study_b | 0.640 | 0.889 | DONOR |
| mixbox | 0.522 | 0.778 | DONOR |
| ryb | 0.448 | 0.833 | RECEIVER |
| km | 0.301 | 0.722 | RECEIVER |
| study_a | 0.206 | 0.944 | RECEIVER |
| study_c | 0.159 | 0.889 | RECEIVER |

## 2. External permanent-gap pairs (physics/calibration limits)

| Source | Competitor | gap_ceiling | Note |
|--------|-----------|-------------|------|
| km | study_b | -0.1667 | physics-limited (irreducible) |
| km | ryb | -0.1111 | physics-limited (irreducible) |
| mixbox | study_b | -0.1111 | physics-limited (irreducible) |
| study_c | study_a | -0.0556 | physics-limited (irreducible) |
| km | mixbox | -0.0556 | physics-limited (irreducible) |
| ryb | study_b | -0.0556 | physics-limited (irreducible) |

## 3. External flippable pairs with N\*

| Source | Competitor | N* | gap_now | gap_ceiling |
|--------|-----------|-----|---------|-------------|
| ryb | mixbox | 100 | -0.0891 | +0.0556 |

## 4. Mutual flip matrix (N* for A to surpass B@N=1)

| Source \ Competitor | spectral | mixbox | km | ryb | study_a | study_b | study_c |
|---|---|---|---|---|---|---|---|
| spectral | — | > | > | > | > | > | > |
| mixbox | 5 | — | > | > | > | 5 | > |
| km | 50 | 20 | — | 15 | > | 50 | > |
| ryb | 20 | 8 | > | — | > | 20 | > |
| study_a | 100 | 50 | 5 | 15 | — | 100 | > |
| study_b | > | > | > | > | > | — | > |
| study_c | 100 | 100 | 15 | 30 | 3 | 100 | — |

## 5. Top candidates for cheap-source replacement

Sources that can replace a stronger donor at minimal experiment cost:

| Source | vs Competitor | N* | Verdict |
|--------|-------------|-----|---------|
| study_c | study_a | 3 | FLIPPABLE |
| mixbox | spectral | 5 | FLIPPABLE |
| mixbox | study_b | 5 | FLIPPABLE |
| study_a | km | 5 | FLIPPABLE |
| ryb | mixbox | 8 | FLIPPABLE |
| km | ryb | 15 | FLIPPABLE |
| study_a | ryb | 15 | FLIPPABLE |
| study_c | km | 15 | FLIPPABLE |
| km | mixbox | 20 | FLIPPABLE |
| ryb | spectral | 20 | FLIPPABLE |
| ryb | study_b | 20 | FLIPPABLE |
| study_c | ryb | 30 | FLIPPABLE |
| km | spectral | 50 | FLIPPABLE |
| km | study_b | 50 | FLIPPABLE |
| study_a | mixbox | 50 | FLIPPABLE |
| study_a | spectral | 100 | FLIPPABLE |
| study_a | study_b | 100 | FLIPPABLE |
| study_c | spectral | 100 | FLIPPABLE |
| study_c | mixbox | 100 | FLIPPABLE |
| study_c | study_b | 100 | FLIPPABLE |
