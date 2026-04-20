# Aggregation Helps Prediction? (Experiment 10)

HF reference: **spectral**  

Train/test split: 73 / 31 spectral targets  


Replicates the core question from Vriza et al. (2024, d3dd00207a) inside TENKi's framework:

> Does adding low-fidelity training data to a high-fidelity training set improve or degrade action prediction accuracy on held-out spectral targets?


## Verdicts by model type

| Model | Mean DeltaMAE (concat - spectral_only) | Verdict |
|-------|-----------------------------------|---------|
| Ridge | +4.584 | DEGRADES  (concat hurts) |
| RandomForest | +4.103 | DEGRADES  (concat hurts) |
| GradientBoosting | +4.837 | DEGRADES  (concat hurts) |

## Per-source MAE table

### LF source: mixbox

| Model | spectral_only | concat | weighted | lf_only | Delta(concat-spec) |
|-------|--------------|--------|----------|---------|----------------|
| Ridge | 2.786 | 7.580 | 6.074 | 12.445 | +4.794 |
| RandomForest | 3.329 | 10.913 | 9.525 | 12.802 | +7.584 |
| GradientBoosting | 3.164 | 12.511 | 9.370 | 12.784 | +9.347 |

### LF source: km

| Model | spectral_only | concat | weighted | lf_only | Delta(concat-spec) |
|-------|--------------|--------|----------|---------|----------------|
| Ridge | 2.786 | 6.821 | 6.353 | 11.195 | +4.035 |
| RandomForest | 3.329 | 12.091 | 10.741 | 13.880 | +8.761 |
| GradientBoosting | 3.164 | 14.163 | 11.606 | 14.420 | +10.999 |

### LF source: ryb

| Model | spectral_only | concat | weighted | lf_only | Delta(concat-spec) |
|-------|--------------|--------|----------|---------|----------------|
| Ridge | 2.786 | 8.778 | 7.754 | 14.376 | +5.992 |
| RandomForest | 3.329 | 12.071 | 10.303 | 14.188 | +8.741 |
| GradientBoosting | 3.164 | 13.802 | 10.192 | 14.046 | +10.638 |

### LF source: study_a

| Model | spectral_only | concat | weighted | lf_only | Delta(concat-spec) |
|-------|--------------|--------|----------|---------|----------------|
| Ridge | 2.786 | 9.346 | 9.415 | 13.946 | +6.560 |
| RandomForest | 3.329 | 3.132 | 3.439 | 6.009 | -0.198 |
| GradientBoosting | 3.164 | 2.734 | 2.708 | 5.344 | -0.430 |

### LF source: study_b

| Model | spectral_only | concat | weighted | lf_only | Delta(concat-spec) |
|-------|--------------|--------|----------|---------|----------------|
| Ridge | 2.786 | 2.686 | 2.639 | 2.814 | -0.100 |
| RandomForest | 3.329 | 2.103 | 2.178 | 5.175 | -1.227 |
| GradientBoosting | 3.164 | 1.970 | 2.096 | 5.346 | -1.194 |

### LF source: study_c

| Model | spectral_only | concat | weighted | lf_only | Delta(concat-spec) |
|-------|--------------|--------|----------|---------|----------------|
| Ridge | 2.786 | 9.010 | 8.803 | 19.453 | +6.224 |
| RandomForest | 3.329 | 4.287 | 4.670 | 15.037 | +0.957 |
| GradientBoosting | 3.164 | 2.823 | 3.114 | 14.950 | -0.341 |

## Interpretation

- **DEGRADES**: Naive concatenation raises action MAE -- the LF source's action patterns conflict with spectral physics. Matches the paper's finding for classical ML.
- **IMPROVES**: Concatenation lowers action MAE -- the LF source covers parts of the action space the spectral training set missed.
- **NEUTRAL**: No statistically meaningful change. Common for sources with high Kendall tau ceiling vs spectral (exp 03).

Cross-reference with `flip_feasibility.json` external_ceilings: sources with low ceiling tend to DEGRADE; sources with high ceiling tend to NEUTRAL or IMPROVE.
