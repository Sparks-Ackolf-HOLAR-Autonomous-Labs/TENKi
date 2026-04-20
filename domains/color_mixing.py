"""
domains/color_mixing.py — Primary TENKi domain.

Wraps the live PEGKi databases produced by the color_mixing_lab engines.
All TENKi experiments (02–10) are supported.

Domain mapping
--------------
policies = ML optimisation policies (grid_search, ucb1_bandit, neural_network, …)
engines  = color-mixing physics simulators
            spectral (Beer-Lambert + CIE D65)
            mixbox   (Kubelka-Munk pigment model)
            km       (measured oil-pigment K/S data)
            ryb      (artist RYB color wheel)
            study_a  (mixbox ∩ RYB gamut intersection)
            study_b  (spectral − RYB gamut difference)
            study_c  (KM − mixbox gamut difference)
score    = best_color_distance_mean per experiment (lower = better)

Key finding (from paper_draft.md §5)
-------------------------------------
Beer-Lambert (spectral), despite being the most physically rigorous engine,
is the WORST knowledge source at low N.  Calibrated Kubelka-Munk (km) is a
better frugal proxy than the nominally higher-fidelity spectral engine.
study_b (spectral−RYB difference) achieves the best bias floor (τ ≈ 0.105)
while study_c (KM−mixbox) has the worst (τ ≈ 0.318).

Real data requirements
-----------------------
Run the PEGKi Phase 1 pipeline for each engine:
    uv run python scripts/generate_policy_data.py --engine spectral --output output/db_spectral
    uv run python scripts/generate_policy_data.py --engine mixbox   --output output/db_mixbox
    uv run python scripts/generate_policy_data.py --engine kubelka_munk --output output/db_km
    uv run python scripts/generate_policy_data.py --engine coloraide_ryb --output output/db_ryb
    # Set-op studies (need shared targets first):
    uv run python scripts/generate_shared_targets.py --n 600 --output output/shared_targets.json
    uv run python scripts/generate_policy_data.py --set-op intersection --engine-a mixbox ...
"""

from __future__ import annotations

from analysis.flip_data import load_many_studies, StudyScores

DEFAULT_STUDY_MAP = {
    "spectral": "output/db_spectral",
    "mixbox":   "output/db_mixbox",
    "km":       "output/db_km",
    "ryb":      "output/db_ryb",
    "study_a":  "output/db_study_a_artist_consensus",
    "study_b":  "output/db_study_b_physics_vs_artist",
    "study_c":  "output/db_study_c_oilpaint_vs_fooddye",
}

# Domain-level metadata for cross-domain comparison report
METADATA = {
    "domain":           "color_mixing",
    "hifi":             "spectral",
    "lower_is_better":  True,
    "score_label":      "best_color_distance_mean",
    "policy_label":     "ML optimisation policy",
    "engine_label":     "physics engine / gamut study",
    "data_source":      "live PEGKi databases (output/db_*)",
    "paradigm_pairs": [
        ("spectral", "km",     "different physical laws (Beer-Lambert vs Kubelka-Munk)"),
        ("mixbox",   "ryb",    "different physical laws (KM pigment vs RYB color wheel)"),
        ("study_a",  "study_b","different gamut regions (intersection vs difference)"),
    ],
}


def load(
    study_map: dict[str, str] | None = None,
    score_key: str = "best_color_distance_mean",
) -> dict[str, StudyScores]:
    """
    Load color-mixing studies from PEGKi databases.

    Returns only studies for which the database directory exists.
    """
    return load_many_studies(study_map or DEFAULT_STUDY_MAP, score_key=score_key)
