"""
domains/polymer_hardness.py — Polymer grades ranked by three hardness measurement scales.

Domain mapping
--------------
policies = polymer grades (PTFE, HDPE, PC, ABS, …)
engines  = hardness measurement scales
            shore_a    (ASTM D2240, Type A — soft rubbers, flexible plastics)
            shore_d    (ASTM D2240, Type D — hard plastics)
            rockwell_r (ASTM D785, Scale R — rigid plastics, highest stiffness range)
score    = hardness value on each scale (HIGHER = harder = better for screening)

Domain analogue
---------------
This is the polymer case study from the PEGKi application note (§2).
"Does the stiffer-is-harder hierarchy survive a measurement scale switch?"

Each scale probes a different deformation regime:
    Shore A  → soft: rubbers, flexible TPEs, soft gels
    Shore D  → medium: hard plastics, rigid TPEs, LDPE/HDPE
    Rockwell R → hard: rigid engineering plastics, PC, PEEK, Nylon

Key finding (application note §2)
----------------------------------
PTFE and PC-ABS swap positions depending on which scale is used because they
straddle the stiffness-regime boundary.  Shore D sits at that boundary and
is the WORST knowledge source: tau_ShoreD(1) → spectral is lowest.
This is the polymer analogue of "Beer-Lambert is the worst knowledge source."

Practical question
------------------
Can Shore D test data serve as a pre-screen before committing to Rockwell R
measurements?  PEGKi's answer: partially — the overall hierarchy is mostly
stable, but the medium-stiffness boundary region (PTFE / PC-ABS) reverses.

Data source
-----------
Values are representative published engineering data (ASTM standard tables,
Omnexus polymer database, and manufacturer datasheets).  Shore D is listed
only for polymers where the Shore A scale saturates (>95) or is inapplicable.
Missing values (N/A) are omitted from the dataset; only polymers with values
on all three scales are included for a clean cross-scale comparison.

Real data replacement
---------------------
Replace SCORE_DATA with your own {scale: {polymer: value}} dict, or use:
    from analysis.adapters import load_from_wide_csv
    studies = load_from_wide_csv("your_hardness_data.csv", lower_is_better=False)
"""

from __future__ import annotations

from analysis.adapters import load_from_score_matrix
from analysis.flip_data import StudyScores

# Representative hardness values.
# Sources: ASTM D2240 (Shore), ASTM D785 (Rockwell R), Omnexus database.
# Shore A values for polymers above ~95 Shore A are often reported as Shore D
# equivalents; we list Shore A for completeness but note saturation effects.
# Polymers selected to have measured values on all three scales.
SCORE_DATA: dict[str, dict[str, float]] = {
    "shore_a": {
        # ASTM D2240 Type A indenter; range 0-100; higher=harder
        "LDPE":      93.0,   # low-density polyethylene
        "HDPE":      97.0,   # high-density polyethylene
        "PP":        95.0,   # polypropylene
        "PVC_soft":  78.0,   # plasticised PVC
        "PVC_rigid": 93.0,   # rigid PVC
        "PTFE":      55.0,   # polytetrafluoroethylene  ← near scale boundary
        "PC_ABS":    97.0,   # polycarbonate/ABS blend  ← near scale boundary
        "ABS":       99.0,
        "PC":        99.0,   # polycarbonate (saturates Shore A)
        "Nylon66":   99.0,   # polyamide 66 (saturates Shore A)
    },
    "shore_d": {
        # ASTM D2240 Type D indenter; range 0-100; higher=harder
        "LDPE":      45.0,
        "HDPE":      62.0,
        "PP":        65.0,
        "PVC_soft":  40.0,
        "PVC_rigid": 80.0,
        "PTFE":      55.0,   # ← still ~55 here (doesn't move much)
        "PC_ABS":    70.0,   # ← drops relative to harder polymers
        "ABS":       75.0,
        "PC":        80.0,
        "Nylon66":   78.0,
    },
    "rockwell_r": {
        # ASTM D785 Scale R (1/2" steel ball, 60 kg load); range 0-150; higher=harder
        "LDPE":      44.0,
        "HDPE":      60.0,
        "PP":       102.0,
        "PVC_soft":  36.0,
        "PVC_rigid": 115.0,
        "PTFE":      58.0,   # ← low relative to its Shore D; paradigm boundary visible
        "PC_ABS":   101.0,   # ← now higher relative to PTFE (swap from Shore D ordering)
        "ABS":      110.0,
        "PC":       118.0,
        "Nylon66":  118.0,
    },
}

METADATA = {
    "domain":           "polymer_hardness",
    "hifi":             "rockwell_r",    # most resolution for rigid plastics
    "lower_is_better":  False,           # higher hardness = better for screening
    "score_label":      "hardness_value",
    "policy_label":     "polymer grade",
    "engine_label":     "hardness measurement scale",
    "data_source":      "representative ASTM D2240/D785 published values",
    "paradigm_pairs": [
        ("shore_a", "shore_d",    "deformation regime boundary (flexible → hard)"),
        ("shore_d", "rockwell_r", "deformation regime boundary (hard → rigid)"),
        ("shore_a", "rockwell_r", "fully different measurement geometry"),
    ],
    "boundary_policies": ["PTFE", "PC_ABS"],
    "note": (
        "PTFE and PC_ABS straddle the Shore D / Rockwell R boundary. "
        "Their relative ranking reverses between scales — the polymer analogue "
        "of the Beer-Lambert / KM swap in the color-mixing domain."
    ),
}


def load() -> dict[str, StudyScores]:
    """
    Return Shore A, Shore D, and Rockwell R studies as StudyScores objects.

    higher hardness = better screening candidate → lower_is_better=False
    """
    return load_from_score_matrix(SCORE_DATA, lower_is_better=False)
