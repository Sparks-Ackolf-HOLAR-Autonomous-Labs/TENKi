"""
domains/ — Domain adapters that translate external case studies into TENKi's
StudyScores format.

Each module exposes a single function:

    load() -> dict[str, StudyScores]

The returned dict maps engine/study names to StudyScores objects compatible
with all TENKi experiments (02–09).  Experiment 10 (aggregation_helps_prediction)
requires trial-level data and is color-mixing specific.

Available domains
-----------------
color_mixing       Primary domain.  Uses live PEGKi databases; all experiments
                   (02–10) supported.  Requires databases to exist under output/.

polymer_hardness   Polymer grades ranked by three hardness measurement scales
                   (Shore A / Shore D / Rockwell R).  Embedded data.
                   Analogue: "does Shore D screening survive the switch to Rockwell R?"

jarvis_leaderboard ML models and one DFT code ranked across eleven JARVIS material
                   property benchmarks.  Embedded representative values.
                   Analogue: "does the best band-gap model stay best on elastic props?"

materials_project  Electron-type families ranked by GGA band gap vs experimental
                   measurement.  Embedded representative values.
                   Analogue: "can cheap DFT identify which electron family to target?"

Adding a new domain
-------------------
1. Create domains/your_domain.py
2. Implement load() -> dict[str, StudyScores]
   - Use analysis.adapters.load_from_score_matrix() or load_from_csv()
3. Document the analogy to the color-mixing case
4. Register in this __init__.py
"""

from domains.color_mixing    import load as load_color_mixing
from domains.polymer_hardness  import load as load_polymer_hardness
from domains.jarvis_leaderboard import load as load_jarvis_leaderboard
from domains.materials_project  import load as load_materials_project

REGISTRY: dict[str, callable] = {
    "color_mixing":      load_color_mixing,
    "polymer_hardness":  load_polymer_hardness,
    "jarvis_leaderboard": load_jarvis_leaderboard,
    "materials_project": load_materials_project,
}
