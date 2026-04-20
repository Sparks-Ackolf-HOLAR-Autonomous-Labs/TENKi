#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
REPO="C:/Users/kinst/OneDrive/Utah/vibe/color_mixing_lab"
LOG="results/tenki_1000/run_open_questions.log"
mkdir -p results/tenki_1000

run_exp() {
    local script="$1"
    echo "=============================" | tee -a "$LOG"
    echo "START: $script  $(date)" | tee -a "$LOG"
    echo "=============================" | tee -a "$LOG"
    cd "$REPO"
    uv run python "extended/gamut_symmetry/TENKi/$script" 2>&1 | tee -a "extended/gamut_symmetry/TENKi/$LOG"
    echo "DONE: $script  $(date)" | tee -a "extended/gamut_symmetry/TENKi/$LOG"
}

run_exp experiments/15_q1_mirror_pair_test.py
run_exp experiments/16_q2_epsilon_symmetry.py
run_exp experiments/17_q3_nash_equilibrium.py
run_exp experiments/18_q4_diversity_allocation.py
run_exp experiments/19_q5_robot_difficulty.py
run_exp experiments/20_q6_per_policy_rho.py
run_exp experiments/21_q7_trueskill2.py
run_exp experiments/22_q9_flip_difficulty.py
run_exp experiments/23_q10_swarm_specialists.py

echo "ALL DONE $(date)" | tee -a "$REPO/extended/gamut_symmetry/TENKi/$LOG"
