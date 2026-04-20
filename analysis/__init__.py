"""
analysis — Reusable modules for the donor-flip pipeline.

Layers:
  flip_data        — data loading and score normalization
  flip_metrics     — ranking, Kendall tau, bootstrap CI, ceiling estimation
  flip_models      — donor-flip logic and FlipResult objects
  flip_reports     — plots and summary artifacts
  swarm_agents     — stateful SwarmAgent with kNN memory + TenKi trust priors
  swarm_consensus  — consensus aggregation (abstention + message passing)
"""
