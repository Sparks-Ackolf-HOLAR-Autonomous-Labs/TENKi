"""
optimizer/ -- Async multi-fidelity optimizer for PEGKi/TENKi.

Modules
-------
types          Core dataclasses (Config, Suggestion, CompletedEval, BeliefState, ...)
runtime_model  Pluggable runtime estimators for offline async simulation
objective      BaseObjective + OfflineReplayObjective (PEGKi database replay)
executor       BaseExecutor + LocalSimExecutor (simulated clock, no sleep)
store          RunStore -- JSON-based persistent run state
pegki_bridge   Belief updates: tau/rho, bias floor, flip probability
policies       BasePolicy + SingleSourceMFPolicy, EnsembleMFPolicy, SwarmMFPolicy
scheduler      AsyncMFScheduler -- event loop + worker dispatch
"""
