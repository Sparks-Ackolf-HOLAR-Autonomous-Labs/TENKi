"""
optimizer/runtime_model.py -- Pluggable runtime estimators.

The runtime model is the core approximation for offline async simulation.
It translates (source, fidelity) pairs into simulated wall-clock durations.

Classes
-------
BaseRuntimeModel          Abstract base
ConstantRuntimeModel      Fixed duration regardless of source/fidelity
SourceFidelityRuntimeModel runtime = fidelity * source_multiplier * scale + noise
EmpiricalRuntimeModel     Tracks observed runtimes and returns running means
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRuntimeModel(ABC):
    @abstractmethod
    def estimate(
        self,
        source: str,
        fidelity: int | float,
        context: dict | None = None,
    ) -> float: ...


class ConstantRuntimeModel(BaseRuntimeModel):
    def __init__(self, runtime: float = 1.0):
        self.runtime = runtime

    def estimate(self, source, fidelity, context=None) -> float:
        return self.runtime


class SourceFidelityRuntimeModel(BaseRuntimeModel):
    """
    runtime = fidelity_scale * fidelity * source_multiplier + noise

    source_multipliers: dict mapping source name -> cost multiplier.
    Sources not in the dict default to 1.0.
    Higher fidelity and more expensive sources take longer.
    """

    def __init__(
        self,
        source_multipliers: dict[str, float] | None = None,
        fidelity_scale: float = 1.0,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.source_multipliers = source_multipliers or {}
        self.fidelity_scale = fidelity_scale
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng(0)

    def estimate(self, source, fidelity, context=None) -> float:
        mult = self.source_multipliers.get(source, 1.0)
        base = self.fidelity_scale * float(fidelity) * mult
        if self.noise_std > 0:
            base += float(self.rng.normal(0, self.noise_std))
        return max(base, 0.01)


class EmpiricalRuntimeModel(BaseRuntimeModel):
    """
    Tracks observed runtimes and returns mean estimates.
    Falls back to `default` when no observations exist for a (source, fidelity) pair.
    """

    def __init__(self, default: float = 1.0):
        self.default = default
        self._obs: dict[tuple[str, int | float], list[float]] = {}

    def record(self, source: str, fidelity: int | float, runtime: float) -> None:
        self._obs.setdefault((source, fidelity), []).append(runtime)

    def estimate(self, source, fidelity, context=None) -> float:
        obs = self._obs.get((source, fidelity), [])
        return float(np.mean(obs)) if obs else self.default
