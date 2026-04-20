"""
optimizer/policies.py -- Job proposal policies.

Each policy owns job suggestion logic for a specific allocation strategy.
The policy has read-only access to RunState and proposes the next Suggestion.

Classes
-------
BasePolicy            Abstract interface: propose(run_state, worker_id) -> Suggestion | None
SingleSourceMFPolicy  HF vs one LF, fidelity ladder, promotion timing
EnsembleMFPolicy      Global source weights, acquisition-value driven
SwarmMFPolicy         Local source routing via UCB-style competence estimates
MFBOPolicy            GP surrogate with cost-aware LCB/EI acquisition
MFMCPolicy            MFMC optimal allocation (Peherstorfer et al. 2016/2018)
HyperbandPolicy       Successive halving / Hyperband
SMACPolicy            Random Forest surrogate (SMAC-style)

Key design constraint: ensemble and swarm are kept as distinct classes, not
collapsed into a single generic averaging routine. The distinction lives in
the optimizer itself, not only in post-hoc reports.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

import numpy as np

from .types import RunState, Suggestion


class BasePolicy(ABC):
    @abstractmethod
    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None: ...


class SingleSourceMFPolicy(BasePolicy):
    """
    Baseline MF optimizer: one LF source + one HF source.

    Strategy:
    1. Evaluate LF at all fidelity levels for all policies (cheap).
    2. Once LF is exhausted, evaluate HF at the highest fidelity.
    3. Acquisition: tau_belief / expected_runtime.
    """

    def __init__(
        self,
        hifi_source: str,
        lf_source: str,
        policies: list[str],
        fidelity_levels: list[int | float],
        runtime_model,
        rng: np.random.Generator | None = None,
    ):
        self.hifi_source = hifi_source
        self.lf_source = lf_source
        self.policies = list(policies)
        self.fidelity_levels = sorted(fidelity_levels)
        self.runtime_model = runtime_model
        self.rng = rng or np.random.default_rng(42)
        self._done: set[tuple[str, str, int | float]] = set()

    def _running_keys(self, run_state: RunState) -> set[tuple[str, str, int | float]]:
        return {(j.source, j.policy_name, j.fidelity) for j in run_state.running_jobs}

    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None:
        running = self._running_keys(run_state)
        # Phase 1: LF at all fidelities
        for pol in self.policies:
            for fid in self.fidelity_levels:
                key = (self.lf_source, pol, fid)
                if key not in self._done and key not in running:
                    self._done.add(key)
                    return Suggestion(
                        job_id=str(uuid.uuid4()),
                        source=self.lf_source,
                        policy_name=pol,
                        fidelity=fid,
                        expected_runtime=self.runtime_model.estimate(self.lf_source, fid),
                        reason=f"lf_sweep fid={fid}",
                    )
        # Phase 2: HF at max fidelity
        for pol in self.policies:
            fid = self.fidelity_levels[-1]
            key = (self.hifi_source, pol, fid)
            if key not in self._done and key not in running:
                self._done.add(key)
                return Suggestion(
                    job_id=str(uuid.uuid4()),
                    source=self.hifi_source,
                    policy_name=pol,
                    fidelity=fid,
                    expected_runtime=self.runtime_model.estimate(self.hifi_source, fid),
                    reason="hf_eval",
                )
        return None


class EnsembleMFPolicy(BasePolicy):
    """
    Global source weights, no per-target adaptation.

    Ensemble decisions:
    - how much budget to allocate per source
    - how much budget to allocate per fidelity
    - one global source-weight vector updated from beliefs

    Acquisition: (tau_belief * source_weight) / expected_runtime
    """

    def __init__(
        self,
        sources: list[str],
        policies: list[str],
        fidelity_levels: list[int | float],
        runtime_model,
        rng: np.random.Generator | None = None,
    ):
        self.sources = list(sources)
        self.policies = list(policies)
        self.fidelity_levels = sorted(fidelity_levels)
        self.runtime_model = runtime_model
        self.rng = rng or np.random.default_rng(42)
        self._done: set[tuple[str, str, int | float]] = set()

    def _source_weights(self, run_state: RunState) -> dict[str, float]:
        weights: dict[str, float] = {}
        for src in self.sources:
            b = next(
                (b for b in run_state.beliefs if b.source_name == src and b.fidelity is None),
                None,
            )
            weights[src] = max(b.quality_score, 0.01) if (b and b.quality_score is not None) else 1.0
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _value(
        self,
        src: str,
        pol: str,
        fid: int | float,
        run_state: RunState,
        source_weights: dict[str, float],
    ) -> float:
        runtime = self.runtime_model.estimate(src, fid)
        b = next(
            (b for b in run_state.beliefs if b.source_name == src and b.fidelity == fid),
            None,
        )
        pegki_value = (
            b.effective_fidelity
            if (b and b.effective_fidelity is not None)
            else (b.quality_score if (b and b.quality_score is not None) else 0.5)
        )
        return (pegki_value * source_weights.get(src, 1.0)) / max(runtime, 0.01)

    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None:
        running = {(j.source, j.policy_name, j.fidelity) for j in run_state.running_jobs}
        sw = self._source_weights(run_state)
        best_val, best_key = -1.0, None
        for src in self.sources:
            for pol in self.policies:
                for fid in self.fidelity_levels:
                    key = (src, pol, fid)
                    if key in self._done or key in running:
                        continue
                    val = self._value(src, pol, fid, run_state, sw)
                    if val > best_val:
                        best_val, best_key = val, key
        if best_key is None:
            return None
        src, pol, fid = best_key
        self._done.add(best_key)
        return Suggestion(
            job_id=str(uuid.uuid4()),
            source=src,
            policy_name=pol,
            fidelity=fid,
            expected_value=best_val,
            expected_runtime=self.runtime_model.estimate(src, fid),
            reason=f"ensemble_mf val={best_val:.3f} w={sw.get(src, 1.0):.3f}",
        )


class SwarmMFPolicy(BasePolicy):
    """
    Local source routing: choose source based on local competence estimates.

    Swarm decisions (distinct from ensemble):
    - which source to query for this specific context/target
    - which fidelity to use for that source
    - exploit strong local specialists vs explore uncertain sources

    Local competence: UCB-style score = tau_belief + explore / sqrt(n_evals)
    This extends the local-awareness idea from experiments/12_ensemble_vs_swarm.py
    into an online adaptive policy.
    """

    def __init__(
        self,
        sources: list[str],
        policies: list[str],
        fidelity_levels: list[int | float],
        runtime_model,
        explore_weight: float = 0.2,
        rng: np.random.Generator | None = None,
    ):
        self.sources = list(sources)
        self.policies = list(policies)
        self.fidelity_levels = sorted(fidelity_levels)
        self.runtime_model = runtime_model
        self.explore_weight = explore_weight
        self.rng = rng or np.random.default_rng(42)
        self._done: set[tuple[str, str, int | float]] = set()
        self._evals_per_source: dict[str, int] = {s: 0 for s in sources}

    def _local_competence(self, src: str, run_state: RunState) -> float:
        n = self._evals_per_source.get(src, 0)
        b = next(
            (b for b in run_state.beliefs if b.source_name == src and b.fidelity is None),
            None,
        )
        exploit = (
            b.effective_fidelity
            if (b and b.effective_fidelity is not None)
            else (b.quality_score if (b and b.quality_score is not None) else 0.5)
        )
        explore = self.explore_weight / max(1.0, float(n) ** 0.5)
        return exploit + explore

    def _value(self, src: str, pol: str, fid: int | float, run_state: RunState) -> float:
        runtime = self.runtime_model.estimate(src, fid)
        comp = self._local_competence(src, run_state)
        return comp / max(runtime, 0.01)

    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None:
        running = {(j.source, j.policy_name, j.fidelity) for j in run_state.running_jobs}
        best_val, best_key = -1.0, None
        for src in self.sources:
            for pol in self.policies:
                for fid in self.fidelity_levels:
                    key = (src, pol, fid)
                    if key in self._done or key in running:
                        continue
                    val = self._value(src, pol, fid, run_state)
                    if val > best_val:
                        best_val, best_key = val, key
        if best_key is None:
            return None
        src, pol, fid = best_key
        self._done.add(best_key)
        self._evals_per_source[src] = self._evals_per_source.get(src, 0) + 1
        comp = self._local_competence(src, run_state)
        return Suggestion(
            job_id=str(uuid.uuid4()),
            source=src,
            policy_name=pol,
            fidelity=fid,
            expected_value=best_val,
            expected_runtime=self.runtime_model.estimate(src, fid),
            reason=f"swarm_mf comp={comp:.3f} n={self._evals_per_source[src]}",
        )


class MFBOPolicy(BasePolicy):
    """
    Multi-Fidelity Bayesian Optimization.

    Fits a GP surrogate over (source, policy, fidelity) -> score.
    Acquisition: cost-aware LCB = (mu - kappa*sigma) / runtime
    Falls back to round-robin warm-up when fewer than `min_obs_for_gp` observations exist.

    Requires sklearn (soft dependency).
    """

    def __init__(
        self,
        sources: list[str],
        policies: list[str],
        fidelity_levels: list[int | float],
        runtime_model,
        acquisition: str = "lcb",   # "lcb" or "ei"
        kappa: float = 2.0,
        min_obs_for_gp: int = 5,
        rng: np.random.Generator | None = None,
    ):
        self.sources = list(sources)
        self.policies = list(policies)
        self.fidelity_levels = sorted(fidelity_levels)
        self.runtime_model = runtime_model
        self.acquisition = acquisition
        self.kappa = kappa
        self.min_obs_for_gp = min_obs_for_gp
        self.rng = rng or np.random.default_rng(42)
        self._done: set[tuple] = set()
        self._gp = None

    def _encode(self, source: str, policy: str, fidelity: int | float) -> list[float]:
        src_i = self.sources.index(source) / max(len(self.sources) - 1, 1)
        pol_i = self.policies.index(policy) / max(len(self.policies) - 1, 1)
        fid_levels = self.fidelity_levels
        fid_i = fid_levels.index(fidelity) / max(len(fid_levels) - 1, 1) if fidelity in fid_levels else 0.5
        return [src_i, pol_i, fid_i]

    def _fit_gp(self, run_state: RunState) -> None:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            self._gp = None
            return

        obs = [
            c for c in run_state.completed_jobs
            if c.source in self.sources
            and c.policy_name in self.policies
            and not (c.score != c.score)  # exclude nan
        ]
        if len(obs) < self.min_obs_for_gp:
            self._gp = None
            return

        X = np.array([self._encode(c.source, c.policy_name, c.fidelity) for c in obs])
        y = np.array([c.score for c in obs])
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), n_restarts_optimizer=3, normalize_y=True, random_state=0
        )
        try:
            gp.fit(X, y)
            self._gp = gp
        except Exception:
            self._gp = None

    def _acq_value(
        self,
        source: str,
        policy: str,
        fidelity: int | float,
        best_observed: float,
        gp,
    ) -> float:
        x = np.array([self._encode(source, policy, fidelity)])
        mu, sigma = gp.predict(x, return_std=True)
        mu, sigma = float(mu[0]), float(sigma[0])
        if self.acquisition == "ei":
            from scipy.stats import norm
            z = (best_observed - mu) / max(sigma, 1e-9)
            ei = (best_observed - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return float(ei)
        # LCB (lower confidence bound; negate for maximization, lower score = better)
        return -(mu - self.kappa * sigma)

    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None:
        self._fit_gp(run_state)
        running = {(j.source, j.policy_name, j.fidelity) for j in run_state.running_jobs}

        candidates = [
            (src, pol, fid)
            for src in self.sources
            for pol in self.policies
            for fid in self.fidelity_levels
            if (src, pol, fid) not in self._done and (src, pol, fid) not in running
        ]
        if not candidates:
            return None

        if self._gp is None:
            # Warm-up: round-robin at lowest fidelity
            low_fid = self.fidelity_levels[0]
            for src, pol, fid in candidates:
                if fid == low_fid:
                    self._done.add((src, pol, fid))
                    return Suggestion(
                        job_id=str(uuid.uuid4()),
                        source=src, policy_name=pol, fidelity=fid,
                        expected_runtime=self.runtime_model.estimate(src, fid),
                        reason="mfbo_warmup",
                    )
            # Fall back to first candidate
            src, pol, fid = candidates[0]
            self._done.add((src, pol, fid))
            return Suggestion(
                job_id=str(uuid.uuid4()),
                source=src, policy_name=pol, fidelity=fid,
                expected_runtime=self.runtime_model.estimate(src, fid),
                reason="mfbo_warmup_fallback",
            )

        valid_obs = [c.score for c in run_state.completed_jobs if not (c.score != c.score)]
        best_obs = min(valid_obs) if valid_obs else 0.0

        best_val, best_key = -np.inf, None
        for src, pol, fid in candidates:
            acq = self._acq_value(src, pol, fid, best_obs, self._gp)
            runtime = self.runtime_model.estimate(src, fid)
            val = acq / max(runtime, 0.01)
            if val > best_val:
                best_val, best_key = val, (src, pol, fid)

        if best_key is None:
            return None
        src, pol, fid = best_key
        self._done.add(best_key)
        return Suggestion(
            job_id=str(uuid.uuid4()),
            source=src, policy_name=pol, fidelity=fid,
            expected_value=best_val,
            expected_runtime=self.runtime_model.estimate(src, fid),
            reason=f"mfbo_{self.acquisition} val={best_val:.3f}",
        )


class MFMCPolicy(BasePolicy):
    """
    Multi-Fidelity Monte Carlo (Peherstorfer et al. 2016/2018).

    Computes optimal LF/HF allocation ratios from PEGKi transfer beliefs.

    Phase 1 (warm-up): evaluate all sources at lowest fidelity.
    Phase 2: compute optimal ratios; allocate remaining budget proportionally.

    Allocation rule (two-level per LF source):
        r_l = rho_l * sqrt(w_HF / w_l) / sqrt(1 - rho_l^2)
    where rho_l = PEGKi score-space transferability for the LF source,
          w_l   = runtime of lf_source at lowest fidelity,
          w_HF  = runtime of hf_source at highest fidelity.
    """

    def __init__(
        self,
        hifi_source: str,
        lf_sources: list[str],
        fidelity_levels: list[int | float],
        runtime_model,
        n_warmup_per_source: int = 3,
        budget_total: float = 100.0,
        rng: np.random.Generator | None = None,
    ):
        self.hifi_source = hifi_source
        self.lf_sources = list(lf_sources)
        self.all_sources = [hifi_source] + self.lf_sources
        self.fidelity_levels = sorted(fidelity_levels)
        self.runtime_model = runtime_model
        self.n_warmup_per_source = n_warmup_per_source
        self.budget_total = budget_total
        self.rng = rng or np.random.default_rng(42)
        self._done: set[tuple] = set()
        self._warmup_counts: dict[str, int] = {s: 0 for s in self.all_sources}
        self._allocation_counts: dict[str, int] = {}  # source -> target total evals
        self._allocation_computed = False

    def _compute_allocation(self, run_state: RunState) -> None:
        """Compute MFMC optimal allocation ratios from PEGKi source beliefs."""
        hf_fid = self.fidelity_levels[-1]
        lf_fid = self.fidelity_levels[0]

        w_hf = self.runtime_model.estimate(self.hifi_source, hf_fid)
        budget_used = run_state.metrics.get("budget_used", 0.0)
        budget_remaining = self.budget_total - budget_used

        # Get tau beliefs for each LF source
        ratios: dict[str, float] = {}
        for lf in self.lf_sources:
            b = next(
                (b for b in run_state.beliefs if b.source_name == lf and b.fidelity is None),
                None,
            )
            tau = b.tau_mean if (b and b.tau_mean is not None) else 0.5
            rho = max(min(tau, 0.999), 0.001)  # tau ≈ rho in rank correlation sense
            w_lf = self.runtime_model.estimate(lf, lf_fid)
            # MFMC allocation ratio: how many LF evals per HF eval
            r = rho * (w_hf / max(w_lf, 0.01)) ** 0.5 / max((1 - rho**2) ** 0.5, 0.01)
            ratios[lf] = max(r, 1.0)

        # Distribute budget: N_HF is baseline, N_LF = N_HF * ratio
        cost_per_hf_unit = w_hf + sum(ratios[lf] * self.runtime_model.estimate(lf, lf_fid) for lf in self.lf_sources)
        n_hf = max(int(budget_remaining * (w_hf / max(cost_per_hf_unit, 0.01)) / w_hf), 1)

        self._allocation_counts[self.hifi_source] = n_hf
        for lf in self.lf_sources:
            self._allocation_counts[lf] = max(int(n_hf * ratios[lf]), 1)

        self._allocation_computed = True

    def _compute_allocation(self, run_state: RunState) -> None:
        """Compute MFMC optimal allocation ratios from PEGKi source beliefs."""
        hf_fid = self.fidelity_levels[-1]
        lf_fid = self.fidelity_levels[0]

        w_hf = self.runtime_model.estimate(self.hifi_source, hf_fid)
        budget_used = run_state.metrics.get("budget_used", 0.0)
        budget_remaining = self.budget_total - budget_used

        ratios: dict[str, float] = {}
        for lf in self.lf_sources:
            belief = next(
                (b for b in run_state.beliefs if b.source_name == lf and b.fidelity is None),
                None,
            )
            tau = belief.tau_mean if (belief and belief.tau_mean is not None) else 0.5
            rho = (
                belief.rho_mean
                if (belief and belief.rho_mean is not None)
                else max(min((tau + 1.0) / 2.0, 0.999), 0.001)
            )
            quality = (
                belief.effective_fidelity
                if (belief and belief.effective_fidelity is not None)
                else (
                    belief.quality_score
                    if (belief and belief.quality_score is not None)
                    else max(min((tau + 1.0) / 2.0, 1.0), 0.1)
                )
            )
            rho = max(min(rho * (0.5 + 0.5 * quality), 0.999), 0.001)
            w_lf = self.runtime_model.estimate(lf, lf_fid)
            r = rho * (w_hf / max(w_lf, 0.01)) ** 0.5 / max((1 - rho**2) ** 0.5, 0.01)
            ratios[lf] = max(r, 1.0)

        cost_per_hf_unit = w_hf + sum(
            ratios[lf] * self.runtime_model.estimate(lf, lf_fid)
            for lf in self.lf_sources
        )
        n_hf = max(int(budget_remaining * (w_hf / max(cost_per_hf_unit, 0.01)) / w_hf), 1)

        self._allocation_counts[self.hifi_source] = n_hf
        for lf in self.lf_sources:
            self._allocation_counts[lf] = max(int(n_hf * ratios[lf]), 1)

        self._allocation_computed = True

    def _warmup_complete(self) -> bool:
        return all(cnt >= self.n_warmup_per_source for cnt in self._warmup_counts.values())

    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None:
        running = {(j.source, j.policy_name, j.fidelity) for j in run_state.running_jobs}

        # Phase 1: warm up — evaluate all sources at lowest fidelity
        if not self._warmup_complete():
            low_fid = self.fidelity_levels[0]
            for src in self.all_sources:
                if self._warmup_counts[src] >= self.n_warmup_per_source:
                    continue
                key = (src, f"warmup_{self._warmup_counts[src]}", low_fid)
                if key not in self._done and key not in running:
                    self._done.add(key)
                    self._warmup_counts[src] += 1
                    return Suggestion(
                        job_id=str(uuid.uuid4()),
                        source=src,
                        policy_name="__warmup__",
                        fidelity=low_fid,
                        expected_runtime=self.runtime_model.estimate(src, low_fid),
                        reason=f"mfmc_warmup src={src}",
                    )

        # Phase 2: compute allocation once, then submit accordingly
        if not self._allocation_computed:
            self._compute_allocation(run_state)

        # Allocate: LF sources at lowest fidelity, HF at highest
        for src in self.lf_sources:
            target = self._allocation_counts.get(src, 1)
            completed = sum(1 for c in run_state.completed_jobs if c.source == src)
            if completed < target:
                low_fid = self.fidelity_levels[0]
                key = (src, f"mfmc_{completed}", low_fid)
                if key not in self._done and key not in running:
                    self._done.add(key)
                    return Suggestion(
                        job_id=str(uuid.uuid4()),
                        source=src,
                        policy_name="__mfmc__",
                        fidelity=low_fid,
                        expected_runtime=self.runtime_model.estimate(src, low_fid),
                        reason=f"mfmc_lf src={src} {completed}/{target}",
                    )

        hf_target = self._allocation_counts.get(self.hifi_source, 1)
        hf_completed = sum(1 for c in run_state.completed_jobs if c.source == self.hifi_source)
        if hf_completed < hf_target:
            high_fid = self.fidelity_levels[-1]
            key = (self.hifi_source, f"mfmc_{hf_completed}", high_fid)
            if key not in self._done and key not in running:
                self._done.add(key)
                return Suggestion(
                    job_id=str(uuid.uuid4()),
                    source=self.hifi_source,
                    policy_name="__mfmc__",
                    fidelity=high_fid,
                    expected_runtime=self.runtime_model.estimate(self.hifi_source, high_fid),
                    reason=f"mfmc_hf {hf_completed}/{hf_target}",
                )

        return None


class HyperbandPolicy(BasePolicy):
    """
    Hyperband / Successive Halving over (source, policy) candidates.

    Bracket structure:
    - Start all (source, policy) candidates at lowest fidelity
    - Keep top 1/eta fraction by score
    - Promote survivors to next fidelity level
    - Repeat until max fidelity

    This is SHA (Successive Halving Algorithm); Hyperband runs multiple SHA
    brackets with different starting budgets (implemented as a single bracket here).
    """

    def __init__(
        self,
        sources: list[str],
        policies: list[str],
        fidelity_levels: list[int | float],
        runtime_model,
        eta: int = 3,
        rng: np.random.Generator | None = None,
    ):
        self.sources = list(sources)
        self.policies = list(policies)
        self.fidelity_levels = sorted(fidelity_levels)
        self.runtime_model = runtime_model
        self.eta = eta
        self.rng = rng or np.random.default_rng(42)

        # All candidates: (source, policy)
        self._all_candidates: list[tuple[str, str]] = [
            (s, p) for s in sources for p in policies
        ]
        # Bracket state
        self._bracket: list[tuple[str, str]] = list(self._all_candidates)
        self._fidelity_idx: int = 0
        self._done_at_level: set[tuple[str, str, int | float]] = set()
        self._awaiting_results: set[tuple[str, str]] = set()
        self._level_results: dict[tuple[str, str], float] = {}

    def _current_fidelity(self) -> int | float:
        if self._fidelity_idx < len(self.fidelity_levels):
            return self.fidelity_levels[self._fidelity_idx]
        return self.fidelity_levels[-1]

    def _advance_bracket(self, run_state: RunState) -> None:
        """Check if current level is done; if so, eliminate bottom (1 - 1/eta) and advance."""
        fid = self._current_fidelity()
        # Collect results for current bracket at current fidelity
        for src, pol in list(self._bracket):
            key = (src, pol, fid)
            if key in self._done_at_level:
                score = next(
                    (c.score for c in run_state.completed_jobs
                     if c.source == src and c.policy_name == pol and c.fidelity == fid
                     and not (c.score != c.score)),
                    None,
                )
                if score is not None:
                    self._level_results[(src, pol)] = score

        # If all bracket members have results for this level, advance
        all_done = all(
            (src, pol, fid) in self._done_at_level and (src, pol) in self._level_results
            for src, pol in self._bracket
        )
        if all_done and self._fidelity_idx < len(self.fidelity_levels) - 1:
            # Keep top 1/eta by score (lower = better)
            ranked = sorted(self._bracket, key=lambda sp: self._level_results.get(sp, np.inf))
            n_keep = max(1, len(ranked) // self.eta)
            self._bracket = ranked[:n_keep]
            self._level_results = {}
            self._fidelity_idx += 1

    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None:
        self._advance_bracket(run_state)
        fid = self._current_fidelity()
        running = {(j.source, j.policy_name, j.fidelity) for j in run_state.running_jobs}

        for src, pol in self._bracket:
            key = (src, pol, fid)
            if key not in self._done_at_level and key not in running:
                self._done_at_level.add(key)
                return Suggestion(
                    job_id=str(uuid.uuid4()),
                    source=src, policy_name=pol, fidelity=fid,
                    expected_runtime=self.runtime_model.estimate(src, fid),
                    reason=f"hyperband fid={fid} bracket={len(self._bracket)}",
                )

        return None


class SMACPolicy(BasePolicy):
    """
    SMAC-style optimization using a Random Forest surrogate.

    RF models score ~ f(source, policy, fidelity) using tree ensemble.
    Uncertainty estimated from std of per-tree predictions.
    Acquisition: cost-aware LCB = (mu - kappa*sigma) / runtime

    Advantages over GP: handles categorical inputs well, cheaper to fit,
    no kernel hyperparameter optimization needed.

    Requires sklearn (soft dependency).
    """

    def __init__(
        self,
        sources: list[str],
        policies: list[str],
        fidelity_levels: list[int | float],
        runtime_model,
        kappa: float = 1.5,
        n_estimators: int = 50,
        min_obs_for_fit: int = 5,
        rng: np.random.Generator | None = None,
    ):
        self.sources = list(sources)
        self.policies = list(policies)
        self.fidelity_levels = sorted(fidelity_levels)
        self.runtime_model = runtime_model
        self.kappa = kappa
        self.n_estimators = n_estimators
        self.min_obs_for_fit = min_obs_for_fit
        self.rng = rng or np.random.default_rng(42)
        self._done: set[tuple] = set()
        self._rf = None

    def _encode(self, source: str, policy: str, fidelity: int | float) -> list[float]:
        src_i = float(self.sources.index(source))
        pol_i = float(self.policies.index(policy))
        fid_levels = self.fidelity_levels
        fid_i = float(fid_levels.index(fidelity)) if fidelity in fid_levels else float(len(fid_levels) - 1)
        return [src_i, pol_i, fid_i]

    def _fit_rf(self, run_state: RunState) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            self._rf = None
            return

        obs = [
            c for c in run_state.completed_jobs
            if c.source in self.sources
            and c.policy_name in self.policies
            and not (c.score != c.score)
        ]
        if len(obs) < self.min_obs_for_fit:
            self._rf = None
            return

        X = np.array([self._encode(c.source, c.policy_name, c.fidelity) for c in obs])
        y = np.array([c.score for c in obs])
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=0, n_jobs=1
        )
        try:
            rf.fit(X, y)
            self._rf = rf
        except Exception:
            self._rf = None

    def _predict(self, source: str, policy: str, fidelity: int | float) -> tuple[float, float]:
        x = np.array([self._encode(source, policy, fidelity)])
        preds = np.array([tree.predict(x)[0] for tree in self._rf.estimators_])
        return float(preds.mean()), float(preds.std())

    def propose(self, run_state: RunState, worker_id: str) -> Suggestion | None:
        self._fit_rf(run_state)
        running = {(j.source, j.policy_name, j.fidelity) for j in run_state.running_jobs}

        candidates = [
            (src, pol, fid)
            for src in self.sources
            for pol in self.policies
            for fid in self.fidelity_levels
            if (src, pol, fid) not in self._done and (src, pol, fid) not in running
        ]
        if not candidates:
            return None

        if self._rf is None:
            # Warm-up: enumerate at lowest fidelity
            low_fid = self.fidelity_levels[0]
            for src, pol, fid in candidates:
                if fid == low_fid:
                    self._done.add((src, pol, fid))
                    return Suggestion(
                        job_id=str(uuid.uuid4()),
                        source=src, policy_name=pol, fidelity=fid,
                        expected_runtime=self.runtime_model.estimate(src, fid),
                        reason="smac_warmup",
                    )
            src, pol, fid = candidates[0]
            self._done.add((src, pol, fid))
            return Suggestion(
                job_id=str(uuid.uuid4()),
                source=src, policy_name=pol, fidelity=fid,
                expected_runtime=self.runtime_model.estimate(src, fid),
                reason="smac_warmup_fallback",
            )

        best_val, best_key = -np.inf, None
        for src, pol, fid in candidates:
            mu, sigma = self._predict(src, pol, fid)
            runtime = self.runtime_model.estimate(src, fid)
            # LCB for minimization; negate to maximize
            val = -(mu - self.kappa * sigma) / max(runtime, 0.01)
            if val > best_val:
                best_val, best_key = val, (src, pol, fid)

        if best_key is None:
            return None
        src, pol, fid = best_key
        self._done.add(best_key)
        mu, sigma = self._predict(src, pol, fid)
        return Suggestion(
            job_id=str(uuid.uuid4()),
            source=src, policy_name=pol, fidelity=fid,
            expected_value=best_val,
            expected_runtime=self.runtime_model.estimate(src, fid),
            reason=f"smac lcb mu={mu:.2f} sigma={sigma:.2f}",
        )
