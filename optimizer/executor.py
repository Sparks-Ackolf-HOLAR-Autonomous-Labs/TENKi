"""
optimizer/executor.py -- Job execution backends.

The executor is the only component that knows whether time is simulated or real.
The scheduler, policies, and belief layer remain backend-agnostic.

Classes
-------
BaseExecutor      Abstract interface
LocalSimExecutor  Offline simulator: min-heap of completion events, no sleep
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod

from .types import Suggestion, CompletedEval, IntermediateState
from .objective import BaseObjective


class BaseExecutor(ABC):
    @abstractmethod
    def submit(self, suggestion: Suggestion, current_time: float) -> None: ...

    @abstractmethod
    def poll_completed(self, current_time: float) -> list[CompletedEval]: ...

    @abstractmethod
    def cancel(self, job_id: str) -> None: ...

    def submit_resume(
        self,
        intermediate: IntermediateState,
        suggestion: Suggestion,
        current_time: float,
    ) -> None:
        self.submit(suggestion, current_time)

    def next_completion_time(self) -> float | None:
        return None

    def n_inflight(self) -> int:
        return 0

    def shutdown(self) -> None: ...


class LocalSimExecutor(BaseExecutor):
    """
    Offline async simulator.

    Rules:
    - Never sleeps
    - Never spawns real workers or threads
    - Advances a simulated clock via completion-time min-heap
    - Returns completed jobs when the scheduler polls at the right sim time

    This is the reference backend. Swapping it for ThreadExecutor or
    ProcessExecutor later requires only changing this class, not the scheduler.
    """

    def __init__(self, objective: BaseObjective):
        self.objective = objective
        # heap entries: (completion_time, job_id, suggestion, intermediate | None)
        self._heap: list[tuple[float, str, Suggestion, IntermediateState | None]] = []
        self._cancelled: set[str] = set()

    def submit(self, suggestion: Suggestion, current_time: float) -> None:
        completion_time = current_time + suggestion.expected_runtime
        heapq.heappush(
            self._heap,
            (completion_time, suggestion.job_id, suggestion, None),
        )

    def submit_resume(
        self,
        intermediate: IntermediateState,
        suggestion: Suggestion,
        current_time: float,
    ) -> None:
        completion_time = current_time + suggestion.expected_runtime
        heapq.heappush(
            self._heap,
            (completion_time, suggestion.job_id, suggestion, intermediate),
        )

    def poll_completed(self, current_time: float) -> list[CompletedEval]:
        results = []
        while self._heap and self._heap[0][0] <= current_time:
            comp_time, job_id, suggestion, intermediate = heapq.heappop(self._heap)
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                continue
            if intermediate is not None:
                result = self.objective.resume(intermediate, suggestion, comp_time)
            else:
                result = self.objective.evaluate(suggestion, comp_time)
            result.runtime_simulated_end = comp_time
            results.append(result)
        return results

    def next_completion_time(self) -> float | None:
        return self._heap[0][0] if self._heap else None

    def n_inflight(self) -> int:
        return len(self._heap)

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    def shutdown(self) -> None:
        self._heap.clear()
        self._cancelled.clear()
