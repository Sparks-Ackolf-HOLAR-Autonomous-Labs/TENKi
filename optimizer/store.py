"""
optimizer/store.py -- Persistent run state storage.

All optimizer state is persisted to a run directory using JSON files.
This gives:
- deterministic offline replay
- interruption recovery
- transparent debugging
- a clean path to file-based real-async coordination later

Directory layout
----------------
runs/<run_name>/
  config.json
  latest_snapshot.json
  beliefs.json
  completed.jsonl
  snapshots/
    state_00001.json
    state_00002.json
    ...
  intermediate/
    <resume_token>.json
  coordination/
    events.jsonl
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .types import (
    Config,
    RunState,
    WorkerState,
    Suggestion,
    CompletedEval,
    IntermediateState,
    BeliefState,
)


def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict_runstate(raw: dict) -> RunState:
    def _ws(d: dict) -> WorkerState:
        return WorkerState(**{k: v for k, v in d.items() if k in WorkerState.__dataclass_fields__})

    def _sg(d: dict) -> Suggestion:
        return Suggestion(**{k: v for k, v in d.items() if k in Suggestion.__dataclass_fields__})

    def _ce(d: dict) -> CompletedEval:
        return CompletedEval(**{k: v for k, v in d.items() if k in CompletedEval.__dataclass_fields__})

    def _is_(d: dict) -> IntermediateState:
        return IntermediateState(**{k: v for k, v in d.items() if k in IntermediateState.__dataclass_fields__})

    def _bs(d: dict) -> BeliefState:
        return BeliefState(**{k: v for k, v in d.items() if k in BeliefState.__dataclass_fields__})

    return RunState(
        sim_time=raw.get("sim_time", 0.0),
        config=raw.get("config", {}),
        workers=[_ws(w) for w in raw.get("workers", [])],
        pending_jobs=[_sg(j) for j in raw.get("pending_jobs", [])],
        running_jobs=[_sg(j) for j in raw.get("running_jobs", [])],
        completed_jobs=[_ce(j) for j in raw.get("completed_jobs", [])],
        intermediate_states=[_is_(s) for s in raw.get("intermediate_states", [])],
        beliefs=[_bs(b) for b in raw.get("beliefs", [])],
        metrics=raw.get("metrics", {}),
    )


class RunStore:
    """
    Persists optimizer run state and coordination events to disk.

    Each save operation is atomic at the file level (write then rename
    would be ideal; for now we write directly for simplicity).

    Design rule: every state mutation should correspond to a coordination
    event appended to events.jsonl.
    """

    def __init__(self, save_dir: str, run_name: str):
        self.root = Path(save_dir) / run_name
        self._snaps_dir = self.root / "snapshots"
        self._inter_dir = self.root / "intermediate"
        self._coord_dir = self.root / "coordination"
        for d in [self._snaps_dir, self._inter_dir, self._coord_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._snap_n = 0
        self._completed_fh = open(self.root / "completed.jsonl", "a", encoding="utf-8")
        self._events_fh = open(self._coord_dir / "events.jsonl", "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Config

    def save_config(self, config: Config) -> None:
        (self.root / "config.json").write_text(
            json.dumps(_to_dict(config), indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Snapshots

    def save_snapshot(self, run_state: RunState) -> None:
        self._snap_n += 1
        d = _to_dict(run_state)
        body = json.dumps(d, indent=2)
        (self._snaps_dir / f"state_{self._snap_n:05d}.json").write_text(body, encoding="utf-8")
        (self.root / "latest_snapshot.json").write_text(body, encoding="utf-8")

    def load_latest_snapshot(self) -> RunState | None:
        path = self.root / "latest_snapshot.json"
        if not path.exists():
            return None
        return _from_dict_runstate(json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Completed results (append-only JSONL)

    def append_completed(self, results: list[CompletedEval]) -> None:
        for r in results:
            self._completed_fh.write(json.dumps(_to_dict(r)) + "\n")
        self._completed_fh.flush()

    # ------------------------------------------------------------------
    # Intermediate states (resumable promotions)

    def save_intermediate_state(self, state: IntermediateState) -> None:
        path = self._inter_dir / f"{state.resume_token}.json"
        path.write_text(json.dumps(_to_dict(state), indent=2), encoding="utf-8")

    def load_intermediate_state(self, resume_token: str) -> IntermediateState | None:
        path = self._inter_dir / f"{resume_token}.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return IntermediateState(**{k: v for k, v in d.items() if k in IntermediateState.__dataclass_fields__})

    # ------------------------------------------------------------------
    # Beliefs

    def save_beliefs(self, beliefs: list[BeliefState]) -> None:
        (self.root / "beliefs.json").write_text(
            json.dumps([_to_dict(b) for b in beliefs], indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Coordination events (append-only JSONL)

    def log_event(self, event: dict) -> None:
        self._events_fh.write(json.dumps(event) + "\n")
        self._events_fh.flush()

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._completed_fh.close()
        self._events_fh.close()
