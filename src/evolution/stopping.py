"""Stopping criteria for prompt evolution."""

from __future__ import annotations

from typing import Any


class StoppingState:
    """Track stopping criteria state across evolution iterations."""

    def __init__(self, *, max_iterations: int, patience: int) -> None:
        self.max_iterations = max_iterations
        self.patience = patience
        self.iteration: int = 0
        self.no_improvement_streak: int = 0
        self.stop_reason: str | None = None

    def record_iteration(self, *, improved: bool) -> None:
        self.iteration += 1
        if improved:
            self.no_improvement_streak = 0
        else:
            self.no_improvement_streak += 1

    def should_stop(self) -> tuple[bool, str | None]:
        if self.iteration >= self.max_iterations:
            return True, f"max_iterations={self.max_iterations} reached"
        if self.no_improvement_streak >= self.patience:
            return True, (
                f"patience={self.patience} exceeded "
                f"({self.no_improvement_streak} consecutive non-improvements)"
            )
        return False, None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "patience": self.patience,
            "iteration": self.iteration,
            "no_improvement_streak": self.no_improvement_streak,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoppingState":
        state = cls(
            max_iterations=data["max_iterations"],
            patience=data["patience"],
        )
        state.iteration = data.get("iteration", 0)
        state.no_improvement_streak = data.get("no_improvement_streak", 0)
        state.stop_reason = data.get("stop_reason")
        return state
