"""Accept or reject candidate prompts using an epsilon improvement gate."""

from __future__ import annotations

from typing import Any


def evaluate_candidate(
    *,
    prompt_id: str,
    candidate_mrr: float,
    current_best_mrr: float,
    epsilon: float,
    iteration: int,
) -> dict[str, Any]:
    """Return a score record with the acceptance decision.

    Accepts the candidate if candidate_mrr > current_best_mrr + epsilon.
    epsilon prevents noise-based swaps; set in evolution.yaml.
    """
    delta = candidate_mrr - current_best_mrr
    accepted = delta > epsilon
    return {
        "prompt_id": prompt_id,
        "dev_mrr": candidate_mrr,
        "accepted": accepted,
        "previous_best_mrr": current_best_mrr,
        "delta": delta,
        "epsilon": epsilon,
        "iteration": iteration,
    }
