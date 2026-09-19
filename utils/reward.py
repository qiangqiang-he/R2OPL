"""Rule-based reward used by PG-OPD training and validation."""

from __future__ import annotations

from typing import Any

from utils.answer_verifier import extract_final_answer, verify_response_answer


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    del data_source, extra_info
    _, valid_format = extract_final_answer(solution_str)
    score = verify_response_answer(solution_str, str(ground_truth))
    return {
        "score": float(score),
        "acc": float(score),
        "format_valid": float(valid_format),
    }


__all__ = ["compute_score"]
