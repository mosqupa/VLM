from __future__ import annotations

import re
from dataclasses import dataclass

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


@dataclass(frozen=True)
class YesNoMetrics:
    total: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    yes_ratio: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    unknown: int


def normalize_yes_no(answer: str) -> str:
    match = _YES_NO_RE.search(answer.strip())
    if match is None:
        return "unknown"
    return match.group(1).lower()


def evaluate_yes_no(labels: list[str], answers: list[str]) -> YesNoMetrics:
    if len(labels) != len(answers):
        raise ValueError(f"labels and answers must have the same length: {len(labels)} != {len(answers)}")

    normalized_labels = [_require_yes_no(label, "label") for label in labels]
    normalized_answers = [normalize_yes_no(answer) for answer in answers]

    tp = sum(1 for label, answer in zip(normalized_labels, normalized_answers) if label == "yes" and answer == "yes")
    fp = sum(1 for label, answer in zip(normalized_labels, normalized_answers) if label == "no" and answer == "yes")
    tn = sum(1 for label, answer in zip(normalized_labels, normalized_answers) if label == "no" and answer == "no")
    fn = sum(1 for label, answer in zip(normalized_labels, normalized_answers) if label == "yes" and answer != "yes")
    unknown = sum(1 for answer in normalized_answers if answer == "unknown")

    total = len(normalized_labels)
    correct = tp + tn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    yes_ratio = _safe_div(sum(1 for answer in normalized_answers if answer == "yes"), total)

    return YesNoMetrics(
        total=total,
        accuracy=_safe_div(correct, total),
        precision=precision,
        recall=recall,
        f1=f1,
        yes_ratio=yes_ratio,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        unknown=unknown,
    )


def _require_yes_no(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"yes", "no"}:
        raise ValueError(f"{field_name} must be 'yes' or 'no', got {value!r}")
    return normalized


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
