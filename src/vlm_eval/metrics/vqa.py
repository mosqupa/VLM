from __future__ import annotations

import re
import string
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")
_ARTICLES = {"a", "an", "the"}


@dataclass(frozen=True)
class ExactMatchMetrics:
    total: int
    correct: int
    accuracy: float


def normalize_short_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    if normalized.startswith("answer:"):
        normalized = normalized.split(":", 1)[1].strip()
    for prefix in ("the answer is", "it is", "it's"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    normalized = normalized.strip(string.whitespace + string.punctuation)
    words = [word for word in _WHITESPACE_RE.split(normalized) if word and word not in _ARTICLES]
    return " ".join(words)


def evaluate_exact_match(labels: list[str], answers: list[str]) -> ExactMatchMetrics:
    if len(labels) != len(answers):
        raise ValueError(f"labels and answers must have the same length: {len(labels)} != {len(answers)}")
    correct = sum(
        1
        for label, answer in zip(labels, answers)
        if normalize_short_answer(label) == normalize_short_answer(answer)
    )
    total = len(labels)
    return ExactMatchMetrics(total=total, correct=correct, accuracy=correct / total if total else 0.0)
