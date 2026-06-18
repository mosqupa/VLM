from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PopeQuestion:
    question_id: int
    image: str
    text: str
    label: str


def load_pope_questions(path: str | Path) -> list[PopeQuestion]:
    questions: list[PopeQuestion] = []
    label_path = Path(path)
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                questions.append(
                    PopeQuestion(
                        question_id=int(record["question_id"]),
                        image=str(record["image"]),
                        text=str(record["text"]),
                        label=_normalize_label(record["label"]),
                    )
                )
            except KeyError as error:
                missing = error.args[0]
                raise ValueError(f"{label_path}:{line_number} missing required key: {missing}") from error
    return questions


def build_pope_prompt(question: str) -> str:
    return f"USER: <image>\n{question} Answer with yes or no.\nASSISTANT:"


def _normalize_label(value: object) -> str:
    label = str(value).strip().lower()
    if label not in {"yes", "no"}:
        raise ValueError(f"POPE labels must be 'yes' or 'no', got {value!r}")
    return label
