from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GqaQuestion:
    question_id: str
    image: str
    text: str
    answer: str | None


def load_gqa_questions(path: str | Path) -> list[GqaQuestion]:
    question_path = Path(path)
    data = json.loads(question_path.read_text(encoding="utf-8"))
    records = data.items() if isinstance(data, dict) else enumerate(data)

    questions: list[GqaQuestion] = []
    for key, record in records:
        question_id = str(record.get("question_id", record.get("questionId", key)))
        image_id = str(record.get("image", record.get("imageId", record.get("image_id", ""))))
        if not image_id:
            raise ValueError(f"{question_path}: question {question_id} is missing imageId")
        image = image_id if image_id.endswith(".jpg") else f"{image_id}.jpg"
        text = str(record.get("question", record.get("text", ""))).strip()
        if not text:
            raise ValueError(f"{question_path}: question {question_id} is missing question text")
        answer = record.get("answer")
        questions.append(
            GqaQuestion(
                question_id=question_id,
                image=image,
                text=text,
                answer=None if answer is None else str(answer),
            )
        )
    return questions


def build_gqa_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nAnswer the question using a single word or phrase.\nASSISTANT:"
