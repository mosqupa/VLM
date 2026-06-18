from __future__ import annotations

import json
from pathlib import Path

from vlm_eval.benchmarks.pope import load_pope_questions
from vlm_eval.metrics.classification import evaluate_yes_no, normalize_yes_no


def test_load_pope_questions_reads_jsonl_records(tmp_path: Path) -> None:
    label_file = tmp_path / "coco_pope_random.json"
    records = [
        {
            "question_id": 1,
            "image": "COCO_val2014_000000016631.jpg",
            "text": "Is there a person in the image?",
            "label": "yes",
        },
        {
            "question_id": 2,
            "image": "COCO_val2014_000000016631.jpg",
            "text": "Is there a refrigerator in the image?",
            "label": "no",
        },
    ]
    label_file.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    questions = load_pope_questions(label_file)

    assert [question.question_id for question in questions] == [1, 2]
    assert [question.image for question in questions] == [
        "COCO_val2014_000000016631.jpg",
        "COCO_val2014_000000016631.jpg",
    ]
    assert [question.label for question in questions] == ["yes", "no"]


def test_normalize_yes_no_extracts_first_clear_answer() -> None:
    assert normalize_yes_no("Yes, there is a dog.") == "yes"
    assert normalize_yes_no("No. The image does not show one.") == "no"
    assert normalize_yes_no("I cannot determine from the image.") == "unknown"


def test_evaluate_yes_no_reports_binary_metrics() -> None:
    labels = ["yes", "no", "yes", "no"]
    answers = ["yes", "yes", "no", "no"]

    metrics = evaluate_yes_no(labels, answers)

    assert metrics.total == 4
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.yes_ratio == 0.5
