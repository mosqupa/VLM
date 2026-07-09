from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_eval.benchmarks.pope import build_pope_prompt, load_pope_questions
from vlm_eval.metrics.classification import evaluate_yes_no, normalize_yes_no


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face LLaVA model on POPE.")
    parser.add_argument("--model-path", default="models/llava-1.5-7b-hf")
    parser.add_argument("--label-file", default="data/benchmarks/pope/coco/coco_pope_random.json")
    parser.add_argument("--image-root", default="data/benchmarks/pope/coco/images")
    parser.add_argument("--output", default="outputs/pope_random_answers.jsonl")
    parser.add_argument("--metrics-output", default="outputs/pope_random_metrics.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")

    args = parser.parse_args()

    questions = load_pope_questions(args.label_file)
    if args.limit is not None:
        questions = questions[: args.limit]

    output_path = Path(args.output)
    metrics_path = Path(args.metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    from vlm_eval.runners.hf_llava import HFLlavaRunner

    runner = HFLlavaRunner(
        args.model_path,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
    )

    answers: list[str] = []
    labels: list[str] = []
    image_root = Path(args.image_root)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, question in enumerate(questions, start=1):
            image_path = image_root / question.image
            prompt = build_pope_prompt(question.text)
            answer = runner.generate(image_path, prompt, max_new_tokens=args.max_new_tokens)
            normalized_answer = normalize_yes_no(answer)
            labels.append(question.label)
            answers.append(answer)

            handle.write(
                json.dumps(
                    {
                        "question_id": question.question_id,
                        "image": question.image,
                        "question": question.text,
                        "label": question.label,
                        "answer": answer,
                        "normalized_answer": normalized_answer,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            print(f"[{index}/{len(questions)}] label={question.label} answer={normalized_answer} raw={answer!r}")

    metrics = evaluate_yes_no(labels, answers)
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(asdict(metrics), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
