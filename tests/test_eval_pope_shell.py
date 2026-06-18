from __future__ import annotations

import subprocess


def test_eval_pope_shell_defaults_to_random_split() -> None:
    result = subprocess.run(
        ["bash", "scripts/eval_pope.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "coco_pope_random.json" in result.stdout
    assert "pope_random_answers.jsonl" in result.stdout
    assert "--load-in-4bit" in result.stdout


def test_eval_pope_shell_expands_all_splits_and_passes_extra_args() -> None:
    result = subprocess.run(
        ["bash", "scripts/eval_pope.sh", "all", "--dry-run", "--limit", "1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "coco_pope_random.json" in result.stdout
    assert "coco_pope_popular.json" in result.stdout
    assert "coco_pope_adversarial.json" in result.stdout
    assert result.stdout.count("--limit 1") == 3
