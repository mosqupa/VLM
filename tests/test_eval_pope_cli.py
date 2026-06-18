from __future__ import annotations

import os
import subprocess
import sys


def test_eval_pope_help_runs_from_repo_root_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/eval_pope.py", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "Evaluate a Hugging Face LLaVA model on POPE." in result.stdout
