from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local or remote LLaVA config.")
    parser.add_argument("--model-path", default="models/llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    config = AutoConfig.from_pretrained(model_path)

    interesting = {
        "model_type": getattr(config, "model_type", None),
        "image_token_index": getattr(config, "image_token_index", None),
        "vision_feature_layer": getattr(config, "vision_feature_layer", None),
        "vision_feature_select_strategy": getattr(config, "vision_feature_select_strategy", None),
        "projector_hidden_act": getattr(config, "projector_hidden_act", None),
        "vision_hidden_size": getattr(getattr(config, "vision_config", None), "hidden_size", None),
        "text_hidden_size": getattr(getattr(config, "text_config", None), "hidden_size", None),
        "vocab_size": getattr(getattr(config, "text_config", None), "vocab_size", None),
    }

    print(json.dumps(interesting, indent=2, ensure_ascii=False))
    print("\nFull architecture config keys:")
    print(json.dumps(sorted(config.to_dict().keys()), indent=2))


if __name__ == "__main__":
    main()
