from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LLaVA image-text generation.")
    parser.add_argument("--model-path", default="models/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--prompt",
        default="USER: <image>\nWhat is in this image?\nASSISTANT:",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model_kwargs = {
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    image = Image.open(Path(args.image)).convert("RGB")
    input_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=args.prompt, images=image, return_tensors="pt").to(input_device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    print(processor.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
