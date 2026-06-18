from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers.utils import ModelOutput
from transformers import AutoProcessor, LlavaForConditionalGeneration


def tensor_summary(value: torch.Tensor) -> str:
    return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"


def shape_of(value: object, *, max_items: int = 4) -> str:
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, ModelOutput):
        keys = [key for key, item in value.items() if item is not None]
        return f"{type(value).__name__}({', '.join(keys)})"
    if isinstance(value, dict):
        items = list(value.items())
        rendered = [f"{key}={shape_of(item, max_items=max_items)}" for key, item in items[:max_items]]
        if len(items) > max_items:
            rendered.append("...")
        return "{" + ", ".join(rendered) + "}"
    if isinstance(value, (list, tuple)):
        rendered = [shape_of(item, max_items=max_items) for item in value[:max_items]]
        if len(value) > max_items:
            rendered.append("...")
        return f"{type(value).__name__}(len={len(value)}, items=[" + ", ".join(rendered) + "])"
    return type(value).__name__


def print_value(name: str, value: object, *, indent: int = 2, max_depth: int = 2, max_items: int = 4) -> None:
    prefix = " " * indent
    if isinstance(value, torch.Tensor):
        print(f"{prefix}{name}: {tensor_summary(value)}")
        return

    if isinstance(value, ModelOutput):
        print(f"{prefix}{name}: {type(value).__name__}")
        if max_depth <= 0:
            return
        for key, item in value.items():
            if item is not None:
                print_value(key, item, indent=indent + 2, max_depth=max_depth - 1, max_items=max_items)
        return

    if isinstance(value, dict):
        print(f"{prefix}{name}: dict(len={len(value)})")
        if max_depth <= 0:
            return
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                print(f"{prefix}  ...")
                break
            print_value(str(key), item, indent=indent + 2, max_depth=max_depth - 1, max_items=max_items)
        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: {type(value).__name__}(len={len(value)})")
        if max_depth <= 0:
            return
        for idx, item in enumerate(value[:max_items]):
            print_value(f"[{idx}]", item, indent=indent + 2, max_depth=max_depth - 1, max_items=max_items)
        if len(value) > max_items:
            print(f"{prefix}  ...")
        return

    print(f"{prefix}{name}: {value!r} ({type(value).__name__})")


def print_input_token_summary(input_ids: torch.Tensor, attention_mask: torch.Tensor, image_token_id: int) -> None:
    active_tokens = int(attention_mask.sum().item())
    image_tokens = int((input_ids == image_token_id).sum().item())
    print("token summary")
    print(f"  active tokens: {active_tokens}")
    print(f"  image token id: {image_token_id}")
    print(f"  image token count: {image_tokens}")
    print(f"  non-image active tokens: {active_tokens - image_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace major LLaVA tensor shapes.")
    parser.add_argument("--model-path", default="models/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="USER: <image>\nWhat is in this image?\nASSISTANT:")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=4)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    hooks = []

    def add_hook(name: str, module: torch.nn.Module) -> None:
        def hook(
            _module: torch.nn.Module,
            inputs: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            print(f"\n{name}")
            print_value("args", inputs, max_depth=args.max_depth, max_items=args.max_items)
            print_value("kwargs", kwargs, max_depth=args.max_depth, max_items=args.max_items)
            print_value("output", output, max_depth=args.max_depth, max_items=args.max_items)

        hooks.append(module.register_forward_hook(hook, with_kwargs=True))

    add_hook("llava", model)
    add_hook("vision_tower", model.vision_tower)
    add_hook("multi_modal_projector", model.multi_modal_projector)
    add_hook("language_model", model.language_model)

    image = Image.open(Path(args.image)).convert("RGB")
    input_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=args.prompt, images=image, return_tensors="pt").to(input_device)

    print("processor outputs")
    for key, value in inputs.items():
        print(f"  {key}: {shape_of(value)}")
    print_input_token_summary(
        inputs["input_ids"],
        inputs["attention_mask"],
        image_token_id=model.config.image_token_index,
    )
    print("config summary")
    print(f"  vision feature layer: {model.config.vision_feature_layer}")
    print(f"  vision feature select strategy: {model.config.vision_feature_select_strategy}")
    print(f"  image seq length: {getattr(model.config, 'image_seq_length', None)}")
    print(f"  vision hidden size: {model.config.vision_config.hidden_size}")
    print(f"  text hidden size: {model.config.text_config.hidden_size}")

    with torch.inference_mode():
        model(**inputs, use_cache=False)

    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
