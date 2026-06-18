from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoConfig, LlavaForConditionalGeneration


def format_shape(shape: Iterable[int]) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def format_count(count: int) -> str:
    return f"{count:,}"


def direct_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def recursive_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=True))


def module_extra(module: nn.Module) -> str:
    if isinstance(module, nn.Linear):
        bias = module.bias is not None
        return f"in={module.in_features}, out={module.out_features}, bias={bias}"
    if isinstance(module, nn.Embedding):
        return f"num_embeddings={module.num_embeddings}, embedding_dim={module.embedding_dim}"
    if isinstance(module, nn.LayerNorm):
        return f"normalized_shape={module.normalized_shape}, eps={module.eps}"
    if module.__class__.__name__ == "LlamaRMSNorm":
        return f"eps={getattr(module, 'variance_epsilon', None)}"
    if isinstance(module, nn.Conv2d):
        return (
            f"in_channels={module.in_channels}, out_channels={module.out_channels}, "
            f"kernel_size={module.kernel_size}, stride={module.stride}, bias={module.bias is not None}"
        )
    if isinstance(module, (nn.GELU, nn.SiLU, nn.ReLU)):
        return ""
    return ""


def module_depth(name: str) -> int:
    if not name:
        return 0
    return name.count(".") + 1


def iter_module_rows(model: nn.Module, *, include_param_shapes: bool, max_depth: int | None) -> list[str]:
    rows: list[str] = []
    for name, module in model.named_modules():
        depth = module_depth(name)
        if max_depth is not None and depth > max_depth:
            continue

        display_name = name or "<root>"
        indent = "  " * depth
        direct_params = direct_parameter_count(module)
        total_params = recursive_parameter_count(module)
        extra = module_extra(module)
        suffix = f" | {extra}" if extra else ""
        rows.append(
            f"{indent}- `{display_name}`: `{module.__class__.__name__}`"
            f" | direct params={format_count(direct_params)}"
            f" | subtree params={format_count(total_params)}{suffix}"
        )

        if include_param_shapes:
            for param_name, parameter in module.named_parameters(recurse=False):
                rows.append(
                    f"{indent}  - param `{param_name}`: "
                    f"{format_shape(parameter.shape)}, dtype={parameter.dtype}, device={parameter.device}"
                )
            for buffer_name, buffer in module.named_buffers(recurse=False):
                rows.append(
                    f"{indent}  - buffer `{buffer_name}`: "
                    f"{format_shape(buffer.shape)}, dtype={buffer.dtype}, device={buffer.device}"
                )

    return rows


def section_for_subtree(model: nn.Module, subtree_name: str, *, include_param_shapes: bool) -> list[str]:
    module = model.get_submodule(subtree_name)
    rows = [f"## `{subtree_name}` 结构", ""]
    for name, child in module.named_modules():
        full_name = subtree_name if not name else f"{subtree_name}.{name}"
        depth = module_depth(name)
        indent = "  " * depth
        direct_params = direct_parameter_count(child)
        total_params = recursive_parameter_count(child)
        extra = module_extra(child)
        suffix = f" | {extra}" if extra else ""
        rows.append(
            f"{indent}- `{full_name}`: `{child.__class__.__name__}`"
            f" | direct params={format_count(direct_params)}"
            f" | subtree params={format_count(total_params)}{suffix}"
        )
        if include_param_shapes:
            for param_name, parameter in child.named_parameters(recurse=False):
                rows.append(
                    f"{indent}  - param `{param_name}`: "
                    f"{format_shape(parameter.shape)}, dtype={parameter.dtype}, device={parameter.device}"
                )
    rows.append("")
    return rows


def build_markdown(model: LlavaForConditionalGeneration, model_path: str, include_param_shapes: bool) -> str:
    config = model.config
    vision_config = config.vision_config
    text_config = config.text_config
    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    patch_grid = image_size // patch_size
    patch_tokens = patch_grid * patch_grid

    rows: list[str] = [
        "# LLaVA 1.5 7B 完整架构",
        "",
        f"模型目录：`{model_path}`",
        "",
        "这个文档由 `scripts/dump_llava_architecture.py` 自动生成。模型在 `meta` 设备上按配置实例化，不会加载 14G 权重。",
        "",
        "## 总览",
        "",
        "```text",
        "pixel_values",
        "  -> vision_tower: CLIPVisionModel",
        "  -> selected hidden state",
        "  -> multi_modal_projector",
        "  -> replace <image> token embeddings",
        "  -> language_model: LlamaForCausalLM / Vicuna",
        "  -> logits",
        "```",
        "",
        "## 关键配置",
        "",
        f"- model type: `{config.model_type}`",
        f"- image token id: `{config.image_token_index}`",
        f"- pad token id: `{config.pad_token_id}`",
        f"- image seq length: `{getattr(config, 'image_seq_length', None)}`",
        f"- vision feature layer: `{config.vision_feature_layer}`",
        f"- vision feature select strategy: `{config.vision_feature_select_strategy}`",
        f"- projector hidden activation: `{config.projector_hidden_act}`",
        f"- text model type: `{text_config.model_type}`",
        f"- text hidden size: `{text_config.hidden_size}`",
        f"- vocab size: `{text_config.vocab_size}`",
        f"- max position embeddings: `{text_config.max_position_embeddings}`",
        f"- vision model type: `{vision_config.model_type}`",
        f"- vision image size: `{image_size}`",
        f"- vision patch size: `{patch_size}`",
        f"- vision patch grid: `{patch_grid} x {patch_grid}`",
        f"- vision patch tokens: `{patch_tokens}`",
        f"- vision hidden size: `{vision_config.hidden_size}`",
        f"- vision layers: `{vision_config.num_hidden_layers}`",
        f"- vision attention heads: `{vision_config.num_attention_heads}`",
        "",
        "## 主要张量形状",
        "",
        "以 batch size = 1、单图输入为例：",
        "",
        "```text",
        f"pixel_values                         -> (1, 3, {image_size}, {image_size})",
        f"vision_tower hidden state             -> (1, {patch_tokens + 1}, {vision_config.hidden_size})  # +1 是 CLS token",
        f"selected image features, default      -> (1, {patch_tokens}, {vision_config.hidden_size})",
        f"projected image features              -> (1, {patch_tokens}, {text_config.hidden_size})",
        f"language_model inputs_embeds          -> (1, text_tokens + {patch_tokens}, {text_config.hidden_size})",
        f"language_model logits                 -> (1, text_tokens + {patch_tokens}, {text_config.vocab_size})",
        "```",
        "",
    ]

    for subtree_name in ("vision_tower", "multi_modal_projector", "language_model"):
        rows.extend(section_for_subtree(model, subtree_name, include_param_shapes=include_param_shapes))

    rows.extend(
        [
            "## 全量模块树",
            "",
            "下面是 `model.named_modules()` 的完整展开。`direct params` 是该模块自己直接持有的参数量；`subtree params` 包含所有子模块。",
            "",
        ]
    )
    rows.extend(iter_module_rows(model, include_param_shapes=include_param_shapes, max_depth=None))
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump the full LLaVA architecture into a Markdown document.")
    parser.add_argument("--model-path", default="models/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--output", default="notes/03_llava_full_architecture.md")
    parser.add_argument("--no-param-shapes", action="store_true")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    with init_empty_weights():
        model = LlavaForConditionalGeneration(config)

    markdown = build_markdown(
        model,
        model_path=args.model_path,
        include_param_shapes=not args.no_param_shapes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
