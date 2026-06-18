from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


class HFLlavaRunner:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path)
        model_kwargs: dict[str, object] = {
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        self.input_device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, image_path: str | Path, prompt: str, *, max_new_tokens: int = 16) -> str:
        image = Image.open(Path(image_path)).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.input_device)
        input_token_count = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        new_tokens = output_ids[0][input_token_count:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
