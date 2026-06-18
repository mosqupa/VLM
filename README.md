# VLM From The Bottom Up

This repository is a learning lab for understanding vision-language models from their architectural primitives. The first target model is `llava-hf/llava-1.5-7b-hf`, because its core path is still simple enough to inspect directly:

```text
image -> vision tower -> projector -> language-model embedding space -> causal LM
```

## Repository Layout

```text
notes/                    Architecture notes and reading logs.
scripts/                  Runnable inspection and inference utilities.
src/vlm_from_scratch/     Minimal building blocks reimplemented for study.
experiments/images/       Local test images.
experiments/outputs/      Generated traces and logs, ignored by git.
models/                   Downloaded Hugging Face model files, ignored by git.
```

## First Model

Primary target:

- `llava-hf/llava-1.5-7b-hf`

Local path after download:

```text
models/llava-hf/llava-1.5-7b-hf/
```

## Setup

Python 3.10 or 3.11 is recommended for the current PyTorch and Transformers ecosystem.

```bash
pip install -r requirements.txt
```

On this machine, the existing `FSDrive` conda environment is already usable:

```bash
conda run -n FSDrive python scripts/inspect_llava_config.py \
  --model-path models/llava-hf/llava-1.5-7b-hf
```

Download the model:

```bash
hf download llava-hf/llava-1.5-7b-hf \
  --local-dir models/llava-hf/llava-1.5-7b-hf
```

If the local proxy at `127.0.0.1:7897` is not running, clear proxy variables and use the mirror endpoint:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  hf download llava-hf/llava-1.5-7b-hf \
  --local-dir models/llava-hf/llava-1.5-7b-hf \
  --max-workers 4
```

## Learning Path

1. Inspect the config and understand which submodules exist.
2. Run one image-question example through the official Transformers model.
3. Trace the tensor shapes around the vision tower, projector, and language model input embeddings.
4. Reimplement the projector and image-token merge in `src/vlm_from_scratch`.
5. Compare the minimal implementation against the official model behavior.

## Useful Commands

Inspect local model config:

```bash
conda run -n FSDrive python scripts/inspect_llava_config.py \
  --model-path models/llava-hf/llava-1.5-7b-hf
```

Run a single inference:

```bash
conda run -n FSDrive python scripts/run_llava_infer.py \
  --model-path models/llava-hf/llava-1.5-7b-hf \
  --image experiments/images/example.jpg \
  --prompt "USER: <image>\nWhat is in this image?\nASSISTANT:" \
  --load-in-4bit
```

Trace tensor shapes:

```bash
conda run -n FSDrive python scripts/trace_llava_tensors.py \
  --model-path models/llava-hf/llava-1.5-7b-hf \
  --image experiments/images/example.jpg
```

Run a small POPE smoke test:

```bash
conda run -n FSDrive bash scripts/eval_pope.sh random --limit 2
```

Run the full POPE random split:

```bash
conda run -n FSDrive bash scripts/eval_pope.sh random
```

Run all three POPE-COCO splits:

```bash
conda run -n FSDrive bash scripts/eval_pope.sh all
```

Dump the full model architecture without loading weights:

```bash
conda run -n FSDrive python scripts/dump_llava_architecture.py \
  --model-path models/llava-hf/llava-1.5-7b-hf \
  --output notes/03_llava_full_architecture.md
```

## Evaluation Data

Local benchmark data is intentionally ignored by git. The current POPE layout is:

```text
data/benchmarks/pope/
  source/                 Official RUCAIBox/POPE checkout.
  coco/
    coco_pope_random.json
    coco_pope_popular.json
    coco_pope_adversarial.json
    images/               500 referenced COCO val2014 images.
    meta/
      images.txt
      image_urls.txt
```
