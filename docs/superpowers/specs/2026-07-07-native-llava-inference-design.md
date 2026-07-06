# Native LLaVA Inference Script Design

## Goal

Provide a minimal project-level command for running one-image inference with the
editable-installed native LLaVA implementation and the local
`models/llava-v1.5-7b` checkpoint.

## Approach

Add a thin Python wrapper around `llava.eval.run_llava.eval_model`. The wrapper
will expose project-friendly defaults while leaving model loading, conversation
template selection, image preprocessing, token insertion, and generation in the
upstream LLaVA implementation.

Add a shell entry point that moves to the repository root, places the local
`LLaVA` checkout first on `PYTHONPATH`, and invokes the Python wrapper. This
ensures imports resolve to the editable source checkout even if the active
environment contains another `llava` installation.

## Interface

The Python script will support:

- `--model-path`, defaulting to `models/llava-v1.5-7b`
- `--image`, required
- `--prompt`, with a simple image-description default
- `--conv-mode`, optional, allowing upstream auto-detection by default
- `--max-new-tokens`
- `--temperature`, defaulting to `0` for deterministic generation
- `--top-p`
- `--num-beams`

The shell script will default to `experiments/images/1.png` and forward any
additional command-line arguments to the Python script.

## Error Handling

The wrapper will validate that the model path and image exist before loading the
model, producing direct path-specific errors. Errors from native LLaVA model
loading and generation will remain visible rather than being masked.

## Verification

Verification will run in the `FSDrive` Conda environment:

1. Compile the new Python script.
2. Confirm `llava` resolves to the local `LLaVA` checkout.
3. Check the CLI help output.
4. Run one real inference using the local native checkpoint and test image.

