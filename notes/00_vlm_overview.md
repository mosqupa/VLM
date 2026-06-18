# VLM Overview

Most autoregressive VLMs can be understood as a bridge between visual tokens and a language model.

## Core Components

- Vision encoder: turns an image into patch-level visual features.
- Multimodal projector: maps visual feature size into the LLM hidden size.
- Token merge: replaces or expands image placeholder tokens with projected visual tokens.
- Language model: performs ordinary causal generation over mixed visual/text embeddings.

## First Target: LLaVA 1.5

LLaVA 1.5 is a good first model because the architecture is direct:

```text
CLIP vision tower -> MLP projector -> LLaMA-family causal LM
```

The important questions to answer while reading the model are:

- Which vision hidden state is selected?
- Is the CLS token kept or dropped?
- How many image tokens are produced per image?
- How are `<image>` placeholders expanded or replaced?
- What shape enters the language model as `inputs_embeds`?
