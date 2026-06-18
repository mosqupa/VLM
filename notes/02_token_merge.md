# Image Token Merge

The crucial VLM operation is not image encoding by itself. The key operation is placing image embeddings into the token stream consumed by the language model.

Conceptually:

```text
input_ids -> token embeddings
image -> vision tower -> projector -> image embeddings
replace <image> positions with image embeddings
feed merged embeddings to the language model
```

Different model families implement this differently:

- Some use one placeholder token that is expanded into many image patch embeddings.
- Some expect the prompt to contain many image placeholder tokens.
- Some insert image embeddings before the text prompt.

For LLaVA, inspect the installed Transformers implementation before assuming the merge behavior.
