# LLaVA Architecture Notes

## Data Flow

```text
PIL image + prompt
  -> LlavaProcessor
  -> input_ids, attention_mask, pixel_values
  -> vision_tower(pixel_values)
  -> selected image hidden states
  -> multi_modal_projector(image_features)
  -> merged text/image embeddings
  -> language_model.generate(...)
```

## What To Inspect First

- `model.config.vision_config`
- `model.config.text_config`
- `model.vision_tower`
- `model.multi_modal_projector`
- `model.language_model`
- `model.config.image_token_index`
- `model.config.vision_feature_layer`
- `model.config.vision_feature_select_strategy`

## Expected Shape Questions

For a single image and one prompt:

- `pixel_values`: usually `[batch, channels, height, width]`
- vision hidden states: usually `[batch, num_patches + cls, vision_hidden]`
- projected image features: `[batch, image_tokens, text_hidden]`
- text embeddings: `[batch, text_tokens, text_hidden]`
