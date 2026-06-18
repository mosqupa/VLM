from __future__ import annotations

import torch


def replace_image_tokens(
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace image-token positions with precomputed image embeddings.

    This helper covers the simple case where the prompt already contains exactly
    one placeholder token per image embedding token.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq_len]")
    if text_embeddings.ndim != 3:
        raise ValueError("text_embeddings must have shape [batch, seq_len, hidden]")
    if image_embeddings.ndim != 3:
        raise ValueError("image_embeddings must have shape [batch, image_tokens, hidden]")

    batch_size, _, hidden_size = text_embeddings.shape
    if image_embeddings.shape[0] != batch_size or image_embeddings.shape[2] != hidden_size:
        raise ValueError("image_embeddings must match batch size and hidden size")

    merged = text_embeddings.clone()
    image_mask = input_ids == image_token_id

    for batch_idx in range(batch_size):
        token_positions = image_mask[batch_idx].nonzero(as_tuple=False).flatten()
        if token_positions.numel() != image_embeddings.shape[1]:
            raise ValueError(
                "number of image placeholder tokens must equal image_embeddings length "
                f"for batch {batch_idx}: got {token_positions.numel()} placeholders and "
                f"{image_embeddings.shape[1]} image tokens"
            )
        merged[batch_idx, token_positions] = image_embeddings[batch_idx]

    return merged
