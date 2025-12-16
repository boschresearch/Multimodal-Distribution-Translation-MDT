# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch
import torch.nn.functional as F


def l2norm(t):
    return F.normalize(t, dim=-1)


def calculate_clip_loss(B1, B2, temperature=0.07):
    # --- our clip loss --- #
    """
    Computes the CLIP loss for two batches of vectors.

    Args:
        B1 (torch.Tensor): Batch 1 of embeddings with shape (B, D).
        B2 (torch.Tensor): Batch 2 of embeddings with shape (B, D).
        temperature (float): Scaling factor for the logits.

    Returns:
        torch.Tensor: The computed CLIP loss.
    """
    B1 = B1.reshape(B1.shape[0], -1)
    B2 = B2.reshape(B1.shape[0], -1)

    # Normalize the vectors to unit length
    B1_norm = F.normalize(B1, dim=1)
    B2_norm = F.normalize(B2, dim=1)

    # Compute the cosine similarity matrix
    logits_per_B1 = torch.matmul(B1_norm, B2_norm.t()) / temperature
    logits_per_B2 = logits_per_B1.t()  # Transpose for the reverse direction

    # Create ground truth labels (diagonal matches)
    batch_size = B1.shape[0]
    labels = torch.arange(batch_size).to(B1.device)

    # Compute cross-entropy loss for both directions
    loss_B1 = F.cross_entropy(logits_per_B1, labels)
    loss_B2 = F.cross_entropy(logits_per_B2, labels)

    # Return the average of the two losses
    return (loss_B1 + loss_B2) / 2
