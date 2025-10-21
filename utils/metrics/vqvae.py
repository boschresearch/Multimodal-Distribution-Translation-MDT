# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

from torch import Tensor

def compute_vq_loss(quantized: Tensor, z: Tensor, beta: float) -> Tensor:
    commitment_loss = ((quantized.detach() - z) ** 2).mean()
    codebook_loss = ((quantized - z.detach()) ** 2).mean()
    
    vq_loss = codebook_loss + (beta * commitment_loss)
    
    return vq_loss