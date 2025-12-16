# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch as th


def vp_logsnr(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -th.log((0.5 * beta_d * (t**2) + beta_min * t).exp() - 1)


def vp_logs(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -0.25 * t**2 * (beta_d) - 0.5 * t * beta_min
