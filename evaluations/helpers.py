# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

from evaluations.shapenet import ShapeNetEvaluator
from evaluations.sr import SuperResolutionEvaluator
from utils.names import Datasets


def load_evaluations(dataset, device):
    if dataset == Datasets.ShapeNet.value:
        evaluator = ShapeNetEvaluator(device)
    elif dataset == Datasets.SR.value:
        evaluator = SuperResolutionEvaluator()
    else:
        raise NotImplementedError(f'Evaluator for {dataset} dataset not implemented.')

    return evaluator
