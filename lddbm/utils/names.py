# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

from enum import Enum


class Datasets(Enum):
    ShapeNet = "ShapeNet"
    SR = "SuperResolution"


class TrainingStrategy(Enum):
    WholeSystemTraining = "WholeSystemTraining"
    IterativeTraining = "IterativeTraining"


class ReconstructionLoss(Enum):
    Reconstruction = "Reconstruction"
    Predictive = "Predictive"


class DistanceMetric(Enum):
    MSE = "MSE"
    LPIPS = "LPIPS"


class BridgeModelsTyps(Enum):
    BridgeTransformer = "BridgeTransformer"


# ----------- models types -------------- #
class Encoders(Enum):
    Conv3DEncoder = "Conv3DEncoder"
    MV2DEncoder = "MV2DEncoder"
    Identity = "Identity"
    KlVaePreTrainedEncoder16 = "KlVaePreTrainedEncoder16"
    KlVaePreTrainedEncoder128 = "KlVaePreTrainedEncoder128"


class Decoders(Enum):
    NoDecoder = "NoDecoder"
    Conv3DDecoder = "Conv3DDecoder"
    KlVaePreTrainedDecoder128 = "KlVaePreTrainedDecoder128"
    Identity = "Identity"
