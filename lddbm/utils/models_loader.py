# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import os

from omegaconf import OmegaConf

from lddbm.models.bridge.bridge_model import BridgeModel
from lddbm.models.bridge.DDBM.diffusion.karras_diffusion import KarrasDenoiser
from lddbm.models.bridge.DDBM.diffusion.resample import (
    LogNormalSampler, LossSecondMomentResampler, RealUniformSampler,
    UniformSampler)
from lddbm.models.bridge.DDBM.transformer.our_transformer import \
    AutoRegressiveTransformer
from lddbm.models.decoders.shapenet.decoder import Conv3DDecoder
from lddbm.models.decoders.shapenet.decoder import Identity as DecIdentity
from lddbm.models.encoders.shapenet.encoder import Conv3DEncoder
from lddbm.models.encoders.shapenet.encoder import Identity as EncIdentity
from lddbm.models.encoders.shapenet.encoder import MV2DEncoder
from lddbm.models.encoders.super_resolution.encoder import (
    AutoencoderKLInnerExtdDecoder, AutoencoderKLInnerExtdEncoder)
from lddbm.utils.names import BridgeModelsTyps, Decoders, Encoders


def create_encoder(encoder_type: Encoders, model_args):
    if encoder_type == Encoders.Conv3DEncoder.value:
        return Conv3DEncoder(
            diff_img_size=model_args.latent_image_size,
            diff_in_channels=model_args.in_channels,
            num_views=model_args.num_of_views,
            num_channels=model_args.num_channels_x,
            dropout=model_args.dropout,
        )

    elif encoder_type == Encoders.MV2DEncoder.value:
        return MV2DEncoder(
            diff_img_size=model_args.latent_image_size,
            diff_in_channels=model_args.in_channels,
            dropout=model_args.dropout,
        )

    elif encoder_type in [
        Encoders.KlVaePreTrainedEncoder16.value,
        Encoders.KlVaePreTrainedEncoder128.value,
    ]:
        base_model_config = OmegaConf.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        )
        return AutoencoderKLInnerExtdEncoder(
            base_model_config.model.params.ddconfig,
            base_model_config.model.params.embed_dim,
            model_type=encoder_type,
        )

    elif encoder_type == Encoders.Identity.value:
        return EncIdentity()

    else:
        raise NotImplementedError(f"Encoder type {encoder_type} not implemented")


def create_bridge(model_args):
    if model_args.denoiser_type == BridgeModelsTyps.BridgeTransformer.value:
        denoiser = AutoRegressiveTransformer(in_channels=model_args.in_channels)
    else:
        raise NotImplementedError(
            f"Bridge Model {model_args.denoiser_type} not implemented"
        )

    diffusion = KarrasDenoiser(
        sigma_data=model_args.sigma_data,
        sigma_max=model_args.sigma_max,
        sigma_min=model_args.sigma_min,
        beta_d=model_args.beta_d,
        beta_min=model_args.beta_min,
        cov_xy=model_args.cov_xy,
        weight_schedule=model_args.weight_schedule,
        pred_mode=model_args.pred_mode,
    )

    schedule_sampler = create_named_schedule_sampler(
        model_args.schedule_sampler, diffusion
    )

    return BridgeModel(denoiser, diffusion, schedule_sampler)


def create_decoder(decoder_type: Decoders, model_args):
    if decoder_type == Decoders.Conv3DDecoder.value:
        return Conv3DDecoder(num_channels=model_args.num_channels_x)

    elif decoder_type == Decoders.NoDecoder.value:
        return None

    elif decoder_type == Decoders.KlVaePreTrainedDecoder128.value:
        base_model_config = OmegaConf.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        )
        return AutoencoderKLInnerExtdDecoder(
            base_model_config.model.params.ddconfig,
            base_model_config.model.params.embed_dim,
        )

    elif decoder_type == Decoders.Identity.value:
        return DecIdentity()

    else:
        raise NotImplementedError(f"Decoder type {decoder_type} not implemented")


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "real-uniform":
        return RealUniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif name == "lognormal":
        return LogNormalSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
