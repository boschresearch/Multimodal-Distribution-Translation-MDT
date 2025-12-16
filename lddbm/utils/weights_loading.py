# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch


def load_weights(mdt, args, logger):
    load_model_weights(mdt, args.bridge_path, logger)
    load_model_weights(mdt, args.encoder_x_path, logger)
    load_model_weights(mdt, args.encoder_y_path, logger)
    load_model_weights(mdt, args.decoder_x_path, logger)
    load_model_weights(mdt, args.decoder_y_path, logger)


def load_model_weights(mdt, model_path, logger, map_location='cpu', strict=True):
    if 'tmp' not in model_path:
        state_dict = torch.load(model_path, map_location=map_location)

        if 'bridge' in model_path:
            mdt.bridge_model.load_state_dict(state_dict, strict=strict)
            logger.log(f"loaded bridge model {model_path}")

        elif 'encoder_x' in model_path:
            mdt.encoder_x.load_state_dict(state_dict, strict=strict)
            logger.log(f"loaded x encoder {model_path}")

        elif 'encoder_y' in model_path:
            mdt.encoder_y.load_state_dict(state_dict, strict=strict)
            logger.log(f"loaded y encoder {model_path}")

        elif 'decoder_x' in model_path:
            mdt.decoder_x.load_state_dict(state_dict, strict=strict)
            logger.log(f"loaded x decoder {model_path}")

        elif 'decoder_y' in model_path:
            mdt.decoder_y.load_state_dict(state_dict, strict=strict)
            logger.log(f"loaded y decoder {model_path}")
