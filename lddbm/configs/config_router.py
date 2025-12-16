# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import argparse


def get_configs():
    """
    Load the configs, override default configs with argument parsed configs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config_name', type=str, required=True)

    if 'multi2shape' == parser.parse_known_args()[0].config_name:
        from configs.shapenet.our_method import load_arguments
        load_arguments(parser)
    elif 'sr' == parser.parse_known_args()[0].config_name:
        from configs.super_resolution.our_method import load_arguments
        load_arguments(parser)
    else:
        raise ModuleNotFoundError(f"No such config file: {parser.parse_known_args()[0].config_name}")

    args = parser.parse_args()
    print(args)

    return args
