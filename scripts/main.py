# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

"""
- For more info about the training setup got to the file names.py to MMLossTypes object
- For more info about architectures ...
"""
import torch

from lddbm.models.mm_dist_trans import ModalityTranslationBridge
from lddbm.utils.models_loader import create_encoder, create_bridge, create_decoder
from lddbm.utils.weights_loading import load_weights
from lddbm.utils import logger
from lddbm.utils.general_utils import create_workdir
from lddbm.configs.config_router import get_configs
from lddbm.datasets import load_data
from lddbm.trainer.trainer import TrainLoop


def main(args):
    # --- create working directory --- #
    workdir = create_workdir(args)

    # --- set the gpu configurations --- #
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{torch.cuda.current_device()}"

    # --- set logger --- #
    logger.configure(dir=workdir)

    # --- create models --- #
    logger.log(
        "creating encoder_x model..."
    )  # create the main multimodal diffusion bridge framework
    encoder_x = create_encoder(args.encoder_x_type, args)
    logger.log(
        "creating encoder_y model..."
    )  # create the main multimodal diffusion bridge framework
    encoder_y = create_encoder(args.encoder_y_type, args)
    logger.log(
        "creating decoder_x model..."
    )  # create the main multimodal diffusion bridge framework
    decoder_x = create_decoder(args.decoder_x_type, args)
    logger.log(
        "creating decoder_y model..."
    )  # create the main multimodal diffusion bridge framework
    decoder_y = create_decoder(args.decoder_y_type, args)
    logger.log(
        "creating bridge model..."
    )  # create the main multimodal diffusion bridge framework
    bridge = create_bridge(args)

    mtb = ModalityTranslationBridge(
        bridge_model=bridge,
        encoder_x=encoder_x,
        encoder_y=encoder_y,
        decoder_x=decoder_x,
        decoder_y=decoder_y,
        rec_loss_type=args.reconstruction_loss_type,
        clip_loss_w=args.clip_loss_w,
        training_strategy=args.training_strategy,
        distance_measure_loss=args.distance_measure_loss,
    )
    # move to gpu
    mtb.to(device)

    # --- load pre-trained models if needed --- #
    load_weights(mtb, args, logger)

    # --- load data --- #
    logger.log("creating data loader...")
    train_data, test_data = load_data(
        arguments=args,
        dataset=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.log("training...")
    TrainLoop(
        mtb=mtb,
        train_data=train_data,
        test_data=test_data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        lr_anneal_steps=args.lr_anneal_steps,
        total_training_steps=args.total_training_steps,
        workdir=workdir,
        dataset=args.dataset,
        use_scheduler=args.use_scheduler,
        device=device,
    ).run_loop()


if __name__ == "__main__":
    arguments = get_configs()
    main(arguments)
