# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

""" From 5 views of the object to 3D reconstruction of it"""
from utils.names import Encoders, Decoders, ReconstructionLoss, TrainingStrategy, BridgeModelsTyps, Datasets, \
    DistanceMetric


def load_arguments(parser) -> None:
    parser.add_argument('--exp_name', type=str, default="shapenet")
    parser.add_argument('--task', type=str, default="multi2shape_4v")
    parser.add_argument('--output_base_path', type=str, default="./")

    # --------- shared configurations --------- #
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--test_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=50000)

    # --------- data and task ----------------- #
    # data and general
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=Datasets.ShapeNet.value)
    parser.add_argument('--num_of_views', type=int, default=4)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--total_training_steps', type=int, default=1000000)

    # --------- multi modal bridge  ----------------- #
    parser.add_argument('--training_strategy', type=str, default=TrainingStrategy.IterativeTraining.value)
    parser.add_argument('--reconstruction_loss_type', type=str, default=ReconstructionLoss.Predictive.value)
    parser.add_argument('--distance_measure_loss', type=str, default=DistanceMetric.MSE.value)
    parser.add_argument('--clip_loss_w', type=float, default=1, help='the weight of the clip loss, if 0 no loss')

    # optimization
    parser.add_argument('--optimizer', type=str, default="RAdam")
    parser.add_argument('--use_scheduler', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ema_rate', type=str, default="0.9999")

    # --------- autoencoders ----------------- #
    # add path to load trained model. The default start a new mode: 'tmp_x'
    parser.add_argument('--encoder_x_path', type=str, default="tmp_encoder_x")
    parser.add_argument('--encoder_x_type', type=str, default=Encoders.Conv3DEncoder.value)
    parser.add_argument('--decoder_x_path', type=str, default="tmp_decoder_x")
    parser.add_argument('--decoder_x_type', type=str, default=Decoders.Conv3DDecoder.value)
    # model
    parser.add_argument('--num_channels_x', type=int, default=1)

    # add path to load trained model. The default start a new mode: 'tmp_y'
    parser.add_argument('--encoder_y_path', type=str, default="tmp_encoder_y")
    parser.add_argument('--encoder_y_type', type=str, default=Encoders.MV2DEncoder.value)
    parser.add_argument('--decoder_y_path', type=str, default="tmp_decoder_y")
    parser.add_argument('--decoder_y_type', type=str, default=Decoders.NoDecoder.value)

    # --------- diffusion bridge ------------- #
    # add path to load trained model. Default: 'tmp_db'
    parser.add_argument('--bridge_path', type=str, default="tmp_bridge")
    parser.add_argument('--w_denoise', type=float, default=1)  # weight on the denoiser (DDBM) loss
    parser.add_argument('--lr_anneal_steps', type=int, default=0)

    # diffusion - VE
    parser.add_argument('--pred_mode', type=str, default="ve")
    parser.add_argument('--sigma_data', type=float, default=0.5)
    parser.add_argument('--cov_xy', type=float, default=0)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--schedule_sampler', type=str, default="real-uniform")
    parser.add_argument('--beta_d', type=float, default=2)
    parser.add_argument('--sigma_max', type=float, default=80.0)
    parser.add_argument('--sigma_min', type=float, default=0.002)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_schedule', type=str, default='karras')

    # sampling
    parser.add_argument('--sampling_steps', type=int, default=40)
    parser.add_argument('--rho', type=float, default=7.0)
    parser.add_argument('--sampler', type=str, default='heun')
    parser.add_argument('--churn_step_ratio', type=float, default=0)
    parser.add_argument('--guidance', type=float, default=0.5)

    # transformer architecture
    parser.add_argument('--denoiser_type', type=str, default=BridgeModelsTyps.BridgeTransformer.value)
    parser.add_argument('--latent_image_size', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=256)

    # conditioning
    parser.add_argument('--use_scale_shift_norm', type=int, default=1)
