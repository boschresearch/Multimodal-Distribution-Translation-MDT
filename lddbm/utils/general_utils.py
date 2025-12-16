# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

from datetime import datetime
from pathlib import Path
import blobfile as bf
import torch


def get_gpu(gpu_id):
    return torch.device(f"cuda:{gpu_id}")


def get_workdir(output_base_path, exp):
    workdir = f'{output_base_path}/workdir/{exp}/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    return workdir


def create_workdir(args):
    workdir = get_workdir(args.output_base_path, args.config_name + "_" + args.exp_name)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    with bf.BlobFile(bf.join(workdir, "args"), "wb") as f:
        torch.save(args, f)

    print(f"workdir: {workdir}")
    return workdir
