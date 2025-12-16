from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split

from lddbm.datasets.aligned_dataset import CelebaDataset
from lddbm.datasets.shapenet.shapes_3d_transforms import (CenterCrop, Compose,
                                                          RandomBackground,
                                                          ToTensor)
from lddbm.utils.names import Datasets


def load_data(
    arguments,
    dataset,
    batch_size,
    data_path,
    num_workers=4,
):
    if dataset == Datasets.ShapeNet.value:
        from .aligned_dataset import ShapeNetDataset

        def normalize(x):
            return x * 2 - 1

        def to_numpy(image):
            image.convert("RGB")
            return [np.asarray(image, dtype=np.float32) / 255]

        image_trans = Compose(
            [
                to_numpy,
                CenterCrop((224, 224), (128, 128)),
                RandomBackground(((240, 240), (240, 240), (240, 240))),
                ToTensor(),
                lambda x: x[0],
                normalize,
            ]
        )

        dataset_params = {
            "annot_path": f"{data_path}/ShapeNet.json",
            "model_path": f"{data_path}/ShapeNetVox32",
            "image_path": f"{data_path}/ShapeNetRendering",
        }

        trainset = ShapeNetDataset(
            **dataset_params,
            image_transforms=image_trans,
            split="train",
            background=(0, 0, 0),
            view_num=arguments.num_of_views,
            data_direction=dataset,
        )

        valset = ShapeNetDataset(
            **dataset_params,
            image_transforms=image_trans,
            split="test",
            mode="first",
            background=(0, 0, 0),
            view_num=arguments.num_of_views,
            data_direction=dataset,
        )

    elif dataset == Datasets.SR.value:
        image_size = 128
        scale_factor = 8
        hr_transforms = T.Compose(
            [
                T.Resize((image_size, image_size), Image.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                ),  # to scale [-1,1] with tanh activation
            ]
        )

        lr_transforms = T.Compose(
            [
                T.Resize(
                    (image_size // scale_factor, image_size // scale_factor),
                    Image.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                ),  # to scale [-1,1] with tanh activation
            ]
        )

        train_paths = sorted(
            [str(p) for p in glob(f"{data_path}/Flicker50k" + "/*.png")]
        )
        trainset = CelebaDataset(
            train_paths,
            lr_transforms=lr_transforms,
            hr_transforms=hr_transforms,
            train=True,
        )

        image_paths = sorted(
            [str(p) for p in glob(f"{data_path}/celebsA_HQ/celeba_hq_256" + "/*.jpg")]
        )
        _, valid_paths = train_test_split(
            image_paths, test_size=5000, shuffle=True, random_state=42
        )
        valset = CelebaDataset(
            valid_paths,
            lr_transforms=lr_transforms,
            hr_transforms=hr_transforms,
            train=False,
        )

    else:
        raise NotImplementedError(f"no such dataset {dataset}")

    loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )

    return loader, val_loader
