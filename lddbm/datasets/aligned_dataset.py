import json
import os.path
from typing import Callable
import torch
import numpy as np
from lddbm.datasets.shapenet.binvox_rw import read_as_3d_array
from PIL import Image


class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, images_paths, lr_transforms, hr_transforms, train=True):
        self.images_paths = images_paths
        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms
        self.train = train

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, ix):
        path = self.images_paths[ix]
        image = Image.open(path)
        lr_image = self.lr_transforms(image)
        hr_image = self.hr_transforms(image)

        return hr_image, lr_image, ix


class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annot_path: str,
        model_path: str,
        image_path: str,
        image_transforms: Callable,
        split: str = "train",
        mode: str = "random",
        background=(255, 255, 255),
        view_num: int = 1,
        data_direction: str = "shapenet_3d_to_2d",
    ):
        """
        @param annot_path: path to the "ShapeNet.json" file
        @param model_path: path to the "ShapeNetVox32" folder
        @param image_path: path to the "ShapeNetRendering" folder
        @param image_transforms: preprocessing transformations for images
        @param split: one of "train", "val", "test"
        @param mode:
            - random: load a random image if there are multiple
            - first: always load the first images
        @param view_num: load view_num of images at once
            - >= 1: image size: view * c * h * w
            - -1: image size: c * h * w

        Stripped down and slightly modified version of ShapeNetDataset from
        https://github.com/sunethma/P3D-FusionNet-backend
        """

        if split not in ["train", "val", "test"]:
            raise ValueError("Unexpected split")

        if mode not in ["random", "first"]:
            raise ValueError("Unexpected mode")

        with open(annot_path) as annot_file:
            annots = json.load(annot_file)

        self._meta_data = []
        for taxonomy in annots:
            for model_id in taxonomy[split]:
                self._meta_data.append(
                    {
                        "taxonomy_id": taxonomy["taxonomy_id"],
                        "taxonomy_name": taxonomy["taxonomy_name"],
                        "model_id": model_id,
                    }
                )

        self._model_path = model_path
        self._image_path = image_path
        self._image_transforms = image_transforms
        self._mode = mode
        self._background = background
        self._view_num = view_num
        self._data_direction = data_direction

    def __getitem__(self, index):
        meta_data = self._meta_data[index]
        taxonomy_id = meta_data["taxonomy_id"]
        model_id = meta_data["model_id"]

        binvox_path = os.path.join(
            self._model_path, taxonomy_id, model_id, "model.binvox"
        )

        with open(binvox_path, "rb") as f:
            raw_voxel = read_as_3d_array(f)
            voxel = raw_voxel.data.astype(np.float32)

        image_base_path = os.path.join(
            self._image_path, taxonomy_id, model_id, "rendering"
        )
        image_file_list = list(os.listdir(image_base_path))
        image_file_list.sort()
        image_file_list.remove("rendering_metadata.txt")
        image_file_list.remove("renderings.txt")

        if self._mode == "random":
            image_indices = torch.randperm(len(image_file_list))
        else:
            image_indices = torch.arange(len(image_file_list))

        image_indices = image_indices[: self._view_num]

        images = []
        for image_index in image_indices:
            image_path = os.path.join(
                image_base_path, image_file_list[image_index.item()]
            )
            rgba = Image.open(image_path)
            image = Image.new("RGB", rgba.size, self._background)
            image.paste(rgba, mask=rgba.split()[3])
            image = self._image_transforms(image)
            images.append(image)

        images = torch.stack(images)
        if self._view_num == 1:
            images = images.squeeze(0)

        if "3d_to_2d" in self._data_direction:
            return images, torch.Tensor(voxel).unsqueeze(0), meta_data["taxonomy_name"]
        else:
            return torch.Tensor(voxel).unsqueeze(0), images, meta_data["taxonomy_name"]

    def __len__(self):
        return len(self._meta_data)


class nuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_paths, train_ratio=0.8, train=True):
        self.paths = pairs_paths
        train_num = int(len(pairs_paths) * train_ratio)
        if train:
            self.paths = self.paths[:train_num]
        else:
            self.paths = self.paths[train_num:]

    def __len__(self):
        return len(self.paths)

    def sparse_to_dense(self, sparse_occ, occ_shape, t_type="regular"):
        if t_type == "regular":
            gt = (
                torch.zeros([occ_shape[0], occ_shape[2], occ_shape[3], occ_shape[4]])
                .to(sparse_occ.device)
                .type(torch.float)
            )
            for i in range(gt.shape[0]):
                coords = sparse_occ[i][:, :3].type(torch.long)
                gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_occ[i][:, 3]

            return gt

    def __getitem__(self, ix):
        path = self.paths[ix]
        mv, occ = torch.load(path)
        normalized_mv = mv / 255
        occ = self.sparse_to_dense(
            sparse_occ=occ, occ_shape=(1, 0, *(200, 200, 16)), t_type="regular"
        )

        return occ, normalized_mv.squeeze(), ix
