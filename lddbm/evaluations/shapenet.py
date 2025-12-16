# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch
from matplotlib import pyplot as plt
from lddbm.utils.metrics.shapenet import iou_3d, f_score, compute_1_nna


def voxel_to_point_cloud(voxel_grid, size_pass=1024, subsample_limit=1024):
    """
    Convert a dense 3D voxel grid into a sparse point cloud representation using PyTorch.

    Args:
        voxel_grid (torch.Tensor): A tensor of shape (B, C, H, W, D) representing voxelized 3D shapes.

    Returns:
        point_clouds (list): A list of torch tensors, each containing point cloud coordinates (B, C, X, Y, Z).
    """
    B, C, H, W, D = voxel_grid.shape
    point_clouds = []

    voxel_grid = (voxel_grid >= 0.5).float()

    for b in range(B):
        batch_clouds = []
        for c in range(C):
            indices = torch.nonzero(
                voxel_grid[b, c], as_tuple=False
            )  # Extract non-empty voxel locations

            if indices.shape[0] < size_pass:
                continue  # Discard if fewer than 1024 points

            sampled_indices = indices[
                torch.randperm(indices.shape[0])[:subsample_limit]
            ]  # Uniformly sample 1024 points
            batch_clouds.append(sampled_indices)

        if batch_clouds:
            point_clouds.append(torch.stack(batch_clouds))

    if len(point_clouds) > 0:
        return torch.stack(point_clouds)
    return []


def plot_point_cloud(point_cloud, angles=[(30, 30), (60, 30), (90, 0)]):
    """
    Plot a point cloud from different angles.

    Args:
        point_cloud (numpy.ndarray): An array of shape (N, 3) representing (X, Y, Z) coordinates.
        angles (list): A list of tuples representing different (elev, azim) angles for visualization.
    """
    fig = plt.figure(figsize=(15, 5))

    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, len(angles), i + 1, projection="3d")
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c="b")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View: Elev={elev}, Azim={azim}")

    plt.show()


class ShapeNetEvaluator:

    def __init__(self, device):
        self.device = device
        super().__init__()

    def run_full_eval(self, model, dl, prefix, save_dir, sampling_steps):
        total_IoU = 0
        total_f_score = 0
        total_IoU_y = 0
        total_IoU_x = 0

        all_test_data = None
        all_our_data = None
        if prefix != "train":
            all_test_data = []
            all_our_data = []

        for i, batch in enumerate(dl):
            x, y, _ = batch
            x = x.to(self.device)
            y = y.to(self.device)

            z_x = model.encoder_x(x)
            z_y = model.encoder_y(y)
            if isinstance(z_y, tuple):
                z_y = z_y[0]
            x_rec = model.decoder_x(z_x)
            if z_y.shape == z_x.shape:
                xy_rec = model.decoder_x(z_y)
                IoU_y = iou_3d(xy_rec, x)
            else:
                IoU_y = torch.tensor(0)

            x_hat = model.sample(y, sampling_steps=sampling_steps)

            IoU = iou_3d(x_hat, x)
            IoU_x = iou_3d(x_rec, x)
            f_s = f_score(x_hat, x)

            total_IoU += IoU
            total_IoU_y += IoU_y
            total_f_score += f_s
            total_IoU_x += IoU_x

            if all_test_data is not None:
                res_x = voxel_to_point_cloud(x)
                if len(res_x) > 0:
                    all_test_data.append(res_x)
                res_x_hat = voxel_to_point_cloud(x_hat)
                if len(res_x_hat) > 0:
                    all_our_data.append(res_x_hat)

        knn_results = None
        if all_test_data is not None:
            # torch.save([torch.concatenate(all_our_data, dim=0), torch.concatenate(all_test_data, dim=0)],
            #            save_dir + '/cloud_points.pt')
            gen_data = torch.concatenate(all_our_data, dim=0)
            real_data = torch.concatenate(all_test_data, dim=0)
            knn_results = compute_1_nna(gen_data, real_data)

        full_IoU = total_IoU / (i + 1)
        full_IoU_y = total_IoU_y / (i + 1)
        full_IoU_x = total_IoU_x / (i + 1)
        total_f_score = total_f_score / (i + 1)

        if prefix is None:
            losses = {
                "full_IoU": full_IoU,
                "full_f_score": total_f_score,
                "rec_full_IoU_y": full_IoU_y,
                "rec_full_IoU_x": full_IoU_x,
            }
            if knn_results is not None:
                losses.update({"full-1-NNA-CD": knn_results["1-NNA-CD"]})

        else:
            losses = {
                f"{prefix}_full_IoU": full_IoU,
                f"{prefix}_full_f_score": total_f_score,
                f"{prefix}_rec_full_IoU_y": full_IoU_y,
                f"{prefix}_rec_full_IoU_x": full_IoU_x,
            }

        return losses

    def run_one_batch_eval(self, model, batch, prefix, sampling_steps=40):
        x, y, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)

        z_x = model.encoder_x(x)
        z_y = model.encoder_y(y)
        if isinstance(z_y, tuple):
            z_y = z_y[0]

        x_rec = model.decoder_x(z_x)

        if z_x.shape == z_y.shape:
            xy_rec = model.decoder_x(z_y)
            IoU_y = iou_3d(xy_rec, x)
        else:
            IoU_y = torch.tensor(0)

        if model.decoder_y is not None:
            y_rec = model.decoder_y(z_y)
            y_rec_loss = torch.pow((y - y_rec), 2).mean()
        else:
            y_rec_loss = torch.tensor(0)

        x_hat = model.sample(y, sampling_steps=sampling_steps)

        # measure

        IoU = iou_3d(x_hat, x)
        IoU_x = iou_3d(x_rec, x)
        f_s = f_score(x_hat, x)

        if prefix is None:
            losses = {
                "IoU": IoU,
                "f_score": f_s,
                "rec_IoU_y": IoU_y,
                "rec_IoU_x": IoU_x,
                "y_rec_loss": y_rec_loss,
            }
        else:
            losses = {
                f"{prefix}_IoU": IoU,
                f"{prefix}_f_score": f_s,
                f"{prefix}_rec_IoU_y": IoU_y,
                f"{prefix}_rec_IoU_x": IoU_x,
                f"{prefix}_y_rec_loss": y_rec_loss,
            }

        return losses

    def evaluate(
        self,
        model,
        dataloader,
        eval_type="full",
        prefix=None,
        save_dir="",
        sampling_steps=40,
    ):
        assert eval_type in [
            "full",
            "one_batch",
        ], "Can run evaulation on full dataloader or batch only."
        model.eval()

        if eval_type == "full":
            metrics_dict = self.run_full_eval(
                model, dataloader, prefix, save_dir, sampling_steps
            )

        else:
            batch = next(iter(dataloader))
            metrics_dict = self.run_one_batch_eval(
                model, batch, prefix, sampling_steps=sampling_steps
            )

        return metrics_dict
