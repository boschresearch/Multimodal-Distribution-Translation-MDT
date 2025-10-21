# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch

import torch
from scipy.spatial import distance_matrix


def chamfer_distance(pc1, pc2):
    """ Compute Chamfer Distance between two point clouds. """
    dist1 = torch.cdist(pc1, pc2)
    diagonal_idx = torch.arange(dist1.shape[-1])
    dist1[:, diagonal_idx, diagonal_idx] = float('inf')
    dist1 = dist1.min(dim=1)[0]

    dist2 = torch.cdist(pc2, pc1)
    diagonal_idx = torch.arange(dist2.shape[-1])
    dist2[:, diagonal_idx, diagonal_idx] = float('inf')
    dist2 = dist2.min(dim=1)[0]

    return dist1.mean(dim=1) + dist2.mean(dim=1)


def earth_mover_distance(pc1, pc2):
    """ Compute Earth Mover's Distance (EMD) between two point clouds. """
    pc1_np = pc1.cpu().numpy()
    pc2_np = pc2.cpu().numpy()
    dist_matrix = distance_matrix(pc1_np[0], pc2_np[0])
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return dist_matrix[row_ind, col_ind].mean()


def compute_1_nna(real_pcs, generated_pcs, bs=1024):
    """
    Computes 1-Nearest Neighbor Accuracy (1-NNA) for point cloud generation evaluation.

    Args:
        real_pcs (torch.Tensor): Real point clouds of shape (N, P, 3).
        generated_pcs (torch.Tensor): Generated point clouds of shape (M, P, 3).

    Returns:
        dict: A dictionary containing 1-NNA accuracy for both Chamfer Distance and EMD.
    """
    num_real = len(real_pcs)
    num_generated = len(generated_pcs)
    labels = torch.cat([torch.ones(num_real), torch.zeros(num_generated)])
    all_pcs = torch.cat([real_pcs, generated_pcs])

    results = {}

    for metric_name, metric_func in zip(["CD"], [chamfer_distance, earth_mover_distance]):
        # Compute pairwise distances
        dist_matrix = torch.zeros((len(all_pcs), len(all_pcs)))
        pairs_i = []
        pairs_j = []
        indices = []
        for i in range(len(all_pcs)):
            for j in range(len(all_pcs)):
                if i != j:
                    pairs_i.append(all_pcs[i].float())
                    pairs_j.append(all_pcs[j].float())
                    indices.append((i, j))

                if len(pairs_i) == 1024:
                    res = metric_func(torch.vstack(pairs_i), torch.vstack(pairs_j))
                    for (i, j), dist in zip(indices, res):
                        dist_matrix[i, j] = dist
                    pairs_i = []
                    pairs_j = []
                    indices = []

        if len(pairs_i) > 0:
            res = metric_func(torch.vstack(pairs_i), torch.vstack(pairs_j))
            for (i, j), dist in zip(indices, res):
                dist_matrix[i, j] = dist

        # Determine nearest neighbors
        dist_matrix.fill_diagonal_(float('inf'))
        nn_indices = dist_matrix.argmin(dim=1)
        nn_labels = labels[nn_indices]

        # Calculate 1-NNA accuracy
        correct = (labels == nn_labels).sum()
        accuracy = correct / len(labels)

        results[f"1-NNA-{metric_name}"] = accuracy

    return results


def iou_3d(prediction, target, threshold=0.5):
    """
    Compute the IoU between predicted and ground-truth voxel grids.

    Args:
        prediction (torch.Tensor): The predicted voxel grid (3D tensor of shape (X, Y, Z)).
        target (torch.Tensor): The ground truth voxel grid (3D tensor of shape (X, Y, Z)).
        threshold (float): Threshold to binarize predictions (default: 0.5).

    Returns:
        float: The computed IoU score.
    """
    # Ensure that both prediction and target have the same shape
    if prediction.shape != target.shape:
        print(
            f"warning: the prediction size {prediction.shape} "
            f"is not equal to the target size {target.shape} in the iou_3d function")
        return torch.tensor(0)

    # Binarize the prediction using the threshold
    prediction = (prediction >= threshold).float()
    target = (target >= threshold).float()  # Ground truth is typically binary, so threshold 0.5

    # Compute intersection (logical AND)
    intersection = torch.sum(prediction * target)

    # Compute union (logical OR)
    union = torch.sum((prediction + target) > 0)

    # Compute IoU
    iou = intersection / union

    return iou


def f_score(pred_points, gt_points, threshold_d=0.001):
    """
    Compute F-score between predicted and ground-truth 3D points.

    Args:
        pred_points (torch.Tensor): Predicted 3D points of shape (N, 3), where N is the number of points.
        gt_points (torch.Tensor): Ground-truth 3D points of shape (M, 3), where M is the number of points.
        threshold_d (float): Distance threshold to consider points as correctly predicted.

    Returns:
        f1 (float): The F1-score.
        precision (float): The precision.
        recall (float): The recall.
    """

    # Calculate pairwise distances between predicted and ground-truth points
    dist_matrix = torch.cdist(pred_points, gt_points)  # Shape (N, M)

    # Precision: Fraction of predicted points within distance threshold from ground-truth points
    min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=1)  # Min distance for each predicted point to any gt point
    precision = (min_dist_pred_to_gt <= threshold_d).float().mean().item()

    # Recall: Fraction of ground-truth points within distance threshold from predicted points
    min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=0)  # Min distance for each gt point to any predicted point
    recall = (min_dist_gt_to_pred <= threshold_d).float().mean().item()

    # F1 score (harmonic mean of precision and recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return torch.tensor(f1)
