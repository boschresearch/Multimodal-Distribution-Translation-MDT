# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch
from torch import nn


class Conv3DEncoder(torch.nn.Module):
    def __init__(self, diff_img_size, diff_in_channels, num_views, num_channels, dropout=0):
        super(Conv3DEncoder, self).__init__()

        self.num_views = num_views
        self.diff_img_size = diff_img_size
        self.diff_in_channels = diff_in_channels
        self.dropout = dropout

        # Encoder using 3D convolutions
        self.encoder = torch.nn.Sequential(
            # (C, V, H, W) -> (32, V/2, H/2, W/2)
            torch.nn.Conv3d(num_channels, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.Dropout(p=dropout),

            # (C, V, H, W) -> (32, V/4, H/4, W/4)
            torch.nn.SiLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),

            # (C, V, H, W) -> (32, V/8, H/8, W/8)
            torch.nn.SiLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        B, C, H, W, D = x.size()
        z = self.encoder(x)
        z = z.reshape(B, self.diff_in_channels, self.diff_img_size, self.diff_img_size)

        return z


class MV2DEncoder(nn.Module):
    def __init__(self, diff_img_size, diff_in_channels, in_channels=3, dropout=0):
        super(MV2DEncoder, self).__init__()

        self.diff_img_size = diff_img_size
        self.diff_in_channels = diff_in_channels
        self.dropout = dropout

        # Shared Encoder for each view

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # (C, H, W) -> (64, H/2, W/2)
            nn.SiLU(True),

            torch.nn.SiLU(True),
            torch.nn.Dropout(p=dropout),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (128, H/4, W/4)

            torch.nn.SiLU(True),
            torch.nn.Dropout(p=dropout),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (256, H/8, W/8)

            torch.nn.SiLU(True),
            torch.nn.Dropout(p=dropout),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # -> (256, H/16, W/16)

            torch.nn.SiLU(True),
            torch.nn.Dropout(p=dropout),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=2),  # -> (256, H/32, W/32)

            torch.nn.SiLU(True),
            torch.nn.Dropout(p=dropout),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, H/64, W/64)

        )

        # Latent space projection for each view
        self.fc_encode = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, 256 * 8 * 8),  # final channel size x final H and W x number of views
        )

    def forward(self, x):
        B, V, C, H, W = x.size()

        z = self.encoder(x.reshape(-1, C, H, W))
        z = self.fc_encode(z.reshape(B, -1))

        return z.reshape(B, self.diff_in_channels, self.diff_img_size, self.diff_img_size)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
