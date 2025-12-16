# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import torch


class Conv3DDecoder(torch.nn.Module):

    def __init__(self, num_channels):
        super(Conv3DDecoder, self).__init__()

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.SiLU(True),
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.SiLU(True),
        )
        self.output_layer = torch.nn.ConvTranspose3d(
            64, num_channels, kernel_size=4, stride=2, padding=1
        )
        self.non_linear = torch.nn.Sigmoid()

    def forward(self, z):
        z = self.decoder(z.reshape(z.shape[0], 256, 4, 4, 4))
        decoded = self.non_linear(self.output_layer(z))

        return decoded

    def get_last_layer(self):
        return self.output_layer.weight


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
