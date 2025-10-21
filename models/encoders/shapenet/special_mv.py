import torch.nn as nn


class ViewEncoder3D(nn.Module):
    """ 3D CNN Encoder to extract volumetric features """

    def __init__(self, feature_dim=512):
        super(ViewEncoder3D, self).__init__()
        self.in_views = nn.Conv3d(4, 32, kernel_size=3, stride=1, padding=1)
        self.in_channels = nn.Conv3d(3, 224, kernel_size=3, stride=1, padding=1)

        self.encoder = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 128, 56, 56, 56)
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, 56, 56, 56)
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 128, 56, 56, 56)
            nn.ReLU(),
            nn.Conv3d(256, feature_dim, kernel_size=3, stride=2, padding=1),  # (B, 256, 28, 28, 28)
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.in_views(x)
        x = self.in_channels(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return self.encoder(x)  # (B, 512, 14, 14, 14)


class ShapeDecoder3D(nn.Module):
    """ 3D CNN Decoder to generate 3D voxel grid """

    def __init__(self, feature_dim=512, output_shape=(32, 32, 32)):
        super(ShapeDecoder3D, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_dim, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 28, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 56, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 112, 112, 112)
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),  # (B, 1, 224, 224, 224)
            nn.Sigmoid(),
        )
        self.final_conv = nn.Conv3d(1, 1, kernel_size=3, stride=7, padding=1)  # Downsample to (B, 1, 32, 32, 32)

    def forward(self, x):
        x = self.decoder(x)  # (B, 1, 224, 224, 224)
        x = self.final_conv(x)  # (B, 1, 32, 32, 32)
        return x


class MultiViewTo3D(nn.Module):
    """ Full pipeline: 3D CNN Encoder -> 3D Shape Reconstruction """

    def __init__(self, feature_dim=512, output_shape=(32, 32, 32)):
        super(MultiViewTo3D, self).__init__()
        self.encoder = ViewEncoder3D(feature_dim)
        self.decoder = ShapeDecoder3D(feature_dim, output_shape)

    def forward(self, x):
        features = self.encoder(x)  # (B, 512, 14, 14, 14)
        shape_3d = self.decoder(features)  # (B, 1, 32, 32, 32)
        return shape_3d
