import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        u = self.up(b)
        d = self.dec1(torch.cat([u, e2], dim=1))

        return torch.sigmoid(self.out(d))