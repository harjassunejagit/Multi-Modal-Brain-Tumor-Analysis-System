import torch
import torch.nn as nn


class TumorClassifier(nn.Module):

    def __init__(self, num_classes=3):
        super(TumorClassifier, self).__init__()

        self.net = nn.Sequential(

            # Input: 3 channels (T1, T2, FLAIR)
            nn.Conv3d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),

            nn.Flatten(),

            nn.Linear(64, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)