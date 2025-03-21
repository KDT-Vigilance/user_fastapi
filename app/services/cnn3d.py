import torch
import torch.nn as nn

class Deeper3DCNN_Gray(nn.Module):
    def __init__(self):
        super(Deeper3DCNN_Gray, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
