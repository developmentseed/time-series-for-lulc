import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(ni, nf, ks):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=1, padding=ks // 2)


def _conv_layer(ni, nf, ks, drop=0.0):
    return nn.Sequential(
        _conv(ni, nf, ks), nn.BatchNorm1d(nf), nn.ReLU(), nn.Dropout(p=drop)
    )


def stacked_conv(ni, nf, ks, drop):
    return nn.Sequential(
        _conv_layer(ni, nf, ks, drop=drop),
        nn.MaxPool1d(kernel_size=2, stride=2),
        _conv_layer(nf, nf * 2, ks, drop=0.0),
        nn.AdaptiveAvgPool1d(output_size=1),
        nn.Flatten(),
    )


class Pixel(nn.Module):
    def __init__(self, num_bands, num_classes, hidden_dims=32, drop=0.25):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_bands)
        self.conv1 = stacked_conv(ni=num_bands, nf=hidden_dims, ks=3, drop=drop)
        self.conv2 = stacked_conv(ni=num_bands, nf=hidden_dims, ks=5, drop=drop)
        self.head = nn.Sequential(
            nn.Linear(in_features=2 * (hidden_dims * 2), out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )
        self.drop = drop

    def forward(self, x):
        # normalize the input
        x = self.bn(x)
        print(x.shape)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        print(x1.shape, x2.shape)

        # bs x (2 * nf)
        x3 = torch.cat([x1, x2], dim=1)
        x3 = F.dropout(x3, p=self.drop)
        print(x3.shape)

        logits = self.head(x3)

        return logits
