# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels


class PSPNet(nn.Module):
    def __init__(self,
                 n_classes,
                 psp_size=2048,
                 psp_bins=(1, 2, 3, 6),
                 dropout=0.1,
                 backbone=None):
        super().__init__()
        self.backbone = backbone
        self.ppmodule = PPModule(
            self.backbone.out_channels,
            psp_size,
            psp_bins,
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(self.ppmodule.out_channels,
                      512,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, n_classes, 1),
        )

    def forward(self, x):
        size = (x.size(2), x.size(3))
        x = self.backbone(x)
        x = self.ppmodule(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = F.log_softmax(x)
        return x


class PPModule(nn.Module):
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.out_channels = in_channels + out_channels
        assert out_channels % len(bins) == 0
        out_channels = out_channels // len(bins)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(s, s)),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for s in bins
        ])

    def forward(self, x):
        size = (x.size(2), x.size(3))
        features = [layer(x) for layer in self.layers]
        features = map(
            lambda i: F.interpolate(
                i,
                size=size,
                mode='bilinear',
                align_corners=True,
            ), features)
        features = list(features) + [x]
        features = torch.cat(features, dim=1)
        return features


class Backbone(nn.Module):
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        model = getattr(tvmodels, name)(*args, **kwargs)
        if isinstance(model, tvmodels.ResNet):
            self.model = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
        else:
            raise NotImplementedError
        self.out_channels = out_channels(self.model)

    def forward(self, x):
        return self.model(x)


def out_channels(model):
    model.eval()
    with torch.no_grad():
        x = torch.zeros((1, 3, 64, 64))
        y = model.forward(x)
    return y.size(1)
