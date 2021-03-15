# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import pytorch_lightning as pl
import argparse
import models


class LightningPSPNet(pl.LightningModule):
    def __init__(self,
                 n_classes,
                 psp_size=2048,
                 psp_bins=(1, 2, 3, 6),
                 dropout=0.1,
                 backbone='resnet50',
                 **kwargs):
        super().__init__()
        self.backbone = models.Backbone(backbone, pretrained=True)
        self.pspnet = models.PSPNet(n_classes=n_classes,
                                    psp_size=psp_size,
                                    psp_bins=psp_bins,
                                    dropout=dropout,
                                    backbone=self.backbone)
        self.save_hyperparameters()

    def forward(self, x):
        return self.pspnet(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        nll_loss = F.nll_loss(y_hat, y)
        return nll_loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        nll_loss = F.nll_loss(y_hat, y)
        return nll_loss

    def setup(self):
        self.train_ds = tv.datasets.Cityscapes(
            root=self.hparams.dataset_root,
            split='train',
            mode='fine',
            target_type='semantic',
            transform=tv.transforms.ToTensor(),
            target_transform=tv.transforms.ToTensor(),
        )
        self.valid_ds = tv.datasets.Cityscapes(
            root=self.hparams.dataset_root,
            split='val',
            mode='fine',
            target_type='semantic',
            transform=tv.transforms.ToTensor(),
            target_transform=tv.transforms.ToTensor(),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )


if __name__ == '__main__':
    module = LightningPSPNet(
        n_classes=35,
        dataset_root='Cityscapes',
        batch_size=16,
        num_workers=8,
        learning_rate=0.1,
    )
    trainer = pl.Trainer(gpus=1)
    trainer.fit(module)
