import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from random import randint
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.transforms.functional import to_pil_image
from models import PSPNet, Backbone
from datamodules import LightningCityscapes


class LightningPSPNet(pl.LightningModule):
    def __init__(self,
                 n_classes,
                 psp_size=2048,
                 psp_bins=(1, 2, 3, 6),
                 dropout=0.1,
                 backbone='resnet50',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pspnet = PSPNet(n_classes=n_classes,
                             psp_size=psp_size,
                             psp_bins=psp_bins,
                             dropout=dropout,
                             backbone=Backbone(backbone, pretrained=True))
        self.ckpts_index = 0

    def forward(self, x):
        return torch.argmax(self.pspnet(x), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [40, 60, 80, 90],
            gamma=0.2,
        )
        return [optimizer], [scheduler]

    def criterion(self, output, target):
        target = target.squeeze(dim=1)
        return F.nll_loss(output, target, ignore_index=0)

    def accuracy(self, output, target):
        output = torch.argmax(output, dim=1)
        target = target.squeeze(dim=1)
        acc = torch.sum(output == target)
        tot = target.size(0) * target.size(1) * target.size(2)
        return acc / tot

    def create_image(self, x, y, y_hat):
        y_hat = torch.argmax(y_hat, dim=0)
        image = to_pil_image(x)
        mask_data = y_hat.squeeze().cpu().detach().numpy()
        gt_mask_data = y.squeeze().cpu().detach().numpy()
        class_labels = LightningCityscapes.class_labels
        return wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": mask_data,
                    "class_labels": class_labels
                },
                "groud_truth": {
                    "mask_data": gt_mask_data,
                    "class_labels": class_labels
                },
            },
        )

    def training_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.pspnet(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('train_loss',
                 loss,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)
        self.log('train_acc',
                 accuracy,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.pspnet(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('val_acc',
                 accuracy,
                 prog_bar=False,
                 on_step=True,
                 on_epoch=True)
        idx = randint(0, x.size(0) - 1)
        wandb.log(
            {'val_image': [self.create_image(x[idx], y[idx], y_hat[idx])]})
        return loss

    def test_step(self, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.pspnet(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('test_loss',
                 loss,
                 prog_bar=False,
                 on_step=True,
                 on_epoch=True)
        self.log('test_acc',
                 accuracy,
                 prog_bar=False,
                 on_step=True,
                 on_epoch=True)
        return {'loss': loss, 'acc': accuracy}


if __name__ == '__main__':
    name = 'resnet34-psp512-fine-dev2'

    cityscapes = LightningCityscapes(
        root='C:\\Users\\yousiki\\Documents\\Cityscapes',
        mode='fine',
        size=(512, 1024),
        batch_size=4,
        num_workers=0,
    )

    cityscapes.setup()

    pspnet = LightningPSPNet(
        n_classes=cityscapes.n_classes,
        psp_size=512,
        psp_bins=(1, 2, 3, 6),
        backbone='resnet34',
        lr=0.1,
    )

    pspnet = LightningPSPNet.load_from_checkpoint(
        os.path.join(name + '_ckpt', 'last.ckpt'))

    wandb_logger = WandbLogger(
        name=name,
        project='PL-PSPNet',
        entity='yousiki',
    )

    wandb_logger.watch(pspnet.pspnet, log='all', log_freq=100)

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), name + '_ckpt'),
        filename='{epoch}-{step}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        verbose=True,
        save_last=True,
        save_top_k=10,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        benchmark=True,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[model_checkpoint],
        accumulate_grad_batches=4,
        limit_val_batches=0.1,
        # fast_dev_run=True,
    )

    trainer.fit(pspnet, datamodule=cityscapes)