from prefetch_generator import BackgroundGenerator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import numpy as np
import torch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ComposeDataset(Dataset):
    def __init__(self, *args):
        super().__init__()
        self.subdatasets = list(args)

    def __len__(self):
        return sum(map(len, self.subdatasets))

    def __getitem__(self, index):
        for dataset in self.subdatasets:
            if index < len(dataset):
                return dataset[index]
            else:
                index -= len(dataset)
        raise IndexError


class LightningCityscapes(LightningDataModule):
    def __init__(self,
                 root,
                 mode='coarse',
                 size=512,
                 batch_size=32,
                 num_workers=8):
        super().__init__()
        self.root = root
        self.mode = mode
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = datasets.Cityscapes.classes
        self.n_classes = len(self.classes)

    def get_transforms(self, train):
        def func(image, target):
            image = transforms.ToTensor()(image)
            target = np.array(target)
            target = torch.tensor(target)
            target = target.unsqueeze(dim=0)
            concat = torch.cat((image, target), dim=0)
            if train:
                concat = transforms.RandomCrop((self.size, self.size))(concat)
                concat = transforms.RandomVerticalFlip()(concat)
                concat = transforms.RandomHorizontalFlip()(concat)
            else:
                concat = transforms.CenterCrop(self.size)(concat)
            image, target = concat[:3, ...], concat[3:, ...]
            target = target.long()
            return image, target

        return func

    def setup(self, stage=None):
        self.ds_train = datasets.Cityscapes(
            root=self.root,
            split='train',
            mode=self.mode,
            target_type='semantic',
            transforms=self.get_transforms(train=True),
        )
        if self.mode == 'coarse':
            self.ds_train = ComposeDataset(
                self.ds_train,
                datasets.Cityscapes(
                    root=self.root,
                    split='train_extra',
                    mode=self.mode,
                    target_type='semantic',
                    transforms=self.get_transforms(train=True),
                ))
        self.ds_val = datasets.Cityscapes(
            root=self.root,
            split='val',
            mode=self.mode,
            target_type='semantic',
            transforms=self.get_transforms(train=False),
        )
        self.ds_test = datasets.Cityscapes(
            root=self.root,
            split='test',
            mode=self.mode,
            target_type='semantic',
            transforms=self.get_transforms(train=False),
        )

    def train_dataloader(self):
        return DataLoaderX(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoaderX(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoaderX(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )