

#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from config import cfg
from utils import transforms


class RecaptureDataset(ImageFolder):
    def __init__(self, root: str, training: bool = True):
        super(RecaptureDataset, self).__init__(root=root)

        self.training = training
        if self.training:
            self.transform = self._training_transform()
        else:
            self.transform = self._evaluation_transform()
        self.dct_transform = self._dct_transform()

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        sample_img = self.transform(sample)
        sample_dct = self.dct_transform(sample)
        return sample_img, sample_dct, target

    @staticmethod
    def _training_transform():
        return T.Compose([
            T.Resize(cfg.img_size),
            # T.RandomHorizontalFlip(p=cfg.h_flip_p),
            # T.RandomVerticalFlip(p=cfg.v_flip_p),
            T.ToTensor(),
            # T.Normalize(mean=cfg.data_mean, std=cfg.data_std)
        ])

    @staticmethod
    def _evaluation_transform():
        return T.Compose([
            T.Resize(cfg.img_size),
            T.ToTensor(),
            # T.Normalize(mean=cfg.data_mean, std=cfg.data_std)
        ])

    @staticmethod
    def _dct_transform():
        return T.Compose([
            T.Resize(cfg.img_size),
            transforms.DCT()
        ])
