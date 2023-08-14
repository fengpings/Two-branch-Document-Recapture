#   Author: fengping su
#   date: 2023-8-14
#   All rights reserved.
#

from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from config import opt


class RecaptureDataset(ImageFolder):
    def __init__(self, root: str, training: bool = True):
        super(RecaptureDataset, self).__init__(root=root)

        self.training = training
        if self.trainig:
            self.transform = self._training_transform()
        else:
            self.transform = self._evaluation_transform()

    @staticmethod
    def _training_transform():
        return T.Compose([
            T.Resize(opt.img_size),
            T.RandomHorizontalFlip(p=opt.h_flip_p),
            T.RandomVerticalFlip(p=opt.v_flip_p),
            T.ToTensor(),
            T.Normalize(mean=opt.data_mean, std=opt.data_std)
        ])

    @staticmethod
    def _evaluation_transform():
        return T.Compose([
            T.Resize(opt.img_size),
            T.ToTensor(),
            T.Normalize(mean=opt.data_mean, std=opt.data_mean)
        ])
