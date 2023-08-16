#   Author: fengping su
#   date: 2023-8-14
#   All rights reserved.
#

from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from config import opt
import cv2


class RecaptureDataset(ImageFolder):
    def __init__(self, root: str, training: bool = True):
        super(RecaptureDataset, self).__init__(root=root)

        self.training = training
        if self.trainig:
            self.transform = self._training_transform()
        else:
            self.transform = self._evaluation_transform()

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        # load using opencv to perform DCT
        # dct_sample = cv2.imread(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)



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
