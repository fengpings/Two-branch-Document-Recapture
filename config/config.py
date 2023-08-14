#   Author: fengping su
#   date: 2023-8-14
#   All rights reserved.
#
import torch
from utils import logger
class DefaultConfig():
    model = "" #model name
    train_root = ""
    val_root = ""
    test_root = ""

    train_model_path = "" # path to checkpoint model for training
    test_model_path = "" # path to model for testing

    train_batch_size = 64
    test_batch_size = 64
    scheduler = torch.optim.lr_scheduler.LRScheduler()
    num_workers = 8
    max_epoch = 20
    lr = 0.1

    # img transforms
    img_size = (224, 224)
    h_flip_p = 0.5
    v_flip_p = 0.5
    data_mean = [0.6235, 0.6006, 0.5880]
    data_std = [0.2236, 0.2346, 0.2490]

    device = None

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                logger.warning(f"{self.__class__} does not have attribute {k}")
            setattr(self, k, v)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info("User Configuration: ")
        print('-'*50)
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
        print('-'*50)