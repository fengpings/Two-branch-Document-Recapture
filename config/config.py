#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import os
import torch
from utils import logger


class DefaultConfig:
    # ------------------------------------------------------------------
    # model config
    # ------------------------------------------------------------------
    model = "Res50TBNet"  # model name
    train_root = "/Users/Tristan/Documents/projects/Two-branch-Document-Recapture/data/test/train"
    val_root = "/Users/Tristan/Documents/projects/Two-branch-Document-Recapture/data/test/val"
    test_root = "/Users/Tristan/Documents/projects/Two-branch-Document-Recapture/data/test/test"

    run_name = "run1"

    train_model_path = ""  # path to checkpoint model for training
    test_model_path = ""  # path to model for testing
    output_path = './output'

    # ------------------------------------------------------------------
    # train test dataloader config
    # ------------------------------------------------------------------
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    num_workers = 0
    prefetch_factor = None
    pin_mem = False
    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------
    optimizer = "RAdam"
    Adam = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1.e-8,
        "weight_decay": 1.e-4,
    }

    SGD = {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "dampening": 0,
        "nesterov": False
    }

    RAdam = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1.e-8,
        "weight_decay": 1.e-4,
    }

    SWATS = {
        "lr": 0.1,
        "betas": (0.9, 0.999),
        "eps": 1.e-8,
        "weight_decay": 1.e-4,
        "nesterov": False
    }
    # ------------------------------------------------------------------
    # loss function
    # ------------------------------------------------------------------
    loss_fn = "CrossEntropyLoss"
    CrossEntropyLoss = {
        'weight': None,
        'size_average': None,
        'reduction': 'mean',
        'label_smoothing': 0.1
    }
    # ------------------------------------------------------------------
    # lr scheduler
    # ------------------------------------------------------------------
    scheduler = "CosineAnnealingWarmupLR"
    max_epoch = 160
    CosineAnnealingWarmupLR = {
        "warmup_iters": 10,
        "max_epochs": max_epoch,
        "lr_max": 0.1,
        "lr_min": 1e-5
    }

    OneCycleLR = {
        'max_lr': 0.1,
        'total_steps': 160*2,
        'epochs': None,
        'steps_per_epoch': None,
        'pct_start': 0.3,
        'anneal_strategy': 'cos',
        'cycle_momentum': True,
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 25.,
        'final_div_factor': 1e4,
        'three_phase': False
    }

    # ------------------------------------------------------------------
    # img transforms
    # ------------------------------------------------------------------
    img_size = (224, 224)
    h_flip_p = 0.5
    v_flip_p = 0.5
    data_mean = [0.6235, 0.6006, 0.5880]
    data_std = [0.2236, 0.2346, 0.2490]

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_model = True

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                logger.warning(f"{self.__class__} does not have attribute {k}")
            setattr(self, k, v)
        logger.info("User Configuration: ")
        print('='*50)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        with open(os.path.join(os.path.join(self.output_path, self.run_name), 'config.txt'), 'w') as f:
            for k, v in self.__class__.__dict__.items():
                if not k.startswith('_') and not isinstance(v, dict) and k != 'parse':
                    section = {'model': "Model Configuration",
                               'train_batch_size': "Dataloader Configuration",
                               "optimizer": "Optimizer",
                               "scheduler": "Scheduler",
                               "img_size": "Image Transformation",
                               'device': "Device Configuration"}
                    if k in section:
                        print('-' * 50)
                        f.write(('-'*50) + '\n')
                        print(section[k])
                        f.write(section[k] + '\n')
                        print('-' * 50)
                        f.write(('-' * 50) + '\n')
                    print(k, getattr(self, k))
                    f.write(f'{k} {getattr(self, k)}\n')
                    if k == 'optimizer' or k == 'scheduler' or k == 'loss_fn':
                        print(getattr(self, v))
                        f.write(f'{getattr(self, v)}\n')
        print('='*50)
