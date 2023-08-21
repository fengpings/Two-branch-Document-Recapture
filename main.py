#  Author: fengping su
#  date: 2023-8-21
#  All rights reserved.

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from utils import logger
from config import opt
from data.dataset import RecaptureDataset
from models.models import TBNet
import models.models as models


def train(**kwargs):
    setup_seed(42)
    opt.parse(kwargs)
    if not os.path.exists('./output'):
        os.mkdir('./output')

    run_out_path = os.path.join('./output', opt.run_name)
    if not os.path.exists(run_out_path):
        os.mkdir(run_out_path)
    # saving run config
    with open(os.path.join(run_out_path, 'config.txt'), 'w') as f:
        for k, v in opt.__class__.__dict__.items():
            if not k.startwith('_'):
                f.write(f'{k}:{v}\n')
    # data preparation
    train_dataset = RecaptureDataset(root=opt.train_root, training=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              prefetch_factor=opt.prefetch_factor,
                              pin_memory=opt.pin_mem)
    val_dataset = RecaptureDataset(root=opt.val_root, training=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=opt.val_batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            prefetch_factor=opt.prefetch_factor,
                            pin_memory=opt.pin_mem)
    # model
    model = getattr(models, opt.model)()
    # optimizer
    op = opt.op(params=model.parameters(), lr=opt.lr)
    op.pa
    # scheduler
    scheduler = opt.scheduler(optimizer=op)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
