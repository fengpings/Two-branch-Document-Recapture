#  Author: fengping su
#  date: 2023-8-21
#  All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import swats
import numpy as np
import random
import os
import utils.schedulers as schedulers
from utils import logger
from config import cfg
from data.dataset import RecaptureDataset
import models.models as models
from utils import cal_acc, cal_pn
import argparse


def train(**kwargs):
    setup_seed(42)
    # make dirs
    run_out_path = os.path.join(cfg.output_path, cfg.run_name)
    if not os.path.exists(run_out_path):
        os.mkdir(run_out_path)
    model_out_path = os.path.join(run_out_path, 'models')
    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)
    cfg.parse(kwargs)
    # data preparation
    train_dataset = RecaptureDataset(root=cfg.train_root, training=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.train_batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              prefetch_factor=cfg.prefetch_factor,
                              pin_memory=cfg.pin_mem)
    val_dataset = RecaptureDataset(root=cfg.val_root, training=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg.val_batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            prefetch_factor=cfg.prefetch_factor,
                            pin_memory=cfg.pin_mem)

    # model
    model = getattr(models, cfg.model)()
    model.to(device=cfg.device)

    # optimizer
    opt = None
    if cfg.optimizer != "SWATS":
        opt = getattr(optim, cfg.optimizer)(params=model.parameters(), **getattr(cfg, cfg.optimizer))
    else:
        opt = getattr(swats, cfg.optimizer)(params=model.parameters(), **getattr(cfg, cfg.optimizer))

    # scheduler
    scheduler = None
    if cfg.optimizer == 'SGD':
        if cfg.scheduler == 'CosineAnnealingWarmupLR':
            scheduler = getattr(schedulers, cfg.scheduler)(optimizer=opt, **getattr(cfg, cfg.scheduler))
        else:
            scheduler = getattr(optim.lr_scheduler, cfg.scheduler)(optimizer=opt, **getattr(cfg, cfg.scheduler))

    # loss function
    loss_fn = getattr(nn, cfg.loss_fn)(**getattr(cfg, cfg.loss_fn))
    loss_fn.to(device=cfg.device)

    # summary writer
    log_dir = os.path.join(run_out_path, 'summary')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model=model, input_to_model=[torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)])

    # training
    train_step = 0
    for epoch in range(cfg.max_epoch):
        # train model
        model.train()
        train_accs, train_losses = [], []
        TP, FP, TN, FN = 0, 0, 0, 0

        for batch, (dcts, rgbs, ys) in enumerate(train_loader):
            # data to device
            dcts = dcts.to(device=cfg.device)
            rgbs = rgbs.to(device=cfg.device)
            ys = ys.to(device=cfg.device)

            # forward pass
            opt.zero_grad()
            logits = model(dcts, rgbs)

            # calculate batch loss, accuracy, tp, fp, tn, fn
            loss = loss_fn(logits, ys)
            acc = cal_acc(logits, ys)
            tp, fp, tn, fn = cal_pn(logits, ys)
            TP += tp.item()
            FP += fp.item()
            TN += tn.item()
            FN += fn.item()

            # backward
            loss.backward()
            # keep track of learning rate in iterations
            cur_lr = list(opt.param_groups)[0]['lr']
            writer.add_scalar('lr', cur_lr, train_step)
            # print log every 500 iters
            if (batch + 1) % 500 == 0:
                logger.debug(f'[train epoch:{epoch}/{cfg.max_epoch}] '
                             f'[batch:{batch}] '
                             f'lr:{cur_lr}'
                             f'loss:{loss.item():.3f} '
                             f'accuracy:{acc.item():.3f}')
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            opt.step()
            # if using SGD, update learning rate by scheduler
            if scheduler:
                scheduler.step()
            train_losses.append(loss.item())
            train_accs.append(acc.item())
            train_step += 1

        # print epoch summary
        epoch_acc = np.mean(train_accs)
        epoch_loss = np.mean(train_losses)
        recap_precision = TP/(TP+FP+1e-8)
        recap_recall = TP/(TP+FN+1e-8)
        nonrecap_precision = TN/(TN+FN+1e-8)
        nonrecap_recall = TN/(TN+FP+1e-8)
        logger.debug(f'[train epoch:{epoch}/{cfg.max_epoch} summary:] '
                     f'accuracy:{epoch_acc:.3f}|'
                     f'loss:{epoch_loss:.3f}|'
                     f'recap precision:{recap_precision:.3f}|'
                     f'recap recall:{recap_recall:.3f}|'
                     f'non-recap precision:{nonrecap_precision:.3f}|'
                     f'non-recap recall:{nonrecap_recall:.3f}')
        writer.add_scalars('train_stats', {
            "accuracy": epoch_acc,
            "loss": epoch_loss,
            "recap precision": recap_precision,
            "recap recall": recap_recall,
            "non-recap precision": nonrecap_precision,
            "non-recap recall": nonrecap_recall
        }, global_step=epoch)

        # evaluation
        val_acc, val_loss, recap_precision, recap_recall, nonrecap_precision, nonrecap_recall = val(model, val_loader, loss_fn)

        # print log
        logger.debug(f'[val epoch:{epoch}/{cfg.max_epoch} summary:] '
                     f'accuracy:{val_acc:.3f}|'
                     f'loss:{val_loss:.3f}|'
                     f'recap precision:{recap_precision:.3f}|'
                     f'recap recall:{recap_recall:.3f}|'
                     f'non-recap precision:{nonrecap_precision:.3f}|'
                     f'non-recap recall:{nonrecap_recall:.3f}')
        # add to tensorboard
        writer.add_scalars('val_stats', {
            "accuracy": val_acc,
            "loss": val_loss,
            "recap precision": recap_precision,
            "recap recall": recap_recall,
            "non-recap precision": nonrecap_precision,
            "non-recap recall": nonrecap_recall
        }, global_step=epoch)

        # save model
        if cfg.save_model:
            torch.save({
                "model": model.state_dict(),
                # "opt": opt,
                # "scheduler": scheduler
            }, os.path.join(model_out_path, f'{epoch}_{val_acc:.3f}.pt'))
            logger.info(f'model of epoch {epoch} saved')

    writer.close()


def test():
    model = getattr(models, cfg.model)()
    model.load_state_dict(torch.load(cfg.test_model_path, map_location='cpu')['model'])
    model.to(device=cfg.device)

    test_dataset = RecaptureDataset(root=cfg.test_root, training=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg.val_batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             prefetch_factor=cfg.prefetch_factor,
                             pin_memory=cfg.pin_mem)

    loss_fn = getattr(nn, cfg.loss_fn)(**getattr(cfg, cfg.loss_fn))
    loss_fn.to(device=cfg.device)

    test_acc, test_loss, recap_precision, recap_recall, \
        nonrecap_precision, nonrecap_recall = val(model, test_loader, loss_fn)

    logger.debug(f'test summary:] '
                 f'accuracy:{test_acc:.3f}|'
                 f'loss:{test_loss:.3f}|'
                 f'recap precision:{recap_precision:.3f}|'
                 f'recap recall:{recap_recall:.3f}|'
                 f'non-recap precision:{nonrecap_precision:.3f}|'
                 f'non-recap recall:{nonrecap_recall:.3f}')


def val(model, val_loader, loss_fn):
    model.eval()
    val_accs, val_losses = [], []
    TP, FP, TN, FN = 0, 0, 0, 0

    for _, (dcts, rgbs, ys) in enumerate(val_loader):
        # data to device
        dcts = dcts.to(device=cfg.device)
        rgbs = rgbs.to(device=cfg.device)
        ys = ys.to(device=cfg.device)

        logits = model(dcts, rgbs)
        # calculate batch loss, accuracy, tp, fp, tn, fn
        loss = loss_fn(logits, ys)
        acc = cal_acc(logits, ys)
        tp, fp, tn, fn = cal_pn(logits, ys)
        TP += tp.item()
        FP += fp.item()
        TN += tn.item()
        FN += fn.item()

        val_accs.append(acc.item())
        val_losses.append(loss.item())

    val_acc = np.mean(val_accs)
    val_loss = np.mean(val_losses)
    recap_precision = TP / (TP+FP+1e-8)
    recap_recall = TP / (TP+FN+1e-8)
    nonrecap_precision = TN / (TN+FN+1e-8)
    nonrecap_recall = TN / (TN+FP+1e-8)

    return val_acc, val_loss, recap_precision, recap_recall, nonrecap_precision, nonrecap_recall

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, required=False)
    args = parser.parse_args()
    if args.train:
        train()
    else:
        test()
