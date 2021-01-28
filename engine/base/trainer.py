from __future__ import print_function, absolute_import

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils import Bar
from utils.meters import AverageMeter
from modeling.metrics.classification import accuracy


class Trainer(object):
    def __init__(self, model, optimizer, distributed=False):
        super(Trainer, self).__init__()
        self.model       = model
        self.optimizer   = optimizer
        self.distributed = distributed

    def __call__(self, data_loader, epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_avg = AverageMeter()
        top1_avg = AverageMeter()
        top5_avg = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader)) if not self.distributed or dist.get_rank() == 0 else None

        for i, (data, target) in enumerate(data_loader):
            data_time.update(time.time() - end)

            data, target = data.cuda(), target.cuda()
            logits, _ = self.model(data)
            loss = F.cross_entropy(logits, target)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            with torch.no_grad():
                prec1, prec5 = accuracy(logits, target, topk=(1, 5))
                if self.distributed:
                    temp = torch.stack([loss, prec1, prec5])
                    dist.barrier()
                    dist.reduce(temp, dst=0)
                    if dist.get_rank() == 0:
                        temp /= dist.get_world_size()
                    loss, prec1, prec5 = temp
                loss_avg.update(loss,  target.size(0))
                top1_avg.update(prec1, target.size(0))
                top5_avg.update(prec5, target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bta:.3f} | " \
                             "Loss {N_lossa:.3f} | Prec1 {N_preca1:.2f} | Prec5 {N_preca5:.2f}".format(
                    N_epoch=epoch, N_batch=i+1, N_size=len(data_loader), N_bta=batch_time.avg,
                    N_lossa=loss_avg.avg, N_preca1=top1_avg.avg, N_preca5=top5_avg.avg
                )
                bar.next()
        if bar is not None:
            bar.finish()
        return
