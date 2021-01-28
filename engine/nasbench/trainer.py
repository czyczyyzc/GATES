from __future__ import print_function, absolute_import

import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from modeling.losses import margin_linear
from utils import Bar
from utils.meters import AverageMeter


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 compare_margin=0.1,
                 compare_threshold=0.0,
                 max_compare_ratio=4.0,
                 choose_pair_criterion='random',
                 distributed=False):
        super(Trainer, self).__init__()
        self.model     = model
        self.optimizer = optimizer
        self.compare_margin    = compare_margin
        self.compare_threshold = compare_threshold
        self.max_compare_ratio = max_compare_ratio
        self.choose_pair_criterion = choose_pair_criterion
        self.distributed = distributed

    def sample(self, accs):
        n = len(accs)
        n_max_pairs = int(self.max_compare_ratio * n)
        acc_diff = np.array(accs)[:, None] - np.array(accs)
        acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
        ex_thresh_inds = np.where(acc_abs_diff_matrix > self.compare_threshold)
        ex_thresh_num = len(ex_thresh_inds[0])
        if ex_thresh_num > n_max_pairs:
            if self.choose_pair_criterion == "diff":
                keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
            elif self.choose_pair_criterion == "random":
                keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
            else:
                raise NotImplementedError()
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
        better_labels = (acc_diff > 0)[ex_thresh_inds]
        return ex_thresh_inds, better_labels

    def __call__(self, data_loader, epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_avg = AverageMeter()
        accs_avg = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(data_loader)) if not self.distributed or dist.get_rank() == 0 else None

        for i, (archs, f_accs, h_accs) in enumerate(data_loader):
            data_time.update(time.time() - end)

            archs = np.array(archs, dtype=object)
            accs = np.array(f_accs)
            ex_thresh_inds, better_labels = self.sample(accs)
            archs_1, archs_2 = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]]

            scores_1 = self.model(archs_1)
            scores_2 = self.model(archs_2)

            # better_pm = 2 * scores_1.new(np.array(better_labels, dtype=np.float32)) - 1
            # zero_ = scores_1.new([0.])
            # margin = [0.1]
            # margin = scores_1.new(margin)
            # loss = torch.mean(torch.max(zero_, margin - better_pm * (scores_2 - scores_1)))
            loss = margin_linear(scores_1, scores_2, better_labels, self.compare_margin)

            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            with torch.no_grad():
                rsts = scores_2 > scores_1
                prec = (rsts == rsts.new(better_labels)).float().mean()
                if self.distributed:
                    temp = torch.stack([loss, prec])
                    dist.barrier()
                    dist.reduce(temp, dst=0)
                    if dist.get_rank() == 0:
                        temp /= dist.get_world_size()
                    loss, prec = temp
                loss_avg.update(loss, rsts.size(0))
                accs_avg.update(prec, rsts.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bta:.3f} | " \
                             "Loss {N_lossa:.3f} | Prec {N_preca:.2f}".format(
                    N_epoch=epoch, N_batch=i+1, N_size=len(data_loader), N_bta=batch_time.avg,
                    N_lossa=loss_avg.avg, N_preca=accs_avg.avg
                )
                bar.next()
        if bar is not None:
            bar.finish()
        return
