from __future__ import print_function, absolute_import

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils import Bar
from utils.meters import AverageMeter
from scipy.stats import stats


class Evaluator(object):
    def __init__(self, model, distributed=False):
        super(Evaluator, self).__init__()
        self.model       = model
        self.distributed = distributed

    def __call__(self, data_loader):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(data_loader)) if not self.distributed or dist.get_rank() == 0 else None

        all_scores = []
        true_accs = []
        for i, (archs, f_accs, h_accs) in enumerate(data_loader):
            data_time.update(time.time() - end)
            archs = np.array(archs, dtype=object)
            accs = np.array(f_accs)
            with torch.no_grad():
                scores = torch.sigmoid(self.model(archs))
                all_scores += list(scores.cpu().numpy())
                true_accs += list(accs)

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Evaluating: [{N_batch}/{N_size}] | Time {N_bta:.3f} | ".format(
                    N_batch=i + 1, N_size=len(data_loader), N_bta=batch_time.avg
                )
                bar.next()
        if bar is not None:
            bar.finish()

        corr = stats.kendalltau(true_accs, all_scores).correlation
        if not self.distributed or dist.get_rank() == 0:
            print("\nResult: Kendall tau: {:.4f}\n".format(corr))
        return corr
