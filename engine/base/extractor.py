from __future__ import print_function, absolute_import

import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from utils import Bar
from utils.meters import AverageMeter
from modeling.metrics.classification import accuracy


class Extractor(object):
    def __init__(self, model):
        super(Extractor, self).__init__()
        self.model = model

    def __call__(self, data_loader, save_path):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_avg = AverageMeter()
        top1_avg = AverageMeter()
        top5_avg = AverageMeter()
        features = []
        labels = []
        end = time.time()
        bar = Bar('Processing', max=len(data_loader))

        for i, (data, target) in enumerate(data_loader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                data, target = data.cuda(), target.cuda()
                logits, feat = self.model(data)
                loss = F.cross_entropy(logits, target)
                prec1, prec5 = accuracy(logits, target, topk=(1, 5))
                loss_avg.update(loss,  target.size(0))
                top1_avg.update(prec1, target.size(0))
                top5_avg.update(prec5, target.size(0))
                features.append(feat.cpu().numpy())
                labels.append(target.cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = "Evaluating: [{N_batch}/{N_size}] | Time {N_bta:.3f} | " \
                         "Loss {N_lossa:.3f} | Prec1 {N_preca1:.2f} | Prec5 {N_preca5:.2f}".format(
                N_batch=i + 1, N_size=len(data_loader), N_bta=batch_time.avg,
                N_lossa=loss_avg.avg, N_preca1=top1_avg.avg, N_preca5=top5_avg.avg
            )
            bar.next()
        bar.finish()
        print("\nResult: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(loss_avg.avg, top1_avg.avg))

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = {'features': features, 'labels': labels}
        with open(save_path, "wb") as wf:
            pickle.dump(data, wf)
        print("\nSave done!\n")
        return top1_avg.avg
