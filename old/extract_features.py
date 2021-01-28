from __future__ import print_function, absolute_import

import os
import sys
import time
import pickle
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from modeling import models
from data import datasets, transforms
from utils.logging import Logger
from utils import Bar
from utils.meters import AverageMeter
from modeling.metrics.classification import accuracy



def argument_parser():
    parser = argparse.ArgumentParser(description='NAS with Pytorch Implementation')
    parser.add_argument('--gpu-ids', type=str, default='0')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=datasets.names())
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    # training
    parser.add_argument('--seed', type=int, default=0)
    # misc
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'temp', 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'temp', 'logs'))
    # distributed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--net-card', type=str, default='', help="Name of the network card.")
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.net_card:
        os.environ['GLOO_SOCKET_IFNAME'] = args.net_card
        os.environ['NCCL_SOCKET_IFNAME'] = args.net_card

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    # cudnn.deterministic = True

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

    test_transforms = transforms.create(args.dataset, train=False)
    data_root = os.path.join(args.data_dir, args.dataset)
    test_dataset = datasets.create(args.dataset, data_root, train=False, transform=test_transforms, download=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    model = models.create(args.arch, num_classes=len(test_dataset.classes))
    model = nn.DataParallel(model).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_avg = AverageMeter()
    top1_avg = AverageMeter()
    top5_avg = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(test_loader))

    for i, (data, target) in enumerate(test_loader):
        data_time.update(time.time() - end)

        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            if self.distributed:
                temp = torch.stack([loss, prec1, prec5])
                dist.barrier()
                dist.reduce(temp, dst=0)
                if dist.get_rank() == 0:
                    temp /= dist.get_world_size()
                loss, prec1, prec5 = temp
            loss_avg.update(loss, target.size(0))
            top1_avg.update(prec1, target.size(0))
            top5_avg.update(prec5, target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if bar is not None:
            bar.suffix = "Evaluating: [{N_batch}/{N_size}] | Time {N_bta:.3f} | " \
                         "Loss {N_lossa:.3f} | Prec1 {N_preca1:.2f} | Prec5 {N_preca5:.2f}".format(
                N_batch=i + 1, N_size=len(data_loader), N_bta=batch_time.avg,
                N_lossa=loss_avg.avg, N_preca1=top1_avg.avg, N_preca5=top5_avg.avg
            )
            bar.next()
    if bar is not None:
        bar.finish()

    if not self.distributed or dist.get_rank() == 0:
        print("\nResult: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(loss_avg.avg, top1_avg.avg))
    return top1_avg.avg




if __name__ == '__main__':
    parser = argument_parser()
    main(parser.parse_args())