from __future__ import print_function, absolute_import

import os
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
from modeling import models
from data import datasets, transforms
from engine.base.extractor import Extractor
from utils.logging import Logger
from utils.serialization import load_checkpoint


def argument_parser():
    parser = argparse.ArgumentParser(description='NAS with Pytorch Implementation')
    parser.add_argument('--gpu-ids', type=str, default='0')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=datasets.names())
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    # training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
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
    # cudnn.benchmark = True
    cudnn.deterministic = True

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))

    # Create dataloaders
    test_transforms = transforms.create(args.dataset, train=False)

    data_root = os.path.join(args.data_dir, args.dataset)
    train_dataset = datasets.create(args.dataset, data_root, train=True,  transform=test_transforms, download=True)
    test_dataset  = datasets.create(args.dataset, data_root, train=False, transform=test_transforms, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Create model
    model = models.create(args.arch, num_classes=len(train_dataset.classes))
    model = nn.DataParallel(model).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Load from checkpoint
    checkpoint = load_checkpoint(os.path.join(args.logs_dir, 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])

    # Create Extractor
    extractor = Extractor(model)

    save_path = os.path.join(data_root, 'features_train.pkl')
    extractor(train_loader, save_path)

    save_path = os.path.join(data_root, 'features_test.pkl')
    extractor(test_loader, save_path)


if __name__ == '__main__':
    parser = argument_parser()
    main(parser.parse_args())
