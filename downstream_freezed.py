#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

import simsiam.loader
import simsiam.builder

from datasets.ucf101 import UCF101VCOPDataset, UCF101ClipRetrievalDataset
from datasets.ucf50 import UCF50ClipRetreival, UCF11, UCF11_pretrained
from torch.utils.data import DataLoader, random_split
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet, Classifier
from models.c3d_small import C3DSMALL

import logging
from tensorboardX import SummaryWriter

EXPERIMENTS_DIR="logs"
EXPERIMENT_NAME="UCF11"
TENSORBOARD_DIR = os.path.join(os.path.dirname(__file__), EXPERIMENTS_DIR, "runs", EXPERIMENT_NAME)
LOG_DIR = os.path.join(os.path.dirname(__file__), EXPERIMENTS_DIR, EXPERIMENT_NAME)
writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

os.makedirs(LOG_DIR, exist_ok=True)
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S.%f")
logging.basicConfig(filename=os.path.join(LOG_DIR, EXPERIMENT_NAME+"_"+ time_stamp +".log"), level=logging.DEBUG)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='datasets/UCF11',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='r21d',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=101, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=512, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

parser.add_argument('--clip_length', type=int, default=8, help='clip length')
parser.add_argument('--clip_interval', type=int, default=4, help='interval')
parser.add_argument('--number_of_clips', type=int, default=1, help='tuple length')

def main():
    args = parser.parse_args()

    args.data = os.path.join(os.path.dirname(__file__), args.data)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print("backend=",args.dist_backend, " init_method=",args.dist_url,
                                " world_size=",args.world_size, " rank=",args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    ########### model ##############
    if args.arch == 'c3d':
        base = C3D(with_classifier=False)
    elif args.arch == 'r3d':
        base = R3DNet(layer_sizes=(1, 1, 1, 1), with_classifier=False)
    elif args.arch == 'r21d':
        base = R2Plus1DNet(layer_sizes=(1, 1, 1, 1), with_classifier=False, num_classes=11, zero_init_residual=True)
    elif args.arch == 'c3d_small':
        base = C3DSMALL(with_classifier=False, num_classes=50)

    model = base

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm


    # use pretrained weights
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> Pretrained path '{}'".format(args.pretrain))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrain)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrain, map_location=loc)

                model_dict = model.state_dict()
                # base encoder keys are different in 2 models
                pretrained_dict = {k.replace(".encoder.0", ""): v for k, v in checkpoint['state_dict'].items() if k.replace(".encoder.0", "") in model_dict}
                for k, v in model_dict.items():
                    if k not in pretrained_dict.keys():
                        pretrained_dict[k] = v
                model_dict.update(pretrained_dict)
                model.load_state_dict(pretrained_dict)

                for name, child in model.named_children():
                    for name, param in child.named_parameters():
                        if not (name == "linear.weight" or name == "linear.bias"):
                            param.requires_grad = False
                        else:
                            param.requires_grad = True


            print("=> loaded pretrained weights '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no pretrained checkpoint found at '{}'".format(args.pretrain))


    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = UCF11_pretrained(traindir, args.clip_length, args.clip_interval, args.number_of_clips, True, augmentation, extensions=("mpg"), model=model, args=args)

    # train_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, True, train_transforms)

    print('TRAIN video number: {}'.format(len(train_dataset)))

    all_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=True)


    l = len(train_dataset)
    train_length = int(l * 0.7)
    val_length = int(l * 0.2)
    test_length = l - train_length - val_length
    train_dataset, val_test_dataset = random_split(train_dataset,  (train_length, val_length + test_length))
    val_dataset, test_dataset = random_split(val_test_dataset,  (val_length, test_length))

    print('TRAIN video number: {}, VAL video number: {}, Test video number: {}.'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
    logging.info('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)'''

    training_losses = []
    training_accs = []
    val_losses = []
    val_accs = []

    model = Classifier(num_classes=11)
    if args.gpu is not None:
        model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        time_start = time.time()

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        val_loss, val_acc = validation(val_dataloader, model, criterion, optimizer, epoch, args)
        print('Epoch: {} -> Epoch time: {:.2f} s. Training loss: {} , Training acc: {} => Validation loss: {}, Validation acc: {}'.format(epoch, time.time() - time_start, train_loss, train_acc, val_loss, val_acc))
        logging.info('Epoch: {} -> Epoch time: {:.2f} s. Training loss: {} , Training acc: {} => Validation loss: {}, Validation acc: {}'.format(epoch, time.time() - time_start, train_loss, train_acc, val_loss, val_acc))

        if epoch % 20 == 0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='downstream_checkpoint_{:04d}.pth.tar'.format(epoch))

        training_losses.append(train_loss)
        training_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    test_loss, test_acc = test(test_dataloader, model, criterion, optimizer, epoch, args)
    print('Test loss: {} , Test acc: {}'.format( test_loss, test_acc))

    print(training_losses)
    print(training_accs)
    print(val_losses)
    print(val_accs)


def train(train_loader, model, criterion, optimizer, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    total_loss = 0.0
    correct = 0

    end = time.time()
    for i, (features, targets) in enumerate(train_loader):
        #print("Batch: ", i, " of ", len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            features = features.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()
        # compute output and loss
        outputs= model(features)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), features.size(0))

        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / len(train_loader.dataset)
    return avg_loss, avg_acc

def validation(validation_loader, model, criterion, optimizer, epoch, args):
    torch.set_grad_enabled(False)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(validation_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.eval()
    total_loss = 0.0
    correct = 0

    end = time.time()
    for i, (features, targets) in enumerate(validation_loader):
        #print("Batch: ", i, " of ", len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            features = features.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        outputs = model(features)

        loss = criterion(outputs, targets)

        losses.update(loss.item(), features.size(0))

        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    avg_loss = total_loss / len(validation_loader)
    avg_acc = correct / len(validation_loader.dataset)
    return avg_loss, avg_acc

def test(test_loader, model, criterion, optimizer, epoch, args):
    torch.set_grad_enabled(False)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.eval()
    total_loss = 0.0
    correct = 0

    end = time.time()
    for i, (features, targets) in enumerate(test_loader):
        #print("Batch: ", i, " of ", len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = features.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        outputs = model(features)

        loss = criterion(outputs, targets)

        losses.update(loss.item(), images[0].size(0))

        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    avg_loss = total_loss / len(test_loader)
    avg_acc = correct / len(test_loader.dataset)
    return avg_loss, avg_acc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(os.path.dirname(__file__), "logs", filename))
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
