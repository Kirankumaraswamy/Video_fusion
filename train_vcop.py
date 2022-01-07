"""Video clip order prediction."""
import os
import math
import itertools
import argparse
import time
import random
import shutil
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets.ucf101 import UCF101VCOPDataset, UCF101ClipRetrievalDataset
from datasets.ucf50 import UCF50ClipRetreival, UCF50VCOP
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
from models.c3d_small import C3DSMALL
from models.vcopn import VCOPN

from helpers.data_utils import VideoAugment

import logging

EXPERIMENTS_DIR="logs"
EXPERIMENT_NAME="VCOP"
TENSORBOARD_DIR = os.path.join(os.path.dirname(__file__), EXPERIMENTS_DIR, "runs", EXPERIMENT_NAME)
LOG_DIR = os.path.join(os.path.dirname(__file__), EXPERIMENTS_DIR, EXPERIMENT_NAME)
writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

os.makedirs(LOG_DIR, exist_ok=True)
time_stamp = datetime.now().strftime("%Y%m%d%H%M%S.%f")
logging.basicConfig(filename=os.path.join(LOG_DIR, EXPERIMENT_NAME+"_"+ time_stamp +".log"), level=logging.DEBUG)


def order_class_index(order):
    """Return the index of the order in its full permutation.
    
    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    return classes.index(tuple(order.tolist()))


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        #print("Batch: ", i, " of ", len(train_dataloader))
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # compute loss and acc
        running_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print statistics and write summary every N batch
        '''if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            avg_acc = correct / (args.pf * args.bs)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch-1)*len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0'''
    # summary params and grads per eopch
    for name, param in model.named_parameters():
        writer.add_histogram('params/{}'.format(name), param, epoch)
        writer.add_histogram('grads/{}'.format(name), param.grad, epoch)

    avg_loss = running_loss / len(train_dataloader)
    return avg_loss


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    logging.info(f'Validation Loss = {avg_loss}, val_acc = {avg_acc}')
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        targets = [order_class_index(order) for order in tuple_orders]
        targets = torch.tensor(targets).to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def parse_args():

    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--data', type=str, default='/home/kiran/kiran/Thesis/code/kiran_code/datasets/UCF50_small1', metavar='DIR', help='path to dataset')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r21d', help='c3d/r3d/r21d/c3d_small')
    parser.add_argument('--cl', type=int, default=8, help='clip length')
    parser.add_argument('--it', type=int, default=4, help='interval')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', default="/home/kiran/kiran/Thesis/code/kiran_code/datasets/logs", type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=4, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=2, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    args = parser.parse_args()
    return args

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(os.path.dirname(__file__), "logs", filename))
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    logging.info(vars(args))

    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    if args.model == 'c3d':
        base = C3D(with_classifier=False)
    elif args.model == 'r3d':
        base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif args.model == 'r21d':   
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif args.model == 'c3d_small':
        base = C3DSMALL(with_classifier=False, num_classes=50)
    vcopn = VCOPN(base_network=base, feature_size=512, tuple_len=args.tl).to(device)

    if args.mode == 'train':  ########### Train #############

        traindir = os.path.join(args.data, "train")

        # augmentation_gpu = VideoAugment(args.crop_size)

        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),  # smaller edge to 128
            transforms.RandomCrop(112),
            transforms.ToTensor()
        ])



        train_dataset = UCF50VCOP(traindir, args.cl, args.it, args.tl, True, train_transforms)
        # train_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, True, train_transforms)
        # split val for 800 videos
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-8, 8))
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        logging.info('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

        if args.ckpt:
            pass
        else:
            # save graph and clips_order samples
            for data in train_dataloader:
                tuple_clips, tuple_orders = data
                for i in range(args.tl):
                    writer.add_video('train/tuple_clips', tuple_clips[:, i, :, :, :, :], i, fps=8)
                    writer.add_text('train/tuple_orders', str(tuple_orders[:, i].tolist()), i)
                tuple_clips = tuple_clips.to(device)
                writer.add_graph(vcopn, tuple_clips)
                break
            # save init params at step 0
            for name, param in vcopn.named_parameters():
                writer.add_histogram('params/{}'.format(name), param, 0)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(vcopn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

        if args.ckpt:  # resume training
            if os.path.isfile(args.ckpt):
                print("=> loading checkpoint '{}'".format(args.ckpt))
                if args.gpu is None:
                    checkpoint = torch.load(args.ckpt)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.ckpt, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                vcopn.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.ckpt, checkpoint['epoch']))

            log_dir = os.path.dirname(args.ckpt)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            if args.desp:
                exp_name = '{}_cl{}_it{}_tl{}_{}_{}'.format(args.model, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_it{}_tl{}_{}'.format(args.model, args.cl, args.it, args.tl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
        writer = SummaryWriter(log_dir)

        prev_best_val_loss = float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train_loss = train(args, vcopn, criterion, optimizer, device, train_dataloader, writer, epoch)
            val_loss = validate(args, vcopn, criterion, device, val_dataloader, writer, epoch)
            print('Epoch: {} -> Epoch time: {:.2f} s. Training loss: {}, Validation loss: {}'.format(epoch, time.time() - time_start, train_loss, val_loss))
            logging.info('Epoch: {} -> Epoch time: {:.2f} s. Training loss: {}, Validation loss: {}'.format(epoch, time.time() - time_start, train_loss, val_loss))
            # scheduler.step(val_loss)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            # save model every 20 epoches
            # save model for the best val
            if epoch % 20 == 0 :
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': vcopn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='vcopn_checkpoint_{:04d}.pth.tar'.format(epoch))
    elif args.mode == 'test':  ########### Test #############
        vcopn.load_state_dict(torch.load(args.ckpt))
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])
        test_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, False, test_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, vcopn, criterion, device, test_dataloader)


