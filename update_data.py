# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pickle
import torch
import argparse
import time
import random
from nats_bench import create
from datasets import *
from gradsign import get_gradsign
from ost import get_ost
import xautodl
from xautodl.models import get_cell_based_tiny_net
from xautodl.config_utils import dict2config, load_config
import os
import numpy as np
from optimizers import get_optim_scheduler

import abc

def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='data',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./train_results',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='orthogonal',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero, orthogonal]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=3, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true',
                        help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    parser.add_argument('--config_path', type=str, default='./configs/nas-benchmark/hyper-opts/200E.config', help='configs for optimizer')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = "cpu"
    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{name}(val={val}, avg={avg}, count={count})".format(
            name=self.__class__.__name__, **self.__dict__
        )


def procedure(xloader, network, criterion, scheduler, optimizer, mode: str):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    if mode == "train":
        network.train()
    elif mode == "valid":
        network.eval()
    else:
        raise ValueError("The mode is not right : {:}".format(mode))
    device = torch.cuda.current_device()
    data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()
    for i, (inputs, targets) in enumerate(xloader):
        if mode == "train":
            scheduler.update(None, 1.0 * i / len(xloader))

        targets = targets.cuda(device=device, non_blocking=True)
        if mode == "train":
            optimizer.zero_grad()
        # forward
        features, logits = network(inputs)
        loss = criterion(logits, targets)
        # backward
        if mode == "train":
            loss.backward()
            optimizer.step()
        # record loss and accuracy
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # count time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg, top1.avg, top5.avg, batch_time.sum


def convert_secs2time(epoch_time, return_str=False):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    if return_str:
        str = "[{:02d}:{:02d}:{:02d}]".format(need_hour, need_mins, need_secs)
        return str
    else:
        return need_hour, need_mins, need_secs


def time_string():
    ISOTIMEFORMAT = "%Y-%m-%d %X"
    string = "[{:}]".format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


if __name__ == '__main__':
    args = parse_arguments()
    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                     args.num_data_workers)
    net = torch.load("model/base.pt")
    net.eval()
    default_device = torch.cuda.current_device()
    acc = 0
    size = 0

    false_input = []
    false_label = []

    criterion = torch.nn.CrossEntropyLoss()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(device=default_device)
        targets = targets.cuda(device=default_device)
        # forward
        # inputs.requires_grad = True

        for _ in range(5):
            inputs = torch.autograd.Variable(inputs, requires_grad=True)
            features, logits = net(inputs)
            loss = criterion(logits, targets)
            net.zero_grad()
            loss.backward()
            data_grad = inputs.grad.data
            inputs = fgsm_attack(inputs, 0.01, data_grad)

        # backward
        # record loss and accuracy

        idx = (logits.data.argmax(dim=1) != targets.data)

        false_input.append(inputs[idx].data.cpu().numpy())
        false_label.append(targets[idx].data.cpu().numpy())

    false_input = np.concatenate(false_input, axis=0)
    false_label = np.concatenate(false_label, axis=0)

    np.savez("./data/cf10_key/adv_false.npz", inputs=false_input, targets=false_label)





