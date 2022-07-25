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


if __name__ == '__main__':
    args = parse_arguments()
    os.environ["TORCH_HOME"] = args.api_loc

    prepare_seed(args.seed)

    # if args.noacc:
    #     api = pickle.load(open(args.api_loc, 'rb'))
    # else:
    #     from nas_201_api import NASBench201API as API
    #
    #     api = API(args.api_loc)
    api = create(None, 'tss', fast_mode=True, verbose=True)
    #
    # torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                     args.num_data_workers)

    cached_res = []
    pre = 'cf' if 'cifar' in args.dataset else 'im'
    pfn = f'nb2_{pre}{get_num_classes(args)}_seed{args.seed}_base.p'
    op = os.path.join(args.outdir, pfn)

    args.end = len(api) if args.end == 0 else args.end

    opt_config = load_config(
        args.config_path, None, None
    )

    # loop over natsbench archs
    for i, arch_str in enumerate(api):

        # if i != 9:
        #     continue

        if i < args.start:
            continue
        if i >= args.end:
            break

        res = {'i': i, 'arch': arch_str}

        config = api.get_net_config(i, args.dataset)

        cost_info = api.get_cost_info(i, args.dataset)

        # if cost_info["params"] != 0.372346:
        #     continue

        start = time.time()
        # print(net)
        score_list = []
        net = get_cell_based_tiny_net(config)
        net.to(args.device)
        optimizer, scheduler, criterion = get_optim_scheduler(net.parameters(), opt_config)
        default_device = torch.cuda.current_device()
        network = torch.nn.DataParallel(net, device_ids=[default_device]).cuda(
            device=default_device
        )
        start_time, epoch_time, total_epoch = (
            time.time(),
            AverageMeter(),
            opt_config.epochs + opt_config.warmup,
        )
        (
            train_losses,
            train_acc1es,
            train_acc5es,
            valid_losses,
            valid_acc1es,
            valid_acc5es,
        ) = ({}, {}, {}, {}, {}, {})
        train_times, valid_times, lrs = {}, {}, {}
        for epoch in range(total_epoch):
            scheduler.update(epoch, 0.0)
            lr = min(scheduler.get_lr())
            train_loss, train_acc1, train_acc5, train_tm = procedure(
                train_loader, network, criterion, scheduler, optimizer, "train"
            )
            train_losses[epoch] = train_loss
            train_acc1es[epoch] = train_acc1
            train_acc5es[epoch] = train_acc5
            train_times[epoch] = train_tm
            lrs[epoch] = lr
            with torch.no_grad():
                valid_loss, valid_acc1, valid_acc5, valid_tm = procedure(
                    val_loader, network, criterion, None, None, "valid"
                )

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            need_time = "Time Left: {:}".format(
                convert_secs2time(epoch_time.avg * (total_epoch - epoch - 1), True)
            )
            print(
                "{:} {:} epoch={:03d}/{:03d} :: Train [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%] Valid [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%], lr={:}".format(
                    time_string(),
                    need_time,
                    epoch,
                    total_epoch,
                    train_loss,
                    train_acc1,
                    train_acc5,
                    valid_loss,
                    valid_acc1,
                    valid_acc5,
                    lr,
                )
            )

        torch.save(network, "model/base.pt")
        info = api.get_more_info(i, 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset, iepoch=None,
                                 hp='200', is_random=False)

        trainacc = info['train-accuracy']
        valacc = info['valid-accuracy']
        testacc = info['test-accuracy']

        print(f"reported: {trainacc}, actual train acc: {train_acc1}")

        print(f"reported: {valacc}, actual val acc: {valid_acc1}")

        score = train_acc1

        res['trainacc'] = trainacc
        res['valacc'] = valacc
        res['testacc'] = testacc
        res['params'] = cost_info["params"]
        res['score'] = score
        # res['mean'] = mean
        # res['std'] = std

        print(score, testacc)

        cached_res.append(res)

        # print(res)

        # write to file
        # if i % args.write_freq == 0 or i == len(api) - 1 or i == 10:
        #     print(f'writing {len(cached_res)} results to {op} up to arch id {i}')
        #     pf = open(op, 'ab')
        #     for cr in cached_res:
        #         pickle.dump(cr, pf)
        #     pf.close()
        #     cached_res = []

