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
from nats_bench import create
from datasets import *
from gradsign import get_gradsign
import xautodl
from xautodl.models import get_cell_based_tiny_net
import os


def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='data',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./results',
                        type=str, help='output directory')
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=1, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--noacc', default=False, action='store_true',
                        help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = "cpu"
    return args


if __name__ == '__main__':
    args = parse_arguments()
    os.environ["TORCH_HOME"] = args.api_loc

    # if args.noacc:
    #     api = pickle.load(open(args.api_loc, 'rb'))
    # else:
    #     from nas_201_api import NASBench201API as API
    #
    #     api = API(args.api_loc)
    api = create(None, 'tss', fast_mode=True, verbose=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                     args.num_data_workers)

    cached_res = []
    pre = 'cf' if 'cifar' in args.dataset else 'im'
    pfn = f'nb2_{pre}{get_num_classes(args)}_seed{args.seed}_base.p'
    op = os.path.join(args.outdir, pfn)

    args.end = len(api) if args.end == 0 else args.end



    # loop over natsbench archs
    for i, arch_str in enumerate(api):

        if i < args.start:
            continue
        if i >= args.end:
            break

        res = {'i': i, 'arch': arch_str}

        info = api.get_cost_info(i, args.dataset)

        config = api.get_net_config(i, args.dataset)
        net = get_cell_based_tiny_net(config)
        net.to(args.device)

        init_net(net, args.init_w_type, args.init_b_type)

        # start = time.time()
        # # print(net)
        # score = get_gradsign(
        #     net,
        #     train_loader,
        #     (args.dataload, args.dataload_info, get_num_classes(args)),
        #     args.device
        # )
        #
        # score = info["params"]

        info = api.get_more_info(i, 'cifar10-valid' if args.dataset == 'cifar10' else args.dataset, iepoch=None,
                                 hp='200', is_random=False)

        trainacc = info['train-accuracy']
        valacc = info['valid-accuracy']
        testacc = info['test-accuracy']

        cost_info = api.get_cost_info(i, args.dataset)

        print(api.simulate_train_eval(1224, dataset='cifar10', hp='12'))
        exit(0)

        if testacc > 90:
            param_size = cost_info["params"]
            print(f"arch str:{arch_str}, "
                  f"train acc: {trainacc}, "
                  f"val acc: {valacc}, "
                  f"test acc: {testacc}, "
                  f"param size: {param_size}")
            print(cost_info)

