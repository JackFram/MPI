##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##############################################################################
# Random Search for Hyper-Parameter Optimization, JMLR 2012 ##################
##############################################################################
# python ./exps/NATS-algos/random_wo_share.py --dataset cifar10 --search_space tss
# python ./exps/NATS-algos/random_wo_share.py --dataset cifar100 --search_space tss
# python ./exps/NATS-algos/random_wo_share.py --dataset ImageNet16-120 --search_space tss
##############################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn

from xautodl.config_utils import load_config, dict2config, configure2str
from xautodl.datasets import get_datasets, SearchDataset
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import CellStructure, get_search_spaces
from nats_bench import create
from gs import *
from regularized_ea import mutate_topology_func, random_topology_func



def random_topology_func(op_names, max_nodes=4):
    # Return a random architecture
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return random_architecture


def random_size_func(info):
    # Return a random architecture
    def random_architecture():
        channels = []
        for i in range(info["numbers"]):
            channels.append(str(random.choice(info["candidates"])))
        return ":".join(channels)

    return random_architecture


def main(xargs, api, gs_api):
    torch.set_num_threads(4)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    logger.log("{:} use api : {:}".format(time_string(), api))
    api.reset_time()

    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    if xargs.search_space == "tss":
        random_arch = random_topology_func(search_space)
    else:
        random_arch = random_size_func(search_space)

    best_arch, best_acc, total_time_cost, history = None, -1, [], []
    current_best_index = []
    while len(total_time_cost) == 0 or total_time_cost[-1] < xargs.time_budget:
        # G-Random
        arch_list = []
        score_list = []
        for i in range(10):
            arch = random_arch()
            arch_id = api.query_index_by_arch(arch)
            score = gs_api.get_score_by_index(arch_id)
            arch_list.append(arch)
            score_list.append(score)
        best_index = np.argmax(np.array(score_list))
        arch = arch_list[best_index]

        accuracy, _, _, total_cost = api.simulate_train_eval(
            arch, xargs.dataset, hp="12"
        )
        total_time_cost.append(total_cost)
        history.append(arch)
        if best_arch is None or best_acc < accuracy:
            best_acc, best_arch = accuracy, arch
        logger.log(
            "[{:03d}] : {:} : accuracy = {:.2f}%".format(len(history), arch, accuracy)
        )
        current_best_index.append(api.query_index_by_arch(best_arch))
    logger.log(
        "{:} best arch is {:}, accuracy = {:.2f}%, visit {:} archs with {:.1f} s.".format(
            time_string(), best_arch, best_acc, len(history), total_time_cost[-1]
        )
    )

    info = api.query_info_str_by_arch(
        best_arch, "200" if xargs.search_space == "tss" else "90"
    )
    logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()

    # added by Zhihao Zhang
    best_index = api.query_index_by_arch(best_arch)
    acc = {}
    acc["cifar10_valid"] = get_final_accuracy("cifar10", api, best_index, True)
    acc["cifar10_test"] = get_final_accuracy("cifar10", api, best_index, False)
    acc["cifar100_valid"] = get_final_accuracy("cifar100", api, best_index, True)
    acc["cifar100_test"] = get_final_accuracy("cifar100", api, best_index, False)
    acc["in_valid"] = get_final_accuracy("ImageNet16-120", api, best_index, True)
    acc["in_test"] = get_final_accuracy("ImageNet16-120", api, best_index, False)

    return logger.log_dir, current_best_index, total_time_cost, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random NAS")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        choices=["tss", "sss"],
        help="Choose the search space.",
    )

    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--loops_if_rand", type=int, default=500, help="The total runs for evaluation."
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()

    api = create(None, args.search_space, fast_mode=True, verbose=False)

    args.save_dir = os.path.join(
        "{:}-{:}".format(args.save_dir, args.search_space),
        "{:}-T{:}".format(args.dataset, args.time_budget),
        "RANDOM",
    )
    print("save-dir : {:}".format(args.save_dir))

    gs_api = load_api(args.dataset)

    if args.rand_seed < 0:
        save_dir, all_info = None, collections.OrderedDict()
        stat = {
            "cifar10_valid": [],
            "cifar10_test": [],
            "cifar100_valid": [],
            "cifar100_test": [],
            "in_valid": [],
            "in_test": [],
                }
        for i in range(args.loops_if_rand):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, args.loops_if_rand))
            args.rand_seed = random.randint(1, 100000)
            save_dir, all_archs, all_total_times, acc = main(args, api, gs_api)
            all_info[i] = {"all_archs": all_archs, "all_total_times": all_total_times}
            for key in stat:
                stat[key].append(acc[key])
        print("Stats:")
        for key in stat:
            print("{}: {}+-{}".format(key, mean(stat[key]), stdev(stat[key])))
        save_path = save_dir / "results.pth"
        print("save into {:}".format(save_path))
        torch.save(all_info, save_path)
    else:
        main(args, api)
