import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
from statistics import mean, stdev


class GsApi:
    def __init__(self, api_loc):
        self.res = []
        print("Loading GradSign API from: {}".format(api_loc))
        f = open(api_loc, 'rb')
        while (1):
            try:
                self.res.append(pickle.load(f))
            except EOFError:
                break
        f.close()
        print("Done!")

    def __getitem__(self, item: int):
        return self.res[item]

    def __len__(self):
        return len(self.res)

    def get_score_by_index(self, index):
        assert index == self.res[index]["i"]
        return self.res[index]["score"]

    def get_time_by_index(self, index):
        assert index == self.res[index]["i"]
        return self.res[index]["time"]

    def get_acc_by_index(self, index):
        return self.res[index]["testacc"]


def load_api(dataset):
    print(f"loading {dataset} gs api......")
    if dataset == 'cifar10':
        return GsApi("./results/nb2_cf10_seed42_base.p")
    elif dataset == 'cifar100':
        return GsApi("./results/nb2_cf100_seed42_base.p")
    elif dataset == 'ImageNet16-120':
        return GsApi("./results/nb2_im120_seed42_base.p")
    else:
        print(f"{dataset} is not a valid dataset to load from")


def get_final_accuracy(dataset, api, uid, trainval):
    if dataset == 'cifar10':
        if not trainval:
            acc_type = 'ori-test'
        else:
            acc_type = 'x-valid'
    else:
        if not trainval:
            acc_type = 'x-test'
        else:
            acc_type = 'x-valid'
    if dataset == 'cifar10' and trainval:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', acc_type)
    else:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics(dataset, acc_type)
    return info['accuracy']