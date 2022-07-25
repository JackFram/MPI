import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random


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
        return self.res[index]["score"]

    def get_time_by_index(self, index):
        return self.res[index]["time"]

    def get_acc_by_index(self, index):
        return self.res[index]["testacc"]

    def get_arch_by_index(self, index):
        return self.res[index]["arch"]


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


if __name__ == "__main__":
    fontsize = 23
    # CIFAR-10
    try:
        gs_api = GsApi("./test_results/nb2_cf10_seed43_base.p")
        # gs_api = GsApi("./stats/nb2_cf10_seed42_base.p")
        print("{} has finished for cifar 10".format(len(gs_api)))

        # for i in range(len(gs_api)):
        #     res = gs_api[i]
        #     if res["testacc"] < 80:
        #         param_size = res["params"]
        #         arch_str = res["arch"]
        #         trainacc = res['trainacc']
        #         valacc = res['valacc']
        #         testacc = res['testacc']
        #         print(f"arch str:{arch_str}, "
        #               f"train acc: {trainacc}, "
        #               f"val acc: {valacc}, "
        #               f"test acc: {testacc}, "
        #               f"param size: {param_size}")

        acc = []
        score = []
        for i in range(len(gs_api)):
            acc.append(gs_api.get_acc_by_index(i))
            score.append(gs_api.get_score_by_index(i))
            # print(i, gs_api.get_acc_by_index(i), gs_api.get_score_by_index(i), gs_api.get_arch_by_index(i), gs_api[i]["mean"], gs_api[i]["std"])
            print(i, gs_api.get_acc_by_index(i), gs_api.get_score_by_index(i), gs_api.get_arch_by_index(i))
        print("Spearman's rho: {}".format(stats.spearmanr(acc, score, nan_policy='omit').correlation))
        tau, p = stats.kendalltau(acc, score)
        print("Kendall's Tau: {}".format(tau))

        # print("Generating correlation plot")
        # sample_num = 250
        # index = random.sample(range(len(gs_api)), k=sample_num)
        # color = {"CIFAR10": "r", "CIFAR100": "g", "ImageNet16-120": "b"}
        # plt.scatter(np.array(score)[index], np.array(acc)[index], color='r')
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.xlabel("GradSign metric score", fontsize=fontsize)
        # plt.ylabel("Model accuracy", fontsize=fontsize)
        # plt.title("{}".format("CIFAR-10"), fontsize=fontsize)
        # plt.savefig("./CIFAR-10.pdf", bbox_inches="tight", dpi=500)
        # plt.clf()
    except:
        pass

    # CIFAR-100
    try:
        gs_api = GsApi("./results/nb2_cf100_seed42_base.p")
        print("{} has finished for cifar 100".format(len(gs_api)))
        acc = []
        score = []
        for i in range(len(gs_api)):
            acc.append(gs_api.get_acc_by_index(i))
            score.append(gs_api.get_score_by_index(i))
        print("Spearman's rho: {}".format(stats.spearmanr(acc, score, nan_policy='omit').correlation))
        tau, p = stats.kendalltau(acc, score)
        print("Kendall's Tau: {}".format(tau))
        # print("Generating correlation plot")
        # sample_num = 1000
        # index = random.sample(range(len(gs_api)), k=sample_num)
        # color = {"CIFAR10": "r", "CIFAR100": "g", "ImageNet16-120": "b"}
        # plt.scatter(np.array(score)[index], np.array(acc)[index], color='g')
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.xlabel("GradSign metric score", fontsize=fontsize)
        # plt.ylabel("Model accuracy", fontsize=fontsize)
        # plt.title("{}".format("CIFAR-100"), fontsize=fontsize)
        # plt.savefig("./CIFAR-100.pdf", bbox_inches="tight", dpi=500)
        # plt.clf()
    except:
        pass

    # ImageNet16-120
    try:
        gs_api = GsApi("./results/nb2_im120_seed42_base.p")
        print("{} has finished for imagenet16-120".format(len(gs_api)))
        acc = []
        score = []
        for i in range(len(gs_api)):
            acc.append(gs_api.get_acc_by_index(i))
            score.append(gs_api.get_score_by_index(i))
        print("Spearman's rho: {}".format(stats.spearmanr(acc, score, nan_policy='omit').correlation))
        tau, p = stats.kendalltau(acc, score)
        print("Kendall's Tau: {}".format(tau))
        # print("Generating correlation plot")
        # sample_num = 1000
        # index = random.sample(range(len(gs_api)), k=sample_num)
        # # color = {"CIFAR10": "r", "CIFAR100": "g", "ImageNet16-120": "b"}
        # plt.scatter(np.array(score)[index], np.array(acc)[index], color='b')
        # plt.xticks(fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        # plt.xlabel("GradSign metric score", fontsize=fontsize)
        # plt.ylabel("Model accuracy", fontsize=fontsize)
        # plt.title("{}".format("ImageNet16-120"), fontsize=fontsize)
        # plt.savefig("./ImageNet16-120.pdf", bbox_inches="tight", dpi=500)
    except:
        pass
