import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import types
import copy

from optimizers import get_optim_scheduler


def get_some_data(train_dataloader, num_batches, device):
    traindata = []
    dataloader_iter = iter(train_dataloader)
    for _ in range(num_batches):
        traindata.append(next(dataloader_iter))
    inputs = torch.cat([a for a, _ in traindata])
    targets = torch.cat([b for _, b in traindata])
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets


def get_some_data_grasp(train_dataloader, num_classes, samples_per_class, device):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(train_dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device)
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y


def get_some_data_ost(dataloader, num_classes, batch_size, device):

    false_data, corr_data = dataloader

    f_inputs, f_targets = false_data["inputs"], false_data["targets"]
    c_inputs, c_targets = corr_data["inputs"], corr_data["targets"]
    f_idx = np.random.choice(f_inputs.shape[0], batch_size*2, replace=False)
    c_idx = np.random.choice(c_inputs.shape[0], batch_size*2, replace=False)
    # inputs_, targets_ = np.concatenate([f_inputs[f_idx], c_inputs[c_idx]], axis=0), \
    #                     np.concatenate([f_targets[f_idx], c_targets[c_idx]], axis=0)

    inputs_, targets_ = f_inputs[f_idx], f_targets[f_idx]

    x = torch.from_numpy(inputs_).to(device)
    y = torch.from_numpy(targets_).to(device)

    return x, y


def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array


def get_flattened_metric(net, metric, verbose=False):
    grad_list = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if 'classifier' not in name:
                grad_list.append(metric(layer).flatten())
                if verbose:
                    print(layer.__class__.__name__, metric(layer).flatten()[:10])
    flattened_grad = np.concatenate(grad_list)

    return flattened_grad


def get_batch_flattened_metric(net, N):
    grad_list = []
    for i in range(N):
        flat = []
        for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    feat = layer.weight.grad1[i].data.cpu().numpy()
                else:
                    feat = torch.zeros_like(layer.weight).cpu().numpy()
                    # if i == 0:
                    # print(layer.__class__.__name__, feat.flatten()[:10])
                flat.append(feat.flatten())
        grad_list.append(np.concatenate(flat))
    batch_grad = np.stack(grad_list)

    return batch_grad


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads

    if type(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_2(model):
    num = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.grad is not None:
                num += layer.weight.grad.data.numpy().shape[0]
            else:
                num += layer.weight.shape[0]
    return num


def prune_net(net, amount=0.25):
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.random_unstructured(m, name="weight", amount=amount)


def _get_ost(net, train_data, opt_config, loss_fn, split_data=1, skip_grad=False):

    prune_net(net)

    default_device = torch.cuda.current_device()

    inputs, targets = train_data
    targets = torch.randint(5, size=targets.size())
    inputs = inputs.cuda(device=default_device)
    targets = targets.cuda(device=default_device)

    # with torch.no_grad():
    #     init_features, _ = net(inputs)
    #     print(init_features.shape, net.classifier.weight.shape)
    #     exit(0)

    optimizer, scheduler, criterion = get_optim_scheduler(net.parameters(), opt_config)

    criterion = criterion.cuda(device=default_device)
    # net.eval()
    # features, logits = net(inputs)
    # loss_orig = criterion(logits, targets)

    # print(loss_orig, logits, targets)

    net.train()
    for i in range(500):
        optimizer.zero_grad()
        features, logits = net(inputs)
        loss = criterion(logits, targets)
        if (i+1) % 50 == 0:
            print(loss)
        if (i+1) == 100:
            ret = loss.item()
        loss.backward()
        optimizer.step()

    # net.eval()
    # default_device = torch.cuda.current_device()
    # criterion = criterion.cuda(device=default_device)
    # features, logits = net(inputs)
    # loss_update = criterion(logits, targets)

    # print(loss_update)

    return ret


def no_op(self,x):
    return x


def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net


def get_ost(net_orig, dataloader, dataload_info, device, opt_config, loss_fn=F.cross_entropy):

    dataload, num_imgs_or_batches, num_classes = dataload_info

    if not hasattr(net_orig,'get_prunable_copy'):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu()
    torch.cuda.empty_cache()

    #given 1 minibatch of data
    val = 0
    if dataload == 'random':
        train_data = get_some_data(dataloader, num_batches=num_imgs_or_batches, device=device)
    elif dataload == 'grasp':
        train_data = get_some_data_grasp(dataloader, num_classes, samples_per_class=num_imgs_or_batches,
                                              device=device)
    elif dataload == 'ost':
        train_data = get_some_data_ost(dataloader, num_classes, batch_size=128,
                                              device=device)
    else:
        raise NotImplementedError(f'dataload {dataload} is not supported')

    done, ds = False, 1

    while not done:
        try:

            val = _get_ost(net_orig.cuda(), train_data, opt_config, loss_fn=loss_fn, split_data=ds)

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done = False
                if ds == train_data[0].shape[0] // 2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong')
                ds += 1
                while train_data[0].shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    # net_orig = net_orig.to(device).train()
    return val
