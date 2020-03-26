from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class Center(nn.Module):
    def __init__(self, classes = 100):
        super(Center, self).__init__()

    def forward(self, inputs, targets, classes, class_weight):
        n = inputs.size(0)
        num_dim = inputs.size(1)
        targets_ = list(set(targets.cpu().numpy().tolist()))
        classes_label = list(range(classes))
        # targets_ = list(set(targets.data))
        batch_num_class = len(targets_)

        # zero_center = torch.zeros(classes, num_dim)

        targets_ = Variable(torch.LongTensor(targets_)).cuda()
        mask_ = targets.repeat(batch_num_class, 1).eq(targets_.repeat(n, 1).t())
        centers = []
        inputs_list = []

        # calculate the centers for every class in one mini-batch
        for i, target in enumerate(targets_):
            mask_i = mask_[i].repeat(num_dim, 1).t()
            input_ = inputs[mask_i].resize(len(inputs[mask_i]) // num_dim, num_dim)
            centers.append(torch.mean(input_, 0))
            inputs_list.append(input_)

        # centers = [centers[i].resize(1, num_dim) for i in range(len(centers))]
        centers_ = []
        j = 0
        for i in range(classes):
            if i in targets_:
                centers_.append(centers[j].resize(1, num_dim))
                j = j + 1
            else:
                centers_.append(class_weight[i].resize(1, num_dim))
        centers = torch.cat(centers_, 0)
 
        return centers


