from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class HardMiningLoss(nn.Module):
    def __init__(self, beta=None, margin=0, **kwargs):
        super(HardMiningLoss, self).__init__()
        self.beta = beta
        self.margin = margin

    def forward(self, inputs, targets, w):

        n = batch_size = inputs.size(0)

        # w = w.t()
        w_ = list()

        for i in targets:
            w_.append(w[i])
        w_ = torch.cat(w_, 0)
        w_ = w_.view(batch_size, -1)

        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, w_.t())
        # print(sim_mat)
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) ^ pos_mask
        # pos_mask = pos_mask - eyes_.eq(1)

        loss = []

        for i in range(n):

            pos_pair = torch.masked_select(sim_mat[i], pos_mask[i])
            neg_pair = torch.masked_select(sim_mat[i], neg_mask[i])

            pos_pair_ = torch.sort(pos_pair)[0]
            neg_pair_ = torch.sort(neg_pair)[0]

            # neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - self.margin)
            # pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + self.margin)
            neg_pair = neg_pair_
            pos_pair = pos_pair_

            pos_loss = torch.mean(1 - pos_pair)
            neg_loss = torch.mean(neg_pair)
            loss.append(pos_loss + neg_loss)

        return sum(loss)/n


        # raw version
        # for i in range(n):

        #     pos_pair = torch.masked_select(sim_mat[i], pos_mask[i])
        #     neg_pair = torch.masked_select(sim_mat[i], neg_mask[i])

        #     pos_pair_ = torch.sort(pos_pair)[0]
        #     neg_pair_ = torch.sort(neg_pair)[0]

        #     neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - self.margin)
        #     pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + self.margin)
        #     neg_pair = neg_pair_
        #     pos_pair = pos_pair_

        #     pos_loss = torch.mean(1 - pos_pair)
        #     neg_loss = torch.mean(neg_pair)
        #     loss.append(pos_loss + neg_loss)

        # return sum(loss)/n



