import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import math

"""
Learning rate adjustment used for CondenseNet model training
"""
def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = config.max_epoch * nBatch
        T_cur = (epoch % config.max_epoch) * nBatch + batch
        lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def to_one_hot(y, num_class):
    y_onehot = torch.FloatTensor(len(y), num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot


def to_float_hot(y, num_class):
    y_floathot = torch.FloatTensor(len(y), num_class)
    y_floathot.fill_(1e-4)
    y_floathot.scatter_(1, y.view(-1,1), 0.99)
    return y_floathot