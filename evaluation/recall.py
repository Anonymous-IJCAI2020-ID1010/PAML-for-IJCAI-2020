import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
# import sklearn.metrics.pairwise


def pairwise_distances(x,y=None):
    x_norm = (x**2).sum(1).view(-1,1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1,-1)
    else:
        y = x
        y_norm = x_norm.view(1,-1)

    dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.t())
    return torch.sqrt(torch.clamp(dist, 1e-12, np.inf))


def assign_by_euclidian_at_k_hist(X, T, k, name):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)

    # get nearest points
    N = X.shape[0]
    indices = torch.zeros(N,k).long()
    if N < 10000:
        distances = pairwise_distances(X)
        #import pdb
        #pdb.set_trace()
        pos_mask = T.expand(N,N).eq(T.expand(N,N).t())
        neg_mask = torch.ones_like(pos_mask) - pos_mask
        pos_dist = torch.masked_select(distances,pos_mask).numpy()
        neg_dist = torch.masked_select(distances,neg_mask).numpy()
        np.random.shuffle(pos_dist)
        np.random.shuffle(neg_dist)
        pos_dist = pos_dist[:10000]
        neg_dist = neg_dist[:10000]
        pos_dist = pos_dist[pos_dist>1e-1]
        plt.figure()
        plt.axis([0,2.5,0,500])
        plt.hist(pos_dist,bins=100,alpha=0.5,label='pos')
        plt.hist(neg_dist,bins=100,alpha=0.5,label='neg')
        plt.legend(loc='upper left',fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(name)
        plt.close('all')
        indices = np.argsort(distances, axis = 1)[:, 1 : k + 1]
    else:
        for i in range(0, N, 2000):
            temp_dist = pairwise_distances(X[i:i+2000],X)
            temp_indice = np.argsort(temp_dist, axis = 1)[:, 1 : k + 1]
            indices[i:i+2000] = temp_indice
        last_dist = pairwise_distances(X[i:],X)
        last_indice = np.argsort(last_dist, axis = 1)[:, 1 : k + 1]
        indices[i:] = last_indice
    return [[T[i] for i in ii] for ii in indices],indices


def assign_by_euclidian_at_k(X, T, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)

    # get nearest points
    N = X.shape[0]
    indices = torch.zeros(N,k).long()
    if N < 10000:
        distances = pairwise_distances(X)
        #import pdb
        #pdb.set_trace()
        pos_mask = T.expand(N,N).eq(T.expand(N,N).t())
        neg_mask = torch.ones_like(pos_mask) ^ pos_mask
        pos_dist = torch.masked_select(distances,pos_mask).numpy()
        neg_dist = torch.masked_select(distances,neg_mask).numpy()
        np.random.shuffle(pos_dist)
        np.random.shuffle(neg_dist)
        pos_dist = pos_dist[:10000]
        neg_dist = neg_dist[:10000]
        pos_dist = pos_dist[pos_dist>1e-1]
        plt.figure()
        plt.hist(pos_dist,bins=100,alpha=0.5,label='pos')
        plt.hist(neg_dist,bins=100,alpha=0.5,label='neg')
        plt.legend(loc='upper right')
        plt.savefig('similar.jpg')
        plt.close('all')
        indices = np.argsort(distances, axis = 1)[:, 1 : k + 1]
    else:
        for i in range(0, N, 2000):
            temp_dist = pairwise_distances(X[i:i+2000],X)
            temp_indice = np.argsort(temp_dist, axis = 1)[:, 1 : k + 1]
            indices[i:i+2000] = temp_indice
        last_dist = pairwise_distances(X[i:],X)
        last_indice = np.argsort(last_dist, axis = 1)[:, 1 : k + 1]
        indices[i:] = last_indice
    return [[T[i] for i in ii] for ii in indices]


def QG_assign_by_euclidian_at_k(Q, G, T, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    # distances = sklearn.metrics.pairwise.pairwise_distances(X)

    # get nearest points
    N = Q.shape[0]
    indices = torch.zeros(N,k).long()
    if N < 10000:
        distances = pairwise_distances(G)
        #import pdb
        #pdb.set_trace()
        pos_mask = T.expand(N,N).eq(T.expand(N,N).t())
        neg_mask = torch.ones_like(pos_mask) - pos_mask
        pos_dist = torch.masked_select(distances,pos_mask).numpy()
        neg_dist = torch.masked_select(distances,neg_mask).numpy()
        np.random.shuffle(pos_dist)
        np.random.shuffle(neg_dist)
        pos_dist = pos_dist[:10000]
        neg_dist = neg_dist[:10000]
        plt.figure()
        plt.hist(pos_dist,bins=100,alpha=0.5,label='pos')
        plt.hist(neg_dist,bins=100,alpha=0.5,label='neg')
        plt.legend(loc='upper right')
        plt.savefig('similar.jpg')
        plt.close('all')
        indices = np.argsort(distances, axis = 1)[:, 0 : k]
    else:
        for i in range(0, N, 2000):
            temp_dist = pairwise_distances(Q[i:i+2000],G)
            temp_indice = np.argsort(temp_dist, axis = 1)[:, 0 : k]
            indices[i:i+2000] = temp_indice
        last_dist = pairwise_distances(Q[i:],G)
        last_indice = np.argsort(last_dist, axis = 1)[:, 0 : k]
        indices[i:] = last_indice
    return [[T[i] for i in ii] for ii in indices],indices


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))
