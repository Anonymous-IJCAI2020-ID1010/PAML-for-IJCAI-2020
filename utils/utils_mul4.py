import evaluation
import numpy as np
import torch
import torch.nn.functional as F

'''
def pairwise_distances(x,y=None):
    x_norm = np.reshape(np.sum(x**2, axis=1), (-1, 1))
    if y is not None:
        y_norm = np.reshape(np.sum(y**2, axis=1), (1, -1))
    else:
        y = x
        y_norm = np.reshape(x_norm, (1,-1))
    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return np.clip(dist, 0, np.inf)
'''

def predict_batchwise(gpu_device, model, dataloader):
    device = torch.device("cuda")
    torch.cuda.set_device(gpu_device)
    with torch.no_grad():
        '''
        X2,X3,X4,X5,Y = zip(*[
            #[x2,x3,x4,x5, y] for X, Y in dataloader
            [x2,x3,x4,x5, y] for X, Y in dataloader
                #for x, y in zip(
                for x2,x3,x4,x5, y in zip(
                    model(X.to(device))[0].cpu(),
                    model(X.to(device))[1].cpu(),
                    model(X.to(device))[2].cpu(),
                    model(X.to(device))[3].cpu(),
                    Y
                )
        ])
        '''
        X2 = []
        X3 = []
        X4 = []
        X5 = []
        Y = []
        for x,y in dataloader:
            Z=model(x.to(device))
            X2.append(Z[0].cpu())
            X3.append(Z[1].cpu())
            X4.append(Z[2].cpu())
            X5.append(Z[3].cpu())
            Y.append(y) 
    X2 = torch.stack(X2).squeeze(),
    X3 = torch.stack(X3).squeeze(),
    X4 = torch.stack(X4).squeeze(),
    X5 = torch.stack(X5).squeeze(),
    Y = torch.stack(Y).squeeze(),
    return X2[0], X3[0], X4[0], X5[0], Y[0]
    #return X4[0], X5[0], Y[0]

def evaluate(gpu_device, model, dataloader, nb_classes):
    # calculate embeddings with model, also get labels (non-batch-wise)
    X2, X3, X4, X5, T = predict_batchwise(gpu_device, model, dataloader)
    #X4, X5, T = predict_batchwise(gpu_device, model, dataloader)
    #import pdb
    #pdb.set_trace()
    X = torch.cat([X2,X3,X4,X5],dim=1)
    #X = torch.cat([X2,X5],dim=1)
    #X = X5
    #X = F.normalize(X, p=2, dim=1)
    # calculate NMI with kmeans clustering
    nmi=0.0
    if X.shape[0]<10000:
        '''
        nmi = evaluation.calc_normalized_mutual_information(
        T, 
        evaluation.cluster_by_kmeans(
            X, nb_classes
        ))
        '''
        nmi = 1.0
        '''
        # get predictions by assigning nearest 8 neighbors with euclidian
        Y = evaluation.assign_by_euclidian_at_k(X2, T, 8)
        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in [1, 2, 4, 8]:
            r_at_k = evaluation.calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
        Y = evaluation.assign_by_euclidian_at_k(X3, T, 8)
        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in [1, 2, 4, 8]:
            r_at_k = evaluation.calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
        Y = evaluation.assign_by_euclidian_at_k(X4, T, 8)
        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in [1, 2, 4, 8]:
            r_at_k = evaluation.calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
        Y = evaluation.assign_by_euclidian_at_k(X5, T, 8)
        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in [1, 2, 4, 8]:
            r_at_k = evaluation.calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
        '''
        Y = evaluation.assign_by_euclidian_at_k(X, T, 8)
        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in [1, 2, 4, 8]:
            r_at_k = evaluation.calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
        return nmi, recall
    else:
        # get predictions by assigning nearest 8 neighbors with euclidian
        Y = evaluation.assign_by_euclidian_at_k(X, T, 1000)
        
        # calculate recall @ 1, 10, 100, 1000
        recall = []
        for k in [1, 10, 100, 1000]:
            r_at_k = evaluation.calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
        return nmi, recall
