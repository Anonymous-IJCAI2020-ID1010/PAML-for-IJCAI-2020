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

flag_save = True

def predict_batchwise(gpu_device, model, dataloader):
    device = torch.device("cuda")
    # torch.cuda.set_device(gpu_device)
    # k = dataloader.dataset.imgs

    with torch.no_grad():
        X, Y = zip(*[
            [x, y] for X, Y in dataloader
                for x, y in zip(
                    model(X.to(device)).cpu(), 
                    Y
                )
        ])

    # if flag_save == True:
    #     feature = [x.numpy() for x in X]
    #     np.save('label_feature.npy', {'label':np.array(k),'feature':feature})

    return torch.stack(X), torch.stack(Y)


def evaluate(gpu_device, model, dataloader, nb_classes, name='tra_similar.jpg'):
    # calculate embeddings with model, also get labels (non-batch-wise)
    X, T = predict_batchwise(gpu_device, model, dataloader)
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
        # get predictions by assigning nearest 8 neighbors with euclidian
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

def QG_evaluate(gpu_device, model, queryloader, galleryloader, nb_classes, name='tra_similar.jpg'):
    # calculate embeddings with model, also get labels (non-batch-wise)
    X, GT_Q = predict_batchwise(gpu_device, model, queryloader)
    Q = X
    X, GT_G = predict_batchwise(gpu_device, model, galleryloader)
    G = X
    # calculate NMI with kmeans clustering
    nmi=0.0
    if Q.shape[0]<10000:
        nmi = 1.0
        #Y = evaluation.assign_by_euclidian_at_k(X, T, 8)
        Y = evaluation.assign_by_euclidian_at_k_hist(G, GT_G, 1000, name)
        # calculate recall @ 1, 2, 4, 8
        recall = []
        for k in [1, 10, 100, 1000]:
            r_at_k = evaluation.calc_recall_at_k(GT_G, Y, k)
            recall.append(r_at_k)
        return nmi, recall
    else:
        # get predictions by assigning nearest 8 neighbors with euclidian
        Y,indices = evaluation.QG_assign_by_euclidian_at_k(Q, G, GT_G, 1000)

        # calculate recall @ 1, 10, 100, 1000
        recall = []
        for k in [1, 10, 100, 1000]:
            r_at_k = evaluation.calc_recall_at_k(GT_Q, Y, k)
            recall.append(r_at_k)
        return nmi, recall


# def QG_evaluate(gpu_device, model, queryloader, galleryloader, nb_classes, name='tra_similar.jpg'):
#     # calculate embeddings with model, also get labels (non-batch-wise)
#     Q, GT_Q = predict_batchwise(gpu_device, model, queryloader)
#     G, GT_G = predict_batchwise(gpu_device, model, galleryloader)
#     nmi=0.0
#     if Q.shape[0]<10000:
#         nmi = 1.0
#         Y = evaluation.assign_by_euclidian_at_k(G, GT_G, 8, name)
#         # calculate recall @ 1, 2, 4, 8
#         recall = []
#         for k in [1, 10, 100, 1000]:
#             r_at_k = evaluation.calc_recall_at_k(GT_G, Y, k)
#             recall.append(r_at_k)
#         return nmi, recall
#     else:
#         # get predictions by assigning nearest 8 neighbors with euclidian
#         Y = evaluation.QG_assign_by_euclidian_at_k(Q, G, GT_G, 40)

#         # calculate recall @ 1, 10, 100, 1000
#         recall = []
#         for k in [1, 10, 100, 1000]:
#             r_at_k = evaluation.calc_recall_at_k(GT_Q, Y, k)
#             recall.append(r_at_k)
#         return nmi, recall
