import torch
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import scipy.sparse as sp


def normalization_adj(adjacency):
    """calculate L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        The normalized adjacency matrix, the type is torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])  # add self-join
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()

    # transform to torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # indices = torch.from_numpy(
    #     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

    # values = torch.from_numpy(sparse_mx.data)
    indices = torch.LongTensor([sparse_mx.row, sparse_mx.col])
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_similarity_matrix(features, method='heat', sigma=1.0):
    """Get the similarity matrix"""
    dist = None
    if method == 'heat':
        dist = - (1/(2*sigma**2)) * pair(features) ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        # features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    return dist


def get_graph(features, topk=10, method='heat'):
    """Generate graph adjacency matrix using different similarity methods"""
    dist = get_similarity_matrix(features, method=method)

    # dist = get_similarity_matrix(dist, method=method)
    # dist = get_similarity_matrix(dist, method=method)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
    return edges_unordered


# single
def get_adjacency(features, n, topk=10, self_join=True, method='heat'):
    """Get the standardized adjacency matrix, sparse and dense"""
    # features = features.cpu().numpy()# to cpu
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = get_graph(features, topk, method)
    edges_unordered = np.array(edges_unordered, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    raw_adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if self_join:
        adj = adj + sp.eye(adj.shape[0])  # add self-join
    # raw_adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, raw_adj


def get_edges(dist, topk=10):
    """Through the similarity matrix, the graph structure is established"""
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
    return edges_unordered
