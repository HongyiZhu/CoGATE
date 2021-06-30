import pickle
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from graph_embedding_config import *

from scipy import sparse


def prepare_graph_data(adj, configs):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes) - sp.eye(num_nodes)  # self-loop
    if not configs.weighted_graph:
        adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col

def prepare_sparse_features(features):
    if not sp.isspmatrix_coo(features):
        features = sparse.csc_matrix(features).tocoo()
        features = features.astype(np.float32)
    indices = np.vstack((features.row, features.col)).transpose()
    return (indices, features.data, features.shape)

def conver_sparse_tf2np(input):
    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]

def load_data(configs):
    temp = open(configs.node_index_filename, 'rb')
    node_index = pickle.load(temp)
    temp.close()

    N = len(node_index)
    edges_unordered = np.genfromtxt(configs.edgelist_filename, dtype=np.int32)
    edges = np.array(edges_unordered).reshape(edges_unordered.shape)
    if configs.weighted_graph:
        adj = sp.coo_matrix((edges[:, 2], ([node_index[x] for x in edges[:, 0]], [node_index[x] for x in edges[:, 1]])),
                        shape=(N, N),
                        dtype=np.float32)
    else:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), ([node_index[x] for x in edges[:, 0]], [node_index[x] for x in edges[:, 1]])),
                        shape=(N, N),
                        dtype=np.float32)
    if configs.have_features:
        fin = open(configs.current_feature_file, 'r')
        feature_dict = {}
        for l in fin.readlines():
            vec = l.split()
            if int(vec[0]) in node_index.keys():
                n = node_index[int(vec[0])]
            else:
                continue
            feature_dict[n] = np.array([float(x) for x in vec[1:]])
        fin.close()
        feature_dim = feature_dict[0].shape[0]
        features = np.zeros((N, feature_dim), dtype=np.float32)
        for key in feature_dict.keys():
            for i, element in enumerate(feature_dict[key]):
                features[key][i] = element
    else:
        features = np.eye((N, N), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    return adj, features

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape