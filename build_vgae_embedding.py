from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
import pickle
from torch import optim
from load_graph_embedding import load_embedding
from graph_embedding_config         import *

from vgae_model import GCNModelVAE
from vgae_utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, loss_function

class VGAE(object):
    def __init__(self, graph, embedding_path, configs):
        self.graph = graph
        self.embedding_path = embedding_path
        self.seed = 42
        self.epochs = 20
        self.hidden1 = 512
        self.hidden2 = 128
        self.lr = 0.01
        self.dropout = 0

        self.adj, self.features = load_data(configs)
        self.n_nodes, self.feat_dim = self.features.shape

        # Store original adjacency matrix (without diagonal entries) for later
        self.adj_orig = self.adj
        self.adj_orig = self.adj_orig - sp.dia_matrix((self.adj_orig.diagonal()[np.newaxis, :], [0]), shape=self.adj_orig.shape)
        self.adj_orig.eliminate_zeros()

        self.model = None
        self.optimizer = None

    def sample_graph(self):
        r = mask_test_edges(self.adj)
        f = open("{}VGAE_samples.pkl".format(self.embedding_path), 'wb')
        pickle.dump(r, f)
        f.close()
        self.adj_train, self.train_edges, self.val_edges, self.val_edges_false, self.test_edges, self.test_edges_false = r

        # g = open("./data/title and description/VGAE.pkl", 'rb')
        # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pickle.load(g)
        # g.close()

        self.adj = self.adj_train

        # Some preprocessing
        self.adj_norm = preprocess_graph(self.adj)
        self.adj_label = self.adj_train + sp.eye(self.adj_train.shape[0])
        # adj_label = sparse_to_tuple(adj_label)
        self.adj_label = torch.FloatTensor(self.adj_label.toarray())

        self.pos_weight = torch.Tensor([float(self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()])
        self.norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2)

    def build_model(self):
        self.model = GCNModelVAE(self.feat_dim, self.hidden1, self.hidden2, self.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, epochs):
        self.sample_graph()
        self.build_model()

        self.hidden_emb = None
        print("VGAE Start Training")
        for epoch in range(epochs):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()

            self.recovered, self.mu, self.logvar = self.model(self.features, self.adj_norm)
            self.loss = loss_function(preds=self.recovered, labels=self.adj_label,
                                    mu=self.mu, logvar=self.logvar, n_nodes=self.n_nodes,
                                    norm=self.norm, pos_weight=self.pos_weight)
            self.loss.backward()
            self.optimizer.step()

            self.cur_loss = self.loss.item()

            self.hidden_emb = self.mu.data.numpy()
            print(self.hidden_emb)
            self.roc_curr, self.ap_curr = get_roc_score(self.hidden_emb, self.adj_orig, self.val_edges, self.val_edges_false)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(self.cur_loss),
                    "val_ap=", "{:.5f}".format(self.ap_curr),
                    "time=", "{:.5f}".format(time.time() - t)
                    )
            f = open("{}/VGAE.nv".format(self.embedding_path), "w")
            f.write(" ".join([str(x) for x in self.hidden_emb.shape]))
            f.write("\n")
            for i in range(self.hidden_emb.shape[0]):
                d = " ".join([str(x) for x in self.hidden_emb[i]])
                f.write("{} {}\n".format(str(i), d))
            f.close()

        print("VGAE Optimization Finished!\n")

def build_vgae(g, path, configs):
    vgae = VGAE(g, path, configs)
    vgae.train(epochs)
    embedding = load_embedding("{}/VGAE.nv".format(path))
    return embedding