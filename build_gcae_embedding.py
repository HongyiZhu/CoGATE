from gcae_model                     import GCNAutoencoder
from gcae_utils                     import load_data
from numpy                          import save
import time
import os
import numpy                        as np
import torch
import torch.optim                  as optim
from torch.autograd                 import Variable
from graph_embedding_config         import *
from load_graph_embedding           import load_embedding
import warnings

warnings.filterwarnings("ignore")

class GCAE(object):
    def __init__(self, graph, embedding_path, configs):
        self.graph = graph
        self.embedding_path = embedding_path
        self.seed = 42  # obtained from pygcn/train.py line 22
        self.epochs = 20  # obtained from pygcn/train.py line 23
        self.learning_rate = 0.01  # obtained from pygcn/train.py line 25
        self.weight_decay = 5e-4  # obtained from pygcn/train.py line 27
        self.dropout = 0.5  # obtained from pygcn/train.py line 31
        self.feature, self.adj, self.adj_inv = load_data(configs)
        # self.feature = torch.FloatTensor(np.eye(self.adj.shape[0])) # Identity matrix
        
        self.model = GCNAutoencoder(nfeat=self.feature.shape[1], dropout=self.dropout)
        # Define the loss function
        self.criterion = torch.nn.MSELoss()
        # Define the optimization function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.feature, self.adj, self.adj_inv = Variable(self.feature, requires_grad=False), Variable(self.adj, requires_grad=False), Variable(self.adj_inv, requires_grad=False)

    def train(self, ep):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        self.output, self.embedding = self.model(self.feature, self.adj, self.adj_inv)
        self.loss_train = self.criterion(self.output, self.feature)
        self.loss_train.backward()
        self.optimizer.step()

        print('Epoch: {:04d}'.format(ep + 1),
                'loss_train: {:.10f}'.format(self.loss_train.data),
                'time: {:.4f}s'.format(time.time() - t))
        self.temp = self.model.embedding.data.numpy()
        
        f = open("{}/GCAE.nv".format(self.embedding_path), "w")
        f.write(" ".join([str(x) for x in self.temp.shape]))
        f.write("\n")
        for i in range(self.temp.shape[0]):
            d = " ".join([str(x) for x in self.temp[i]])
            f.write("{} {}\n".format(str(i), d))
        f.close()
        
def build_gcae(g, path, configs):
    gcae = GCAE(g, path, configs)
    t_total = time.time()
    print("GCAE Start Training")
    for epoch in range(epochs):
        gcae.train(epoch)
    print("GCAE Optimization Finished!")
    print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total))
    embedding = load_embedding("{}/GCAE.nv".format(path))

    return embedding


