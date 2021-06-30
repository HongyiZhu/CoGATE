from gcn_layers import GraphConvolutionDecoder, GraphConvolutionEncoder
import torch.nn as nn
import torch.nn.functional as F

class GCNAutoencoder(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCNAutoencoder, self).__init__()

        self.ec1 = GraphConvolutionEncoder(nfeat, 128)
        # self.ec2 = GraphConvolutionEncoder(256, 128)
        # self.dc1 = GraphConvolutionDecoder(128, 256)
        self.dc2 = GraphConvolutionDecoder(128, nfeat)

        # self.ec1 = GraphConvolutionEncoder(nfeat, 256)
        # self.ec2 = GraphConvolutionEncoder(7068, 5012)
        # self.ec3 = GraphConvolutionEncoder(256, 128)
        # self.dc1 = GraphConvolutionDecoder(128, 256)
        # self.dc2 = GraphConvolutionDecoder(5012, 7068)
        # self.dc3 = GraphConvolutionDecoder(256, nfeat)

        # self.ec4 = GraphConvolutionEncoder(nfeat, 128)
        # self.dc4 = GraphConvolutionDecoder(128, nfeat)

        self.dropout = dropout
        self.embedding = None

    def forward(self, x, adj, adj_inv):
        # Encoder
        x = F.relu(self.ec1(x, adj))  # Combine feature and adjacency matrices
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.ec2(x, adj))  # Combine feature and adjacency matrices
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.ec3(x, adj))  # Combine feature and adjacency matrices

        # Got embedding here
        self.embedding = x
        # x = F.dropout(x, self.dropout, training=self.training)

        # Decoder
        # x = F.relu(self.dc1(x, adj_inv))  # Combine feature and adjacency matrices
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.dc2(x, adj_inv))  # Combine feature and adjacency matrices
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.dc3(x, adj_inv)  # Combine feature and adjacency matrices

        # x = F.leaky_relu(self.ec4(x, adj))
        # self.embedding = x
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.dc4(x, adj_inv)

        return x, self.embedding