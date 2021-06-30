import numpy as np
import math
import torch
import gcae_utils
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# class SparseMM(torch.autograd.Function):
#     """
#     Sparse x dense matrix multiplication with autograd support.
#     Implementation by Soumith Chintala:
#     https://discuss.pytorch.org/t/
#     does-pytorch-support-autograd-on-sparse-matrix/6156/7
#     """

#     def forward(self, matrix1, matrix2):
#         self.save_for_backward(matrix1, matrix2)
#         return torch.mm(matrix1, matrix2)

#     def backward(self, grad_output):
#         matrix1, matrix2 = self.saved_tensors
#         grad_matrix1 = grad_matrix2 = None

#         if self.needs_input_grad[0]:
#             grad_matrix1 = torch.mm(grad_output, matrix2.t())

#         if self.needs_input_grad[1]:
#             grad_matrix2 = torch.mm(matrix1.t(), grad_output)

#         return grad_matrix1, grad_matrix2


class GraphConvolutionEncoder(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # output = SparseMM()(adj, support)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionDecoder(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionDecoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_inv):
        support = torch.mm(input, self.weight)
        # output = SparseMM()(adj_inv, support)
        output = torch.spmm(adj_inv, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'