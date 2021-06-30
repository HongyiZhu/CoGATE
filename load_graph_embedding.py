from graph_embedding_config             import *
import numpy as np
import pickle


def load_embedding(filename):
    f = open(filename, "r")
    n, d = [int(x) for x in f.readline().split(" ")]
    embedding = np.zeros((n, d))
    for i in range(n):
        l = np.asarray([float(x) for x in f.readline().split(" ")])
        for j in range(d):
            embedding[int(l[0])][j] = l[j + 1]
    f.close()

    return embedding
