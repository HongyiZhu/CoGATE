import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scipy                  import stats
from sklearn.cluster        import KMeans, DBSCAN
from sklearn.manifold       import TSNE
from sklearn                import metrics
from mygraph                import Graph_Int, Graph_Str
from build_embedding        import *
from load_graph_embedding   import load_embedding
from gem.evaluation         import evaluate_graph_reconstruction as gr
from graph_embedding_config import *
from gva_utils              import load_json, dict2dotdict, dotdict2dict
import numpy as np
import networkx as nx
import argparse
import pickle
import time
import json


# evaluate embedding
def evaluate_embedding(graph, embedding):   
    class _Dummy(object):
        def __init__(self, embedding):
            self.embedding = embedding

        def get_reconstructed_adj(self, X=None, node_l=None):
            node_num = self.embedding.shape[0]
            adj_mtx_r = np.zeros((node_num, node_num))
            for v_i in range(node_num):
                for v_j in range(node_num):
                    if v_i == v_j:
                        continue
                    adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
            return adj_mtx_r
        
        def get_edge_weight(self, i, j):
            return np.dot(self.embedding[i, :], self.embedding[j, :])

    dummy_model = _Dummy(embedding)
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(graph, dummy_model, embedding, None)
    return (MAP, prec_curv)
    

# evaluate clustering
def evaluate_clustering_performance(X, prediction):
    silhouette = metrics.silhouette_score(X, prediction)
    chs = metrics.calinski_harabasz_score(X, prediction)
    dbs = metrics.davies_bouldin_score(X, prediction)
    
    return (silhouette, chs, dbs)


def process_node_index(edgelist_filename, node_index_filename, embedding_mapping):
    f = open(edgelist_filename, 'r')
    nodes = []
    for line in f.readlines():
        elements = line.strip().split()
        if len(elements) < 2:
            continue
        else:
            nodes.append(int(elements[0]))
            nodes.append(int(elements[1]))
    f.close()
    nodes = sorted(list(set(nodes)), key=lambda x: int(x))
    nodes_index = {x:i for i, x in enumerate(nodes)}
    f = open(node_index_filename, 'wb')
    pickle.dump(nodes_index, f)
    f.close()

    f = open(embedding_mapping, 'w')
    f.write("EmbeddingID, NodeID\n")
    for i, x in enumerate(nodes):
        f.write("{},{}\n".format(str(i), str(x)))
    f.close()

def main(configs, LOAD_TRAINED_EMBEDDING, n_cluster):
    process_node_index(configs.edgelist_filename, configs.node_index_filename, configs.embedding_mapping)
    temp = open(configs.node_index_filename, 'rb')
    node_index = pickle.load(temp)
    temp.close()

    # load dataset
    print("====================\nLoading edgelist")
    t1 = time.time()
    # load graph from edgelist and feature file
    graph = Graph_Int()
    graph.read_edgelist(filename=configs.edgelist_filename, node_index=node_index, weighted=configs.weighted_graph, directed=False)
    graph_str = Graph_Str()
    graph_str.read_edgelist(filename=configs.edgelist_filename, node_index=node_index, weighted=configs.weighted_graph, directed=False)
    if configs.have_features:
        graph.read_node_features(node_index=node_index, filename=configs.current_feature_file)
    print("Data Loaded. Time elapsed: {:.3f}\n====================\n".format(time.time() - t1))

    graph_embeddings = {}
    if LOAD_TRAINED_EMBEDDING:
        # load graph embeddings
        print("====================\nLoading Graph Embeddings\n")
        for model in configs.models:
            embedding_file = (f"{configs.current_embedding_path}/{model}.nv")
            graph_embeddings[model] = load_embedding(embedding_file)
        print("Embeddings Loaded.\n====================")
    else:
        # build graph embedding
        print("====================\nBuilding Graph Embeddings\n")
        t2 = time.time()
        for model in configs.models:
            graph_embeddings[model] = build_embedding(graph, graph_str, model, configs.current_embedding_path, configs)
        print("Embeddings Constructed. Total time elapsed: {:.3f}\n====================".format(time.time() - t2))

    # GEM graph reconstruction evaluation
    print("====================\nEvaluating Graph Embeddings")
    t3 = time.time()
    reconstruction_performance = {}
    for model in configs.models:
        reconstruction_performance[model] = evaluate_embedding(graph.G, graph_embeddings[model])
    print("Embeddings Evaluated. Total time elapsed: {:.3f}\n====================".format(time.time() - t3))

    # clustering evaluation
    print("====================\nEvaluating Node Clusters")
    t4 = time.time()
    kmeans_performance = {}
    dbscan_performance = {}
    
    # KMeans
    if KMEANS_EVAL:
        kmeans_prediction = {}
        tsne_kmeans = {}
        for model in configs.models:
            print("[KMeans] Clustering {} Embedding".format(model))
            temp_t = time.time()
            kmeans = KMeans(n_clusters=n_cluster).fit(graph_embeddings[model])
            kmeans_prediction[model] = kmeans.labels_
            kmeans_performance[model] = evaluate_clustering_performance(graph_embeddings[model], kmeans_prediction[model])
            print("[KMeans] Clustering Finished for {} Embedding. Time elapsed: {:.3f}".format(model, time.time() - temp_t))
    
    # DBSCAN
    if DBSCAN_EVAL:
        dbscan_predcition = {}
        tsne_dbscan = {}
        for model in configs.models:
            print("[DBSCAN] Clustering {} Embedding".format(model))
            temp_t = time.time()
            dbscan = DBSCAN(eps=eps).fit(graph_embeddings[model])
            dbscan_predcition[model] = dbscan.labels_
            dbscan_performance[model] = evaluate_clustering_performance(graph_embeddings[model], dbscan_predcition[model])
            print("[DBSCAN] Clustering Finished for {} Embedding. Time elapsed: {:.3f}".format(model, time.time() - temp_t))
    
    tsne_result = {}
    tsne_time = {}
    for model in configs.models:
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        temp_t = time.time()
        tsne_result[model] = tsne.fit_transform(graph_embeddings[model])
        t_model = time.time() - temp_t
        print("t-SNE for {} embedding finished ({}s)".format(model, t_model))
        tsne_time[model] = t_model

    print("Clustering Results Evaluated. Total time elapsed: {:.3f}\n====================".format(time.time() - t4))

    # Generate Report
    f = open("{}results-{}.tsv".format(configs.current_report_path, str(n_cluster)), "w")
    for model in configs.models:
        f.write("{}\t".format(model))
        MAP, prec_curv = reconstruction_performance[model]
        f.write("{:.3f}\t".format(MAP))
        if KMEANS_EVAL:
            k_s, k_c, k_d = kmeans_performance[model]
            f.write("{:.3f}\t{:.3f}\t{:.3f}\t".format(k_s, k_c, k_d))
        if DBSCAN_EVAL:
            d_s, d_c, d_d = dbscan_performance[model]
            f.write("{:.3f}\t{:.3f}\t{:.3f}\t".format(d_s, d_c, d_d))
        f.write("{}\n".format("\t".join(["{:.3f}".format(x) for x in prec_curv[:10]])))
    f.close()
    
    # dump data to cache
    f = open("{}experiment-{}.cache".format(configs.current_report_path, str(n_cluster)), "wb")
    data_cache = [graph_embeddings, reconstruction_performance, tsne_result, tsne_time]
    if KMEANS_EVAL:
        data_cache.append(kmeans_prediction)
        data_cache.append(kmeans_performance)
    if DBSCAN_EVAL:
        data_cache.append(dbscan_predcition)
        data_cache.append(dbscan_performance)
    pickle.dump(data_cache, f)
    f.close()


def get_parser():
    parser = argparse.ArgumentParser(description="Parser for Embedding Building and Clustering")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the json config file")
    parser.add_argument("--n_cluster", type=int, required=True, help="Number of clusters")
    parser.add_argument("--load_trained_embedding", type=bool, required=False, help="Whether load trained embeddings")
    parser.add_argument("--feature_file", type=str, required=True, help="Select feature file")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    configs = load_json(args.json_path)
    configs = dict2dotdict(configs)
    configs.current_embedding_path = f"{configs.EMBEDDING_PATH}{args.feature_file}/"
    configs.current_report_path = f"{configs.REPORT_PATH}{args.feature_file}/"
    configs.current_feature_file = f"./data/{configs.org}/{configs.dataset}_{args.feature_file}.features"
    main(configs, args.load_trained_embedding, args.n_cluster)