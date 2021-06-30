from time                   import time
from graph_embedding_config import *
from gva_utils              import *
import numpy                as np
import matplotlib.pyplot    as plt
import pickle
import argparse

def plot_embedding(data, label, title, path):
    x_min, x_max = np.min(data), np.max(data)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(i),
                 color=plt.cm.Set1(label[i] / float(np.max(label))),
                 fontdict={'weight': 'regular', 'size': 6})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(path, format='svg')
    plt.close()

def main(configs):
    for feature_file in configs.feature_files:
        for n_cluster in configs.n_clusters_list:
            f = open("{}experiment-{}.cache".format(f"{configs.REPORT_PATH}{feature_file}/", str(n_cluster)), "rb")
            result = pickle.load(f)
            f.close()
            tsne_result = result[2]
            tsne_time = result[3]
            for model in configs.models:
                if KMEANS_EVAL:
                    prediction = result[4]
                    path = "{}[{}]{}_KMeans_{}.svg".format(configs.TSNE_PATH, model, feature_file, str(n_cluster))
                    plot_embedding(tsne_result[model], prediction[model],
                                't-SNE visualization of KMeans Clustering of\n{} Embeddings (time {:.3f}s)'.format(model, tsne_time[model]),
                                path)
                if DBSCAN_EVAL and KMEANS_EVAL:
                    prediction = result[6]
                    path = "{}[{}]{}_DBSCAN.svg".format(configs.TSNE_PATH, model, feature_file)
                    plot_embedding(tsne_result[model], prediction[model],
                                't-SNE visualization of DBSCAN Clustering of\n{} Embeddings (time {:.3f}s)'.format(model, tsne_time[model]),
                                path)
                elif DBSCAN_EVAL and not KMEANS_EVAL:
                    prediction = result[4]
                    path = "{}[{}]{}_DBSCAN.svg".format(configs.TSNE_PATH, model, feature_file)
                    plot_embedding(tsne_result[model], prediction[model],
                                't-SNE visualization of DBSCAN Clustering of {} Embeddings (time {:.3f}s)'.format(model, tsne_time[model]),
                                path)


def get_parser():
    parser = argparse.ArgumentParser(description="Parse args for export")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the json config file")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    configs = load_json(args.json_path)
    configs = dict2dotdict(configs)
    
    main(configs)