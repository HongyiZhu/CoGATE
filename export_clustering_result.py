
from graph_embedding_config        import *
from gva_utils                     import *
import pandas                      as pd
import pickle
import argparse


def main(configs):
    # load node_index
    # node_index: node_index[nodeID] = EmbeddingID
    temp = open(configs.node_index_filename, 'rb')
    nodeID2embeddingID = pickle.load(temp)
    embeddingID2nodeID = {nodeID2embeddingID[k]: k for k in nodeID2embeddingID.keys()}
    temp.close()

    # read nodeID & repoID/userID mapping
    # nodes: nodes[repoID/userID] = nodeID
    origID2nodeID = {}
    f = open(configs.node_file, 'r')
    for i, l in enumerate(f.readlines()):
        # repoID => nodeID mapping
        origID2nodeID[l.strip()] = str(i)
    nodeID2origID = {origID2nodeID[k]: k for k in origID2nodeID.keys()}
    f.close()

    # One reconstruction performance per dataset
    # Row: feature file
    # Column: Model
    print(f"Exporting {configs.REPORT_PATH}/MAP.xlsx")
    header = ["Feature"]
    header += configs.models
    df_map = pd.DataFrame(columns=header)
    for feature_file in configs.feature_files:
        n_cluster = configs.n_clusters_list[0]
        f = open("{}experiment-{}.cache".format(f"{configs.REPORT_PATH}/{feature_file}/", n_cluster), "rb")
        data_cache = pickle.load(f)
        reconstruction_performance = data_cache[1]
        df_map.loc[0 if pd.isnull(df_map.index.max()) else df_map.index.max() + 1] = [feature_file] + [reconstruction_performance[model][0] for model in configs.models]
    df_map.to_excel(f"{configs.REPORT_PATH}/MAP.xlsx", index=False, sheet_name="Mean Averaged Precision")

    if DBSCAN_EVAL:
        # One DBSCAN clustering result per feature file
        # Row: Node
        # Column: Model
        # Worksheet: feature_file
        writer = pd.ExcelWriter(f"{configs.REPORT_PATH}/DBSCAN_results.xlsx", engine="openpyxl")
        header = ["EmbeddingID", "NodeID", "OriginalID"]
        header += [f"{m}_DBSCAN" for m in configs.models]
        print("Exporting DBSCAN clustering results (labels)...")
        for feature_file in configs.feature_files:
            df_dbscan = pd.DataFrame(columns=header)
            n_cluster = configs.n_clusters_list[0]
            f = open("{}experiment-{}.cache".format(f"{configs.REPORT_PATH}/{feature_file}/", n_cluster), "rb")
            data_cache = pickle.load(f)
            if KMEANS_EVAL:
                dbscan_prediction = data_cache[6]
            else:
                dbscan_prediction = data_cache[4]
            f.close()

            for i in range(len(nodeID2embeddingID.keys())):
                df_dbscan.loc[0 if pd.isnull(df_dbscan.index.max()) else df_dbscan.index.max() + 1] = [
                    str(i), 
                    str(embeddingID2nodeID[i]), 
                    nodeID2origID[str(embeddingID2nodeID[i])]
                    ] + [str(dbscan_prediction[model][i]) for model in configs.models]

            df_dbscan.to_excel(writer, index=False, sheet_name=feature_file[:31])
            writer.save()
        writer.close()
        
        # One DBSCAN clustering result per feature file
        # Row: Node
        # Column: Model
        # Worksheet: feature_file

    if KMEANS_EVAL:
        # One KMeans clustering label results per feature file
        # WorkSheet: Feature_file
        # Row: Node
        # Column: Model x n_cluster
        print("Exporting KMeans clustering results (labels)...")
        writer = pd.ExcelWriter(f"{configs.REPORT_PATH}/KMeans_labels.xlsx", engine="openpyxl")
        for feature_file in configs.feature_files:
            header = ["NodeID", "OriginalID"]
            for n_cluster in configs.n_clusters_list:
                header += [f"{m}_{n_cluster}" for m in configs.models]
            df_kmeans_label = pd.DataFrame(columns=header, index=[str(i) for i in range(len(nodeID2embeddingID.keys()))])
            df_kmeans_label.index.name = "EmbeddingID"

            for n_cluster in configs.n_clusters_list:
                # load predictions
                f = open("{}experiment-{}.cache".format(f"{configs.REPORT_PATH}/{feature_file}/", n_cluster), "rb")
                data_cache = pickle.load(f)
                kmeans_prediction = data_cache[4]
                f.close()
                for i in range(len(nodeID2embeddingID.keys())):
                    df_kmeans_label.at[f"{str(i)}", "NodeID"] = str(embeddingID2nodeID[i])
                    df_kmeans_label.at[f"{str(i)}", "OriginalID"] = nodeID2origID[str(embeddingID2nodeID[i])]
                    for m in configs.models:
                        df_kmeans_label.at[f"{str(i)}", f"{m}_{n_cluster}"] =  kmeans_prediction[m][i]
            
            df_kmeans_label.to_excel(writer, index=True, sheet_name=feature_file[:31])
            writer.save()
        writer.close()

        # One KMeans clustering performance results per feature file
        # WorkSheet: Metrics
        # Row: Feature_file
        # Column: Model x n_cluster
        print("Exporting KMeans clustering results (performance)...")
        writer = pd.ExcelWriter(f"{configs.REPORT_PATH}/KMeans_performance.xlsx", engine="openpyxl")
        metrics = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
        dfs = {}
        for metric in metrics:
            header = []
            for model in configs.models:
                header += [f"{model}_{n_cluster}" for n_cluster in configs.n_clusters_list]
            dfs[metric] = pd.DataFrame(columns=header, index=[feature for feature in configs.feature_files])
            dfs[metric].index.name = "Feature"

        for feature_file in configs.feature_files:
            for n_cluster in configs.n_clusters_list:
                # load predictions
                f = open("{}experiment-{}.cache".format(f"{configs.REPORT_PATH}/{feature_file}/", n_cluster), "rb")
                data_cache = pickle.load(f)
                kmeans_performance = data_cache[5]
                f.close()
                for model in configs.models:
                    for i, metric in enumerate(metrics):
                        dfs[metric].at[feature_file, f"{model}_{n_cluster}"] =  kmeans_performance[model][i]
        for metric in metrics:    
            dfs[metric].to_excel(writer, index=True, sheet_name=metric)
            writer.save()
        writer.close()


def get_parser():
    parser = argparse.ArgumentParser(description="Parse args for export")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the json config file")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    configs = load_json(args.json_path)
    configs = dict2dotdict(configs)
    main(configs)
