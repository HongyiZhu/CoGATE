from gva_utils      import dotdict2dict, dict2dotdict
import argparse
import subprocess
import json
import os


def get_parser():
    parser = argparse.ArgumentParser(description='Automated GVA Processor.')
    parser.add_argument("--org", type=str, required=True, help='Organization of the analysis.')
    parser.add_argument("--dataset", type=str, required=True, choices=['user', 'repo'], help="Process 'user' or 'repo' dataset.")
    parser.add_argument("--n_clusters", type=str, required=True, help="Comma delimited list input (e.g., 2,3,4,5,6), default=2.", default="2,")
    parser.add_argument("--have_features", type=bool, required=False, help="Whether the network has nodal features, default=True.", default=True)
    parser.add_argument("--weighted_graph", type=bool, required=False, help="Whether the edges are weighted, default=True.", default=True)
    parser.add_argument("--models", type=str, required=False, help="Comma delimited model names (e.g., TADW,GCAE,GATE), default=TADW,GCAE,GATE", default="TADW,GCAE,GATE.")
    parser.add_argument("--commit_edgelist", type=bool, help="Use edgelist constructed with commit info, default=False.", required=False, default=False)
    
    parser.add_argument("--step", type=str, required=False, help="Perform a particular step ([P]reprocess, [B]uild embedding, [E]xport results, Plot [T]SNE) or [A]ll steps), default=A.", choices=["P", "B", "E", "T", "A"], default="A")

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # read arguments
    org = args.org
    dataset = args.dataset

    # Embedding models
    # models = [
        # 'LE', 
        # 'GF', 
        # 'LLE', 
        # 'HOPE', 
        # 'GraRep',    
        # 'DeepWalk', 
        # 'node2vec',                 
        # 'SDNE',                         
        # 'LINE',                                  
        # 'GCAE',
        # 'TADW',
        # 'VGAE',
        # 'DANE',
        # 'CANE',
        # 'GATE',
    # ]
    # models = [model.upper() for model in models]

    # compile environment config file   
    configs = dict2dotdict(None)
    configs.org = org
    configs.dataset = dataset
    configs.have_features = args.have_features
    configs.weighted_graph = args.weighted_graph
    configs.models = [x.upper() for x in args.models.split(',') if x != ""]
    configs.n_clusters_list = [int(x) for x in args.n_clusters.split(',') if x != ""]

    # paths
    configs.node_file = f"./data/{org}/{dataset}_nodes.csv"
    configs.edgelist_filename = f"./data/{org}/{dataset}.edgelist" if not args.commit_edgelist else f"./data/{org}/{dataset}commits.edgelist"
    configs.node_index_filename = f"./data/{org}/{dataset}.index"
    configs.embedding_mapping = f"./data/{org}/{dataset}_mapping.csv"
    configs.node_file = f"./data/{org}/{dataset}_nodes.csv"

    configs.EMBEDDING_PATH = f"./embeddings/{org}/{dataset}/" if not args.commit_edgelist else f"./embeddings/{org}/{dataset}-commits/"
    configs.REPORT_PATH = f"./results/{org}/{dataset}/" if not args.commit_edgelist else f"./results/{org}/{dataset}-commits/"
    configs.FEATURE_PATH = f"./data/{org}/feature_matrices/" if configs.have_features else None
    
    configs.TSNE_PATH = f"{configs.REPORT_PATH}tsne/"
    if not os.path.exists(f"{configs.TSNE_PATH}"):
        os.makedirs(f"{configs.TSNE_PATH}")

    configs.feature_files = [filename.split(".")[0] for filename in os.listdir(configs.FEATURE_PATH) if filename.startswith(configs.dataset)]

    json_configs = dotdict2dict(configs)

    json_path = f"./data/{org}/{dataset}_config.json"
    with open(json_path, 'w') as fp:
        json.dump(json_configs, fp)

    if args.step == "P" or args.step == "A":
        # Run preprocess_data.py
        _preprocess = subprocess.run(["python", "preprocess_data.py", "--json_path", f"{json_path}"])

    if args.step == "B" or args.step == "A":
        # For each feature matrix, generate node embeddings and cluster them
        for feature_file in configs.feature_files:
            # Create output directories
            if not os.path.exists(f"{configs.EMBEDDING_PATH}{feature_file}"):
                os.makedirs(f"{configs.EMBEDDING_PATH}{feature_file}/")
            if not os.path.exists(f"{configs.REPORT_PATH}{feature_file}"):
                os.makedirs(f"{configs.REPORT_PATH}{feature_file}/")

            for i, n_cluster in enumerate(configs.n_clusters_list):
                # The first clustering will follow embedding building
                if i == 0:
                    _evaluate = subprocess.run(["python", "evaluation.py", "--json_path", f"{json_path}", "--n_cluster", f"{n_cluster}", "--feature_file", f"{feature_file}"])
                # Other clustering tasks will load constructed embeddings
                else:
                    _evaluate = subprocess.run(["python", "evaluation.py", "--json_path", f"{json_path}", "--n_cluster", f"{n_cluster}", "--feature_file", f"{feature_file}", "--load_trained_embedding", "True"])
    if args.step == "E" or args.step == "A":
        # Compile reports for all models in each dataset
        _export = subprocess.run(["python", "export_clustering_result.py", "--json_path", f"{json_path}"])
    
    if args.step == "T" or args.step == "A":
        # Save T-SNE plots for all models in each dataset
        _plot = subprocess.run(["python", "plot_tsne.py", "--json_path", f"{json_path}"])
    