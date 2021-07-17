from main_utils      import dotdict2dict, dict2dotdict
import argparse
import subprocess
import json
import os


def get_parser():
    parser = argparse.ArgumentParser(description='Automated CoGATE Processor.')
    parser.add_argument("--dataset", type=str, required=True, help="Choose dataset (title and description or title only", default="title only", choices=["title and description", "title only"])
    parser.add_argument("--attackPlatform", type=str, required=True, help="Choose attack platforms", default="both", choices=["linux", "windows", "both"])
    parser.add_argument("--attackType", type=str, required=True, help="Choose attack types", default="both", choices=["local", "remote", "both"])
    parser.add_argument("--have_features", type=bool, required=False, help="Whether the network has nodal features, default=True.", default=True)
    parser.add_argument("--weighted_graph", type=bool, required=False, help="Whether the edges are weighted, default=False.", default=False)
    parser.add_argument("--directed_graph", type=bool, required=False, help="Whether the edges are directed, default=True.", default=True)
    parser.add_argument("--standard_features", type=bool, required=False, help="Generate fastText and one-hot features for each word", default=False)
    parser.add_argument("--fastText_dim", type=int, required=False, help="fastText word embedding dimension", default=300)
    parser.add_argument("--models", type=str, required=False, help="Comma delimited model names (e.g., TADW,GCAE,GATE), default=TADW,GCAE,GATE", default="TADW,GCAE,GATE.")
    
    parser.add_argument("--step", type=str, required=False, help="Perform a particular step ([I]nitialize folders and/or feature files, [P]reprocess feature files, [B]uild embedding, [E]xport results) or [A]ll steps (P, B, & E)), default=A.", choices=["I", "P", "B", "E", "A"], default="A")

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # read arguments
    dataset = args.dataset
    attackPlatform = args.attackPlatform
    attackType = args.attackType

    # Embedding models
    models = [
        'LE', 
        'GF', 
        'LLE', 
        'HOPE', 
        'GraRep',    
        'DeepWalk', 
        'node2vec',                 
        'SDNE',                         
        'LINE',                                  
        'GCAE',
        'TADW',
        'VGAE',
        # 'DANE',
        # 'CANE',
        'GATE',
        # 'COGATE',
        # 'COGATE_SINGLE'
    ]
    models = [model.upper() for model in models]

    # compile environment config file   
    configs = dict2dotdict(None)
    configs.dataset = dataset
    configs.attackType = attackType
    configs.attackPlatform = attackPlatform
    configs.have_features = args.have_features
    configs.weighted_graph = args.weighted_graph
    configs.directed_graph = args.directed_graph
    configs.models = [x.upper() for x in args.models.split(',') if x != ""]

    # paths
    configs.node_file = f"./data/{dataset}/{attackType}_{attackPlatform}_fullwordlist.pkl"
    configs.edgelist_filename = f"./data/{dataset}/{attackType}_{attackPlatform}.edgelist"
    configs.node_index_filename = f"./data/{dataset}/{attackType}_{attackPlatform}.index"
    configs.embedding_mapping = f"./data/{dataset}/{attackType}_{attackPlatform}_mapping.csv"

    configs.EMBEDDING_PATH = f"./embeddings/{dataset}/{attackType}_{attackPlatform}/"
    configs.REPORT_PATH = f"./results/{dataset}/{attackType}_{attackPlatform}/"
    configs.FEATURE_PATH = f"./data/{dataset}/feature_matrices/{attackType}_{attackPlatform}/" if configs.have_features else None

    json_configs = dotdict2dict(configs)
    json_path = f"./data/{dataset}/{attackType}_{attackPlatform}_config.json"
    with open(json_path, 'w') as fp:
        json.dump(json_configs, fp)


    if args.step == "I":
        # Initialize folders
        if not os.path.exists(f"{configs.REPORT_PATH}"):
            os.makedirs(f"{configs.REPORT_PATH}")
        if not os.path.exists(f"{configs.EMBEDDING_PATH}"):
            os.makedirs(f"{configs.EMBEDDING_PATH}")
        if configs.have_features and not os.path.exists(f"{configs.FEATURE_PATH}"):
            os.makedirs(f"{configs.FEATURE_PATH}")
        
        # Generate Standard Features
        # if args.standard_features:
            # _preprocess = subprocess.run(["python", "generate_fasttext_word_feature.py", "--json_path", f"{json_path}", "--fastText_dim", f"{args.fastText_dim}"])

        print("Folder and feature initialization complete.")
        exit(0)

    if args.step == "P" or args.step == "A":
        # Generate .features files from .csv files            
        if configs.have_features:
            _preprocess = subprocess.run(["python", "preprocess_word_feature.py", "--json_path", f"{json_path}"])

    if args.step == "B" or args.step == "A":
        # For each feature matrix, generate node embeddings
        if not configs.have_features:
            _evaluate = subprocess.run(["python", "build_embedding.py", "--json_path", f"{json_path}"])
        else:
            feature_files = [filename.split(".")[0] for filename in os.listdir(configs.FEATURE_PATH)]
            for feature_file in feature_files:
                # Create output directories
                if not os.path.exists(f"{configs.EMBEDDING_PATH}{feature_file}"):
                    os.makedirs(f"{configs.EMBEDDING_PATH}{feature_file}/")
                if not os.path.exists(f"{configs.REPORT_PATH}{feature_file}"):
                    os.makedirs(f"{configs.REPORT_PATH}{feature_file}/")

                _evaluate = subprocess.run(["python", "build_embedding.py", "--json_path", f"{json_path}", "--feature_file", f"{feature_file}"])
                    
    if args.step == "E" or args.step == "A":
        _evaluate = subprocess.run(["python", "evaluation.py", "--json_path", f"{json_path}", "--feature_file", f"{feature_file}", "--load_trained_embedding", "True"])

        # Compile reports for all models in each dataset
        _export = subprocess.run(["python", "export_clustering_result.py", "--json_path", f"{json_path}"])
    
    # if args.step == "T" or args.step == "A":
    #     # Save T-SNE plots for all models in each dataset
    #     _plot = subprocess.run(["python", "plot_tsne.py", "--json_path", f"{json_path}"])
    