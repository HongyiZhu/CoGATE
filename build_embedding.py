from openne.models.line                import LINE
from openne.models.grarep              import GraRep
from openne.models.node2vec            import Node2vec
from openne.models.lle                 import LLE
from openne.models.lap                 import LaplacianEigenmaps
from openne.models.sdne                import SDNE
from openne.models.gf                  import GraphFactorization
from openne.models.hope                import HOPE
from openne.models.tadw                import TADW
from graph_embedding_config     import *
from build_gcae_embedding       import build_gcae
from build_vgae_embedding       import build_vgae
from build_gate_embedding       import build_gate
from build_cogate_embedding     import build_cogate, build_cogate_single
# from build_cane_embedding       import build_cane
# from build_dane_embedding       import build_dane
from load_graph_embedding       import load_embedding
from mygraph                    import Graph_Int, Graph_Str
from main_utils                 import load_json, dict2dotdict, dotdict2dict
import argparse
import pickle
import time


def build_le(g, path, configs):
    # Laplacian Eigenmaps OpenNE
    print("Lapacian Eigenmaps processing...")
    model_lap = LaplacianEigenmaps(g, rep_size=embedding_size)
    model_lap.save_embeddings("{}/Lap.nv".format(path))
    print("Laplacian Eigenmaps finished\n")
    embedding = load_embedding("{}/Lap.nv".format(path))
    return embedding

def build_gf(g, path, configs):
    # GF OpenNE
    print("GF processing...")
    model_gf = GraphFactorization(graph=g, rep_size=embedding_size)
    model_gf.save_embeddings("{}/GF.nv".format(path))
    print("GF finished\n")
    embedding = load_embedding("{}/GF.nv".format(path))
    return embedding

def build_lle(g, path, configs):
    # LLE OpenNE
    print("LLE processing...")
    model_lle = LLE(graph=g, d=embedding_size)
    model_lle.save_embeddings("{}/LLE.nv".format(path))
    print("LLE finished\n")
    embedding = load_embedding("{}/LLE.nv".format(path))
    return embedding

def build_hope(g, path, configs):
    # HOPE OpenNE
    print("HOPE processing...")
    model_hope = HOPE(graph=g, d=embedding_size)
    model_hope.save_embeddings("{}/HOPE.nv".format(path))
    print("HOPE finished\n")
    embedding = load_embedding("{}/HOPE.nv".format(path))
    return embedding

def build_grarep(g, path, configs):
    # GraRep OpenNE
    print("GraRep processing...")
    model_grarep = GraRep(graph=g, Kstep=kstep, dim=embedding_size)
    model_grarep.save_embeddings("{}/GraRep.nv".format(path))
    print("GraRep finished\n")
    embedding = load_embedding("{}/GraRep.nv".format(path))
    return embedding

def build_dw(g, path, configs):
    # DeepWalk OpenNE
    print("DeepWalk processing...")
    model_deepwalk = Node2vec(graph=g, path_length=walk_length, num_paths=number_walks, 
                    dim=embedding_size, window=window_size, workers=workers, dw=True)
    model_deepwalk.save_embeddings("{}/DeepWalk.nv".format(path))
    print("DeepWalk finished\n")
    embedding = load_embedding("{}/DeepWalk.nv".format(path))
    return embedding

def build_n2v(g, path, configs):
    # node2vec OpenNE
    print("Node2vec processing...")
    model_n2v = Node2vec(graph=g, path_length=walk_length, num_paths=number_walks, dim=embedding_size,
                        workers=workers, p=p, q=q, window=window_size)
    model_n2v.save_embeddings("{}/Node2vec.nv".format(path))
    print("Node2vec finished\n")
    embedding = load_embedding("{}/Node2vec.nv".format(path))
    return embedding

def build_sdne(g, path, configs):
    # SDNE OpenNE
    print("SDNE processing...")
    model_sdne = SDNE(g, encoder_layer_list=encoder_list, epoch=epochs)
    model_sdne.save_embeddings("{}/SDNE.nv".format(path))
    print("SDNE finished\n")
    embedding = load_embedding("{}/SDNE.nv".format(path))
    return embedding

def build_line(g, path, configs):
    # LINE OpenNE
    print("LINE processing...")
    model_line = LINE(g, epoch=epochs, rep_size=embedding_size)
    model_line.save_embeddings("{}/LINE.nv".format(path))
    print("LINE finished\n")
    embedding = load_embedding("{}/LINE.nv".format(path))
    return embedding

def build_tadw(g, path, configs):
    # TADW OpenNE
    print("TADW processing...")
    model_tadw = TADW(g, dim=embedding_size, lamb=lamb)
    model_tadw.save_embeddings("{}/TADW.nv".format(path))
    print("TADW finished\n")
    embedding = load_embedding("{}/TADW.nv".format(path))
    return embedding

def build_embedding(graph, graph_str, model, path, configs):
    build_functions = {
        'LE': build_le, 
        'GF': build_gf, 
        'LLE': build_lle, 
        'HOPE': build_hope, 
        'GRAREP': build_grarep,    
        'DEEPWALK': build_dw, 
        'NODE2VEC': build_n2v,                 
        'SDNE': build_sdne, 
        'VGAE': build_vgae, 
        'GATE': build_gate, 
        # 'CANE': build_cane,
        # 'DANE': build_dane,      
        'LINE': build_line,
        'GCAE': build_gcae,
        'TADW': build_tadw,
        'COGATE': build_cogate,
        'COGATE_SINGLE': build_cogate_single                               
    }
    func = build_functions.get(model)
    embedding = func(graph_str, path, configs) if model in ['DEEPWALK', "NODE2VEC"] else func(graph, path, configs)
    return embedding

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


def main(configs):
    process_node_index(configs.edgelist_filename, configs.node_index_filename, configs.embedding_mapping)
    temp = open(configs.node_index_filename, 'rb')
    node_index = pickle.load(temp)
    temp.close()

    # load dataset
    print("====================\nLoading edgelist")
    t1 = time.time()
    # load graph from edgelist and feature file
    graph = Graph_Int()
    graph.read_edgelist(filename=configs.edgelist_filename, node_index=node_index, weighted=configs.weighted_graph, directed=configs.directed_graph)
    graph_str = Graph_Str()
    graph_str.read_edgelist(filename=configs.edgelist_filename, node_index=node_index, weighted=configs.weighted_graph, directed=configs.directed_graph)
    if configs.have_features:
        graph.read_node_features(node_index=node_index, filename=configs.current_feature_file)
    print("Data Loaded. Time elapsed: {:.3f}\n====================\n".format(time.time() - t1))

    graph_embeddings = {}
    # build graph embedding
    print("====================\nBuilding Graph Embeddings\n")
    t2 = time.time()
    for model in configs.models:
        graph_embeddings[model] = build_embedding(graph, graph_str, model, configs.current_embedding_path, configs)
    print("Embeddings Constructed. Total time elapsed: {:.3f}\n====================".format(time.time() - t2))


def get_parser():
    parser = argparse.ArgumentParser(description="Parser for Embedding Building")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the json config file")
    parser.add_argument("--feature_file", type=str, required=False, help="Select feature file")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    configs = load_json(args.json_path)
    configs = dict2dotdict(configs)
    configs.current_embedding_path = f"{configs.EMBEDDING_PATH}{args.feature_file}/"
    configs.current_report_path = f"{configs.REPORT_PATH}{args.feature_file}/"
    if configs.have_features and not args.feature_file:
        print("Please specify a feature file to load")
        exit(0)
    if configs.have_features and args.feature_file:
        configs.current_feature_file = f"./data/{configs.dataset}/{configs.attackType}_{configs.attackPlatform}_{args.feature_file}.features"
    main(configs)