from load_graph_embedding           import load_embedding
from graph_embedding_config         import *
from gate_model                     import GATE
from gate_trainer                   import Trainer
import gate_utils

def build_gate(g, embedding_path, configs):
    print("GATE processing...")
    G, X = gate_utils.load_data(configs)
    feature_dim = X.shape[1]
    gate_args['hidden_dims'] = [feature_dim] + gate_args['hidden_dims']

    G_tf,  S, R = gate_utils.prepare_graph_data(G, configs)

    trainer = Trainer(gate_args)
    trainer(G_tf, X, S, R)
    embeddings, attentions = trainer.infer(G_tf, X, S, R)
    f = open("{}/GATE.nv".format(embedding_path), "w")
    f.write(" ".join([str(x) for x in embeddings.shape]))
    f.write("\n")
    for i in range(embeddings.shape[0]):
        d = " ".join([str(x) for x in embeddings[i]])
        f.write("{} {}\n".format(str(i), d))
    f.close()
    print("GATE finished\n")
    embedding = load_embedding("{}/GATE.nv".format(embedding_path))

    return embedding