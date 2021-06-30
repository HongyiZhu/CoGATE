from load_graph_embedding           import load_embedding
from graph_embedding_config         import *
from cogate_model                   import CoGATE
from cogate_trainer                 import Trainer
import cogate_utils

def build_cogate(g, embedding_path, configs):
    print("CoGATE processing...")
    G, X = cogate_utils.load_data(configs)
    feature_dim = X.shape[1]
    cogate_args['hidden_dims'] = [feature_dim] + cogate_args['hidden_dims']

    G_tf,  S, R = cogate_utils.prepare_graph_data(G, configs)

    trainer = Trainer(cogate_args)
    trainer(G_tf, X, S, R)
    embeddings, attentions = trainer.infer(G_tf, X, S, R)
    f = open("{}/CoGATE.nv".format(embedding_path), "w")
    f.write(" ".join([str(x) for x in embeddings.shape]))
    f.write("\n")
    for i in range(embeddings.shape[0]):
        d = " ".join([str(x) for x in embeddings[i]])
        f.write("{} {}\n".format(str(i), d))
    f.close()
    print("CoGATE finished\n")
    embedding = load_embedding("{}/CoGATE.nv".format(embedding_path))

    return embedding