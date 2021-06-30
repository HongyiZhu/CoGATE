embedding_size = 128            # must be smaller than the number of observations in LLE
epochs = 40                     # for LINE
kstep = 4                       # for GraRep
window_size = 10                # for DeepWalk and Node2vec
walk_length = 80                # for DeepWalk and Node2vec
number_walks = 10               # for DeepWalk and Node2vec
workers = 8                     # for DeepWalk and Node2vec
p = 1                           # for Node2vec
q = 1                           # for Node2vec
encoder_list = [512, 128]       # for SDNE
lamb = 0.2                      # for TADW
KMEANS_EVAL = True              # Use KMeans for clustering
DBSCAN_EVAL = False             # Use DBSCAN for clustering
eps = 0.01                      # for DBSCAN

gate_args = {
    'lr': 0.0001,
    'n_epochs': epochs,
    'hidden_dims': [256, embedding_size], 
    'lambda_': 1.0, 
    'dropout': 0.0,
    'gradient_clipping': 5.0,
}

cogate_args = {
    'lr': 0.0001,
    'n_epochs': epochs,
    'hidden_dims': [256, embedding_size], 
    'lambda_': 1.0, 
    'dropout': 0.0,
    'gradient_clipping': 5.0,
}

cane_args = {
    'ratio': 0.55,
    'MAX_LEN': 300,
    'neg_table_size': 1000000,
    'NEG_SAMPLE_POWER': 0.75,
    'batch_size': 64,
    'num_epoch': 200,
    'embed_size': 200,
    'lr': 1e-3
}
