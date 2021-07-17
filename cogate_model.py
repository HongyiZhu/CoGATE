import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy                as np

class CoGATE():
    def __init__(self, hidden_dims, lambda_):
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) -1
        self.W, self.v, self.W2 = self.define_weights(hidden_dims)
        self.C = {}

    def __call__(self, A, F, X, S, R):
        # Encoder
        Fp = tf.sparse.reorder(tf.identity(F))
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, Fp, H, layer)
        # Final node representations
        self.H = H
        H1 = H2 = H
        # Decoder 1
        for layer in range(self.n_layers - 1, -1, -1):
            H1 = self.__decoder(H1, layer)
        X_ = H1

        # The reconstruction loss of node features
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        for layer in range(self.n_layers -1 , -1, -1):
            H2 = self.__reconstruct(H2, layer)
        F_ = H2

        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        structure_loss = tf.reduce_sum(structure_loss)

        # Degree matrix reconstruction
        self.S_deg_emb = tf.nn.embedding_lookup(F_, S)
        self.R_deg_emb = tf.nn.embedding_lookup(F_, R)
        deg_sim = tf.reduce_sum(self.S_deg_emb * self.R_deg_emb, axis=-1)
        # indice = np.vstack((S, R)).transpose()
        temp = tf.nn.relu(deg_sim)
        temp = tf.SparseTensor(Fp.indices, temp, A.dense_shape)
        temp = tf.sparse_add(temp, tf.negative(Fp))
        degree_loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sparse.to_dense(temp), 2)))

        # Total loss
        self.loss = features_loss + self.lambda_ * structure_loss + degree_loss

        return self.loss, self.H, self.C

    def __encoder(self, A, F, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, F, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __reconstruct(self, H, layer):
        H = tf.matmul(H, self.W2[layer])
        return H

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        W2 = {}
        for i in range(self.n_layers):
            W2[i] = tf.get_variable("W2%s" % i, shape=(hidden_dims[-1], hidden_dims[-1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))
            Ws_att[i] = v

        return W, Ws_att, W2

    def graph_attention_layer(self, A, F, M, v, layer):
        with tf.variable_scope("layer_%s"% layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            logits = tf.sparse.reorder(logits)

            unnormalized_attentions = tf.sparse.from_dense(tf.multiply(tf.nn.sigmoid(tf.sparse.to_dense(logits)), tf.sparse.to_dense(F)))

            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions


class CoGATE_single():
    def __init__(self, hidden_dims, lambda_):
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) -1
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}

    def __call__(self, A, F, X, S, R):
        # Encoder
        Fp = tf.sparse.reorder(tf.identity(F))
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, Fp, H, layer)
        # Final node representations
        self.H = H

        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H
        # The reconstruction loss of node features
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        node_sim = tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)
        structure_loss = -tf.log(tf.sigmoid(node_sim))
        structure_loss = tf.reduce_sum(structure_loss)
        S_ = tf.convert_to_tensor(S)
        R_ = tf.convert_to_tensor(R)
        # Degree matrix reconstruction
        indice = tf.transpose(tf.stack([S, R]))
        temp = tf.nn.relu(node_sim)
        temp2 = tf.SparseTensor(indice, temp, A.dense_shape)
        temp2 = tf.sparse.reorder(temp2)
        temp2 = tf.sparse_add(temp2, tf.negative(Fp))
        degree_loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sparse.to_dense(temp2), 2)))

        # Total loss
        self.loss = features_loss + self.lambda_ * structure_loss + degree_loss

        return self.loss, self.H, self.C

    def __encoder(self, A, F, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, F, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def graph_attention_layer(self, A, F, M, v, layer):
        with tf.variable_scope("layer_%s"% layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            logits = tf.sparse.reorder(logits)

            unnormalized_attentions = tf.sparse.from_dense(tf.multiply(tf.nn.sigmoid(tf.sparse.to_dense(logits)), tf.sparse.to_dense(F)))

            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions