import tensorflow.compat.v1 as tf
import cogate_utils
from cogate_model import CoGATE, CoGATE_single


class Trainer():
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        cogate = CoGATE(self.args['hidden_dims'], self.args['lambda_'])
        self.loss, self.H, self.C = cogate(self.A, self.F, self.X, self.S, self.R)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.F = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args['lr'])
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args['gradient_clipping'])
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))


    def __call__(self, A, F, X, S, R):
        for epoch in range(self.args['n_epochs']):
            self.run_epoch(epoch, A, F, X, S, R)


    def run_epoch(self, epoch, A, F, X, S, R):
        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.F: F,
                                                    self.X: X,
                                                    self.S: S,
                                                    self.R: R})
        #print("Epoch: %s, Loss: %.2f" % (epoch, loss))
        return loss

    def infer(self, A, F, X, S, R):
        H, C = self.session.run([self.H, self.C],
                           feed_dict={self.A: A,
                                      self.F: F,
                                      self.X: X,
                                      self.S: S,
                                      self.R: R})


        return H, cogate_utils.conver_sparse_tf2np(C)


class Trainer_single():
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        cogate = CoGATE_single(self.args['hidden_dims'], self.args['lambda_'])
        self.loss, self.H, self.C = cogate(self.A, self.F, self.X, self.S, self.R)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.F = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args['lr'])
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args['gradient_clipping'])
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))


    def __call__(self, A, F, X, S, R):
        for epoch in range(self.args['n_epochs']):
            self.run_epoch(epoch, A, F, X, S, R)


    def run_epoch(self, epoch, A, F, X, S, R):

        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.F: F,
                                                    self.X: X,
                                                    self.S: S,
                                                    self.R: R})

        #print("Epoch: %s, Loss: %.2f" % (epoch, loss))
        return loss

    def infer(self, A, F, X, S, R):
        H, C = self.session.run([self.H, self.C],
                           feed_dict={self.A: A,
                                      self.F: F,
                                      self.X: X,
                                      self.S: S,
                                      self.R: R})


        return H, cogate_utils.conver_sparse_tf2np(C)