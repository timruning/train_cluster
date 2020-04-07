# coding=UTF-8


import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm


class NMF_Net:
    def __init__(self, train_features, train_labels, batch_size, hidden_factor, layers, loss_type, features_M,
                 random_seed, batch_norm,
                 lamda_bilinear, optimizer_type, learning_rate, lambda_bilinear, keep_prob):

        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.loss_type = loss_type
        # self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        # self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.lambda_bilinear = lambda_bilinear
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        self.train_features = train_features  # None * features_M
        self.train_labels = train_labels  # None * 1

        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        # self.graph = tf.Graph()
        # with self.graph.as_default():  # , tf.device('/cpu:0'):
        # Set graph level random seed
        tf.set_random_seed(self.random_seed)
        # Input data.

        self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
        self.train_phase = tf.placeholder(tf.bool)
        # self.train_phase=tf.constant(True)
        # Variables.
        self.weights = self._initialize_weights()

        # Model.
        # _________ sum_square part _____________
        # get the summed up embeddings of features.
        nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
        self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)  # None * K
        # get the element-multiplication
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # _________ square_sum part _____________
        self.squared_features_emb = tf.square(nonzero_embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # ________ FM __________
        self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
        if self.batch_norm:
            self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
        self.FM = tf.nn.dropout(self.FM, self.dropout_keep[-1])  # dropout at the bilinear interactin layer

        # ________ Deep Layers __________
        for i in range(0, len(self.layers)):
            self.FM = tf.add(tf.matmul(self.FM, self.weights['layer_%d' % i]),
                             self.weights['bias_%d' % i])  # None * layer[i] * 1
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_%d' % i)  # None * layer[i] * 1
            self.FM = tf.nn.relu(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
        self.FM = tf.matmul(self.FM, self.weights['prediction'])  # None * 1

        # _________out _________
        Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
        self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),
                                          1)  # None * 1
        Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
        self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1

        # Compute the loss.
        if self.loss_type == 'square_loss':
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.l2_loss(
                    tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                    self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
        elif self.loss_type == 'log_loss':
            # self.out = tf.sigmoid(self.out)
            if self.lambda_bilinear > 0:
                # self.loss =tf.reduce_mean(-tf.reduce_sum(self.train_labels * tf.log(self.out), reduction_indices=[1]))
                # self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
                #                                        scope=None) + tf.contrib.layers.l2_regularizer(
                #     self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                # self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels)
                self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.train_labels, logits=self.out,
                                                            weights=1, label_smoothing=0)
            else:
                self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
                                                       scope=None)
        self.params = tf.trainable_variables()
        for v in self.params:
            tf.summary.histogram(name=v.name, values=v)

        # Optimizer.
        with tf.name_scope("gradients"):
            if self.optimizer_type == 'AdamOptimizer':
                self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                  epsilon=1e-8)

                self.params_ = tf.trainable_variables(scope='abcdfdf')

                self.grads = self.opt.compute_gradients(loss=self.loss, var_list=self.params_)
                print('----------------------', self.grads)
                self.optimizer = self.opt.minimize(self.loss)

                # self.optimizer = self.opt.apply_gradients(self.grads, global_step=self.global_step, name='optimizer')
            elif self.optimizer_type == 'AdagradOptimizer':
                self.opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                     initial_accumulator_value=1e-8)
                params = tf.trainable_variables(scope='abcdfdf')
                print('params shape = ', tf.shape(params))
                self.grads = self.opt.compute_gradients(loss=self.loss, var_list=params)
                print('----------------------', self.grads)
                self.optimizer = self.opt.minimize(self.loss)

            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

                params = tf.trainable_variables(scope='abcdfdf')

                self.grads = self.opt.compute_gradients(loss=self.loss, var_list=params)
                self.optimizer = self.opt.minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
                params = tf.trainable_variables(scope='abcdfdf')
                self.grads = self.opt.compute_gradients(loss=self.loss, var_list=params)
                self.optimizer = self.opt.minimize(
                    self.loss)
            elif self.optimizer_type == "lazyAdamOptimizer":
                self.opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate,
                                                            beta1=0.9,
                                                            beta2=0.999,
                                                            epsilon=1e-08,
                                                            use_locking=False,
                                                            name='Adam')
                self.optimizer = self.opt.minimize(self.loss)

        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        # if self.verbose > 0:
        #     print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        with tf.variable_scope("abcdfdf"):
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K
            print('tf.shape feature_embeddings ', all_weights['feature_embeddings'])
        all_weights['feature_bias'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 0.0),
                                                  name='feature_bias')  # features_M * 1
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        # deep layers
        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32,
                name='layer_0')
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                                dtype=np.float32, name='bias_0')  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32, name='layer_%d' % i)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32,
                    name='bias_%d' % i)  # 1 * layer[i]
                # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                                    dtype=np.float32, name='prediction')  # layers[-1] * 1

        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32), name='prediction')  # hidden_factor * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
