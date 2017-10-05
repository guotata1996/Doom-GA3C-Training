from Config import Config
import tensorflow as tf
import numpy as np
import os
import re

class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels * 2 * 3], name='X') #image: ? x 120 x 120 x 24

        self.game_vars = tf.placeholder(tf.float32, [None, 2], name = 'Game_Variables')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr') #value

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # As implemented in A3C paper
        self.n1 = self.conv2d_layer(self.x, 7, 32, 'conv11', strides=[1, 2, 2, 1])
        self.n2 = self.conv2d_layer(self.n1, 7, 64, 'conv12', strides=[1, 2, 2, 1])
        self.n3 = self.maxpooling_layer(self.n2, 3, strides=[1, 2, 2, 1])
        self.n4 = self.conv2d_layer(self.n3, 3, 128, 'conv13', strides=[1, 2, 2, 1])
        self.n5 = self.maxpooling_layer(self.n4, 3, strides=[1, 2, 2, 1])

        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        _input = self.n5

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1 = self.dense_layer(self.flat, 128, 'dense1')
        self.d2 = self.dense_layer(self.game_vars, 8, 'dense2')
        self.d = tf.concat([self.d1, self.d2], 1)

        self.logits_v = tf.squeeze(self.dense_layer(self.d, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_p = self.dense_layer(self.d, self.num_actions, 'logits_p', func=None)
        if Config.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

            self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                        * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                      self.softmax_p, axis=1)
        
        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)

        self.cost_all = self.cost_p + self.cost_v
        self.opt = tf.train.AdamOptimizer(
                learning_rate=self.var_learning_rate,
                epsilon=Config.Adam_EPSILON)

        if Config.USE_GRAD_CLIP:
            self.opt_grad = self.opt.compute_gradients(self.cost_all)
            self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
            self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)

        self.test_rewards = tf.placeholder(tf.float32, shape=[])
        self.test_frags = tf.placeholder(tf.float32, shape = [])


    def _create_tensor_board(self):
        param_summaries = []
        param_summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        param_summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        param_summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        param_summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        param_summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        param_summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            param_summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        param_summaries.append(tf.summary.histogram("activation_n1", self.n1))
        param_summaries.append(tf.summary.histogram("activation_n2", self.n2))
        param_summaries.append(tf.summary.histogram("activation_d2", self.d))
        param_summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        param_summaries.append(tf.summary.histogram("activation_p", self.softmax_p))
        self.param_summary_op = tf.summary.merge(param_summaries)

        eval_summaries = []
        eval_summaries.append(tf.summary.scalar("test_rewards", self.test_rewards))
        eval_summaries.append(tf.summary.scalar("test_frags", self.test_frags))
        self.eval_summary_op = tf.summary.merge(eval_summaries)

        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='VALID') + b
            if func is not None:
                output = func(output)

        return output

    def maxpooling_layer(self, input, kern_size, strides, func = tf.nn.max_pool):
        return func(input, [1, kern_size, kern_size, 1], strides, padding="SAME")

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_v(self, x):
        #print np.asarray(x[1:]).shape
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x[0], self.game_vars: np.transpose(np.asarray(x[1:]))})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x[0], self.game_vars: np.transpose(np.asarray(x[1:]))})
        return prediction
    
    def predict_p_and_v(self, x):
        
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x[0], self.game_vars: np.transpose(np.asarray(x[1:]))})
    
    def train(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x[0], self.game_vars: np.transpose(np.asarray(x[1:])), self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log_param(self, x, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x[0], self.game_vars: np.transpose(np.asarray(x[1:])), self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.param_summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)
        self.log_writer.flush()

    def log_eval(self, frag, rew):
        step, summary = self.sess.run([self.global_step, self.eval_summary_op], feed_dict={self.test_rewards:rew, self.test_frags:frag})
        self.log_writer.add_summary(summary, step)
        self.log_writer.flush()

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
