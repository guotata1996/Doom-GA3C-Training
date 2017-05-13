__author__ = 'guotata'
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import cPickle
import os
from Config import Config

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               device="/gpu:0"):
    self._action_size = action_size
    self._device = device

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()

  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  # conv_w x conv_h x input_channels x output_channels
  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

  def save_model(self, sess, global_t, wall_t):
    save_file_name = 'checkpoints/' + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())) + '.save'
    save_file = open(save_file_name, "w")
    cPickle.dump(global_t, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(wall_t, save_file, protocol=cPickle.HIGHEST_PROTOCOL)

    for var in self.get_vars():
      cPickle.dump(sess.run(var), save_file, protocol=cPickle.HIGHEST_PROTOCOL)
    save_file.close()

  # returns: global_t of this checkpoint
  def load_model(self, sess):
    save_time = []
    for dirname in os.listdir('checkpoints/'):
      checkpoint_time = dirname.split('.')[0]
      timeArray = time.strptime(checkpoint_time, "%Y-%m-%d_%H:%M:%S")
      timeStamp = int(time.mktime(timeArray))
      save_time.append(timeStamp)

    if len(save_time) == 0:
      return None, None
    save_time.sort()
    last_file_name = 'checkpoints/' + time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(save_time[-1])) + '.save'
    save_file = open(last_file_name, "r")
    global_t = cPickle.load(save_file)
    wall_t = cPickle.load(save_file)
    for var in self.get_vars():
      sess.run(tf.assign(var, cPickle.load(save_file)))
    save_file.close()
    return global_t, wall_t

  def get_vars(self):
    pass

# Actor-Critic FF Network
# action-size when init has been powered by 2
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               device="/gpu:0"):
    GameACNetwork.__init__(self, action_size, device)

    with tf.device(self._device):
      self.W_conv1, self.b_conv1 = self._conv_variable([7, 7, 24, 32])
      self.W_conv2, self.b_conv2 = self._conv_variable([7, 7, 32, 64])
      self.W_conv3, self.b_conv3 = self._conv_variable([3, 3, 64, 128])

      self.W_fc1, self.b_fc1 = self._fc_variable([3*3*128, 128]) # extracted features
      self.W_fc11, self.b_fc11 = self._fc_variable([8, 128]) # ammo and health

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([128, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([128, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 120, 120, 24])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 2) + self.b_conv1) # stride=2 in-120x120 out-57x57
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2) # stride=2 in-57x57 out-26x26
      h_maxpool1 = tf.nn.max_pool(h_conv2,[1, 3, 3, 1],[1, 2, 2, 1], 'SAME') # stride=2 in-26x26 out-13x13
      h_conv3 = tf.nn.relu(self._conv2d(h_maxpool1, self.W_conv3, 2) + self.b_conv3) # stride=2  in-13x13 out-6x6
      h_maxpool2 = tf.nn.max_pool(h_conv3,[1, 3, 3, 1],[1, 2, 2, 1], 'SAME') # stride=2 in-6x6 out-3x3

      h_flat = tf.reshape(h_maxpool2, [-1, 3*3*128])
      h_fc1 = tf.nn.relu((tf.matmul(h_flat, self.W_fc1) + self.b_fc1))

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output)
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1])

    self.prepare_loss(0.0)

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
      # R (input for value)
      self.r = tf.placeholder("float", [None])

      # temporary difference (R-V) (input for policy)
      self.td = tf.subtract(self.r, self.v)

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

      # policy entropy
      entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t): #s_t : None x 120 x 120 x 24
    pi_out, v_out = sess.run([self.pi, self.v], feed_dict = {self.s:s_t} )
    return pi_out, v_out

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : s_t} )
    return pi_out

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : s_t})
    return v_out

  def train(self, sess, a, r, s):
    #input: list of action, ..
    actions = np.asarray(a)
    rewards = np.asarray(r)
    states = np.asarray(s)
    print 'in train'
    with tf.device(self._device):
      for v in tf.trainable_variables():
        print v.name
      sess.run(tf.train.AdamOptimizer().minimize(self.total_loss), feed_dict={self.a: actions, self.r:rewards, self.s: states})

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,

            self.W_fc1, self.b_fc1,

            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]