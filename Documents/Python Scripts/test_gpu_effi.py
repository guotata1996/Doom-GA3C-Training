__author__ = 'guotata'
import numpy as np
from network import GameACFFNetwork
import tensorflow as tf
import time

BATCH_SIZE = 256
x = np.random.random([BATCH_SIZE, 120, 120, 24])

TEST_ROUND = 50
sess = tf.Session()
network = GameACFFNetwork(action_size=3)
sess.run(tf.global_variables_initializer())
start_time = time.time()
for i in range(TEST_ROUND):
    print(i)
    network.run_policy_and_value(sess, x)

time_diff = time.time() - start_time
print("{} steps in {} seconds, {} steps/hour".format(BATCH_SIZE * TEST_ROUND, time_diff, BATCH_SIZE * TEST_ROUND * 3600 / time_diff))