__author__ = 'guotata'
from Environment import Environment, AVAILABLE_ACTIONS
import multiprocessing
import threading
import zmq
import random
import signal

from msgpack_numpy import dumps as dump
from msgpack_numpy import loads as load
import uuid
import os
import time
from NetworkVP import NetworkVP
from collections import deque
from six.moves import queue
from tornado.concurrent import Future
from Config import Config

import numpy as np
LOG_FREQ = 5000
SAVE_FREQ = 500000
SUMMARY_BATCH_FREQ = 200

SIMULATOR_PROC = 80
PREDICTOR_THREAD = 2
BATCH_SIZE = 128
LOCAL_T_MAX = 5
GAMMA = 0.99

ctr_c = False #quit sign for all processes

def signal_handler(signum, frame):
    global ctr_c
    if signum == 2:
        time.sleep(2)

class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, value):
        self.state = state
        self.action = action
        self.reward = reward
        self.value = value # only useful for last experience

class MasterProcess(threading.Thread):
    def __init__(self, pipe_c2s, pipe_s2c):
        super(MasterProcess, self).__init__()
        self.context = zmq.Context()
        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.global_t = 0
        self.start_time = time.time()
        self.network = NetworkVP(device='/gpu:0', model_name='gagaga', num_actions=len(AVAILABLE_ACTIONS))
        if Config.LOAD_CHECKPOINT:
            self.global_t = self.network.load()
        self.send_queue = queue.Queue(maxsize=100)

        def f(frames):
            policies, values = self.network.predict_p_and_v(frames)

            rtn = np.zeros([policies.shape[0]], dtype = np.int32)
            for i in range(policies.shape[0]):
                rtn[i] = np.random.choice(range(len(AVAILABLE_ACTIONS)), p=policies[i])
            return rtn, values

        self.predictor = PredictorMaster(PREDICTOR_THREAD, f)

        class Sender(threading.Thread):
            def __init__(self, queue, socket):
                super(Sender, self).__init__()
                self.q = queue
                self.socket = socket
            def run(self):
                while True:
                    msg = self.q.get()
                    self.socket.send_multipart(msg, copy = False)

        self.send_thread = Sender(self.send_queue, self.s2c_socket)
        self.send_thread.start()

        self.client = {}
        self.training_queue = queue.Queue(maxsize=BATCH_SIZE*10)

        training_queue = self.training_queue
        network = self.network

        class TrainingThread(threading.Thread):
            def __init__(self):
                super(TrainingThread, self).__init__()
                self.t = 0

            def run(self):
                while True:
                    states, actions, Rs = [], [], []
                    state, action, R = training_queue.get()
                    states.append(state)
                    actions.append(action)
                    Rs.append(R)
                    self.t += 1
                    while len(states) < BATCH_SIZE:
                        try:
                            state, action, R = training_queue.get_nowait()
                            states.append(state)
                            actions.append(action)
                            Rs.append(R)
                        except queue.Empty:
                            break  # do not wait

                    network.train(states, Rs, actions)

                    if self.t % SUMMARY_BATCH_FREQ == 0:
                        network.log(states, Rs, actions)
                    
        trainingworker = TrainingThread()
        trainingworker.start()


    def run(self):
        print "Master pid is :%d\n" % os.getpid()
        self.predictor.start()

        while True:
            self.global_t += 1
            if self.global_t % LOG_FREQ == 0:
                t_diff = time.time() - self.start_time
                print("{} steps in {} seconds, {} steps/h".format(LOG_FREQ, t_diff, 3600*LOG_FREQ/t_diff))
                self.start_time = time.time()

            if self.global_t % SAVE_FREQ == 0:
                self.network.save(self.global_t)

            identity, frame, reward, isover = load(self.c2s_socket.recv(copy = False).bytes)
            if len(self.client[identity]) > 0:
                self.client[identity][-1].reward = reward

            #print 'frame received from {}'.format(identity)
            self._on_state(frame, identity)

            if isover:
                self._parse_memory(identity, 0, True)
            else:
                if len(self.client[identity]) == LOCAL_T_MAX + 1:
                    self._parse_memory(identity, self.client[identity][-1].value, False)

    def _on_state(self, frame, ident):
        def cb(output):
            action, value = output.result()
            #print 'action 2b sent is {} to {}'.format(action, ident)
            #self.s2c_socket.send_multipart([ident, dump(action)])
            self.send_queue.put([ident, dump(action)])
            self.client[ident].append(TransitionExperience(frame, action, None, value))

        self.predictor.put_task([frame], cb)

    #when local_t_max is reached or episode finished, cal reward and send to trainig thread
    def _parse_memory(self, ident, init_r, isOver):
        mem = self.client[ident]
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -5, 5) + GAMMA * R
            one_hot_action = [0] * len(AVAILABLE_ACTIONS)
            one_hot_action[k.action] = 1
            self.training_queue.put([k.state, one_hot_action, R])

        if not isOver:
            self.client[ident] = [last]
        else:
            self.client[ident] = []


class ClientProcess(multiprocessing.Process):
    def __init__(self, index, pipe_c2s, pipe_s2c):
        super(ClientProcess, self).__init__()
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c
        self.index = index

        self.name = u'simulator-{}'.format(index)
        self.identity = self.name.encode('utf-8')

    def run(self):

        print "My pid is :%d\n" % os.getpid()

        self.player = Environment(self.index * 113)
        context = zmq.Context()
        self.c2s_socket = context.socket(zmq.PUSH)
        self.c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        self.c2s_socket.connect(self.c2s)

        self.s2c_socket = context.socket(zmq.DEALER)
        self.s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        self.s2c_socket.connect(self.s2c)
        rew, isover = None, False

        while True:
            frame = self.player.current_state()
            self.c2s_socket.send(dump((self.identity, frame, rew, isover)), copy = False) #rew is last action's reward
            action = load(self.s2c_socket.recv(copy = False).bytes)
            #print '{} received {}'.format(self.identity, action)
            rew, isover = self.player.action(action)
            #print '{}: action advanced'.format(self.identity)

class PredictorMaster(threading.Thread):
    def __init__(self, predictor_num, f):
        super(PredictorMaster, self).__init__()
        self.input_queue = queue.Queue(maxsize = predictor_num * 100)
        self.threads = [PredictorThread(self.input_queue, k, f) for k in range(predictor_num)]

    def put_task(self, datapoint, callback):
        f = Future()
        f.add_done_callback(callback)
        self.input_queue.put((datapoint, f))
        return f

    def run(self):
        for i in self.threads:
            i.start()

class PredictorThread(threading.Thread):
    def __init__(self, inqueue, id, f):
        super(PredictorThread, self).__init__()
        self.input_queue = inqueue
        self.index = id
        self.predict_function = f

    def run(self):
        while True:
            data, futures = self.fetch_batch()
            actions, rew = self.predict_function(data[0]) #only 1 img is considered
            for idx, f in enumerate(futures):
                f.set_result((actions[idx], rew[idx]))


    def fetch_batch(self):
        #ensure return at least one img
        image, future = self.input_queue.get()
        nr_input_var = len(image)
        batched, futures = [[] for _ in range(nr_input_var)], []
        for k in range(nr_input_var):
            batched[k].append(image[k])
        futures.append(future)
        while len(futures) < BATCH_SIZE:
            try:
                inp, f = self.input_queue.get_nowait()
                for k in range(nr_input_var):
                    batched[k].append(image[k])
                futures.append(f)
            except queue.Empty:
                break   # do not wait
        return batched, futures  #batched: nr_input_var x ? x 120 x 120 x 24


if __name__ == '__main__':
    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [ClientProcess(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    master = MasterProcess(namec2s, names2c)
    master.start()

    for p in procs:
        master.client[p.identity] = [] #list of experience
        p.start()


    signal.signal(signal.SIGINT, signal_handler) # 2
    signal.pause()

    child_all_dead = False
    while not child_all_dead:
        child_all_dead = True
        for p in procs:
            if p.is_alive():
                child_all_dead = False
    os.system('kill -9 {}'.format(os.getpid()))
