from Environment_cig import Environment, AVAILABLE_ACTIONS
from NetworkVP import NetworkVP
import numpy as np
import time

DURATION = 5000

game = Environment(100, display=True)
network = NetworkVP(device='/gpu:0', model_name='gagaga', num_actions=len(AVAILABLE_ACTIONS))
network.load()

for _ in range(DURATION):
    frame = game.current_state()
    nr_input_var = len(frame)
    batched = [[] for _ in range(nr_input_var)]
    for k in range(nr_input_var):
        batched[k].append(frame[k])
    policy = network.predict_p(batched)[0]
    action = np.where(policy == max(policy))[0][0]
    game.action(action)
    time.sleep(0.15)
