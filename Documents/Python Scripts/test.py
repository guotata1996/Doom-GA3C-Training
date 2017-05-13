from Environment import Environment, AVAILABLE_ACTIONS
from NetworkVP import NetworkVP
import numpy as np
import time

DURATION = 200

game = Environment(100, display=True)
network = NetworkVP(device='/gpu:0', model_name='gagaga', num_actions=len(AVAILABLE_ACTIONS))
network.load()

for _ in range(DURATION):
    frame = game.current_state()
    policy = network.predict_p([frame])[0]
    action = np.where(policy == max(policy))[0][0]
    game.action(action)
    time.sleep(0.1)