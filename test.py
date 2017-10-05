__author__ = 'guotata'

from Environment_cig import Environment
import numpy as np

class Test:
    #runs the game for one episode, results are added in summary
    def __init__(self, network, env):
        self.game = env
        self.network = network

    def run_one_episode(self):
        total_rew = 0
        self.game.reset()

        while True:
            frame = self.game.current_state()
            nr_input_var = len(frame)
            batched = [[] for _ in range(nr_input_var)]
            for k in range(nr_input_var):
                batched[k].append(frame[k])
            values = self.network.predict_p(batched)
            
            values = values[0]
            action = np.where(values == max(values))
            rew, new_episode = self.game.action(action[0][0])
            total_rew += rew
            if new_episode:
                break
        return self.game.total_frag_count, total_rew
