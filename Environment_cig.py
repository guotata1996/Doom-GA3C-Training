from vizdoom import *
import cv2
import numpy as np
import math
from collections import deque

AVAILABLE_ACTIONS = [[1, 0, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0, 0]]

import numpy as np

from Config import Config
resolution = (120,120,6)
frame_repeat = 3

# config For DefendTheCenter
class Environment:
    def __init__(self, rand_seed, display = False, HAND_MODE = False):
        self.game = DoomGame()
        self.game.set_seed(rand_seed)
        self.game.load_config("map/cig.cfg")
        self.game.set_doom_map("map01")  # Limited deathmatch.
        self.game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                                "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
        self.game.add_game_args("+name AI +colorset 0")

        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.HEALTH)
        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(GameVariable.FRAGCOUNT)

        self.last_health = 0
        self.last_ammo = 0
        self.last_fragcount = 0

        self.total_frag_count = 0

        self.game.set_window_visible(display)
        if HAND_MODE:
            self.game.set_mode(Mode.SPECTATOR)
        self.game.init()

        self.game.send_game_command("removebots")
        for i in range(Config.BOTS):
            self.game.send_game_command("addbot")

        self.hand_mode = HAND_MODE

        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = deque(maxlen=self.nb_frames)
        self.reset()
        print "Doomgame instance established"

    def _get_frame(self):
        screen = self.game.get_state().screen_buffer  # 3 x h x w
        screen = screen.transpose((1, 2, 0))  # h x w x 3
        whole_screen = cv2.resize(screen, resolution[:2])  # 120 x 120 x 3
        whole_screen = whole_screen.astype(np.float32)
        centered_screen = screen[160:280, 260:380, :] #for 640x480
        centered_screen = centered_screen.astype(np.float32)
        return np.concatenate((whole_screen, centered_screen), axis=2) #120 x 120 x 6

    def _update_frame_q(self):
        screen = self._get_frame() / 128.0 - 1.0
        self.frame_q.append(screen)
        assert (len(self.frame_q) > 0)

    def reset(self):
        self.frame_q.clear()
        self._update_frame_q()
        self.last_health = 100
        self.last_ammo = 15

    def action(self, action):
        old_position = [self.game.get_game_variable(GameVariable.POSITION_X),
                        self.game.get_game_variable(GameVariable.POSITION_Y)]
        if self.hand_mode:
            self.game.advance_action()
        else:
            self.game.make_action(AVAILABLE_ACTIONS[action], frame_repeat)

        reward = -0.008
        new_position = [self.game.get_game_variable(GameVariable.POSITION_X),
                        self.game.get_game_variable(GameVariable.POSITION_Y)]

        new_fragcount = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        self.total_frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        new_health = max(self.game.get_game_variable(GameVariable.HEALTH), 0)
        new_ammo = self.game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)

        if new_fragcount - self.last_fragcount > 0:
            reward += 2  # killed someone
        if self.game.is_player_dead():
            reward += -1  # killed by someone or by self
        reward += 0.01 * (new_health - self.last_health)
        reward += 0.01 * (new_ammo - self.last_ammo)

        reward += 5e-5 * (
        math.sqrt(math.pow(old_position[0] - new_position[0], 2) + math.pow(old_position[1] - new_position[1], 2)) - 8)
        self.last_fragcount = new_fragcount
        self.last_health = new_health
        self.last_ammo = new_ammo

        new_episode = False  # returns true if game_state after calling this func belongs to a new episode / isOver

        if self.game.is_episode_finished():
            new_episode = True
            self.game.new_episode()
            self.reset()
            self.last_fragcount = 0
        else:
            if self.game.is_player_dead():
                self.reset()
                self.game.respawn_player()
            else:
                self._update_frame_q()
        return reward, new_episode

    def current_state(self):
        assert(len(self.frame_q) > 0)
        diff_len = self.nb_frames - len(self.frame_q)
        if diff_len == 0:
            return np.concatenate(self.frame_q, axis=2), self.game.get_game_variable(GameVariable.FRAGCOUNT)/15.0, max(self.game.get_game_variable(GameVariable.HEALTH), 0)/100.0
        else:
            zeros = [np.zeros_like(self.frame_q[0]) for _ in range(diff_len)] 
            for k in self.frame_q:
                zeros.append(k)
            assert len(zeros) == self.nb_frames
            return np.concatenate(zeros, axis=2), self.game.get_game_variable(GameVariable.FRAGCOUNT)/15.0, max(self.game.get_game_variable(GameVariable.HEALTH), 0)/100.0

    def get_num_actions(self):
        return len(AVAILABLE_ACTIONS)
