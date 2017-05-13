from vizdoom import *
import numpy

AVAILABLE_ACTIONS = [[1, 0, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0, 0]]

game = DoomGame()
game.set_seed(0)
game.load_config("map/cig.cfg")
game.set_doom_map("map01")  # Limited deathmatch.
game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                                "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
game.add_game_args("+name AI +colorset 0")
game.init()

game.send_game_command("removebots")
for i in range(10):
	game.send_game_command("addbot")

while True:
	action = numpy.random.choice(range(len(AVAILABLE_ACTIONS)))
	game.make_action(AVAILABLE_ACTIONS[action])
	if game.is_player_dead():
		break

game.respawn_player()
print game.get_state()
