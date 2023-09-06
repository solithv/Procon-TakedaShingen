import glob
import os
import tracemalloc

import numpy as np
import srl
from srl.algorithms import ql

# from srl.algorithms import agent57_light
from srl.utils import common

import srlEnv
from srl_util import runner
common.logger_print()


def main():
    param_path = "./model/game/model_param.pkl"

    runner.load_parameter(param_path)
    runner.model_summary()

    runner.set_players()
    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print("mean", np.mean(rewards))

    # --- rendering
    runner.render_window()
    # path = os.path.join(os.path.dirname(__file__), "_game.gif")
    # runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
