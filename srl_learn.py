import os
import tracemalloc

import numpy as np
from srl.utils import common

import srlEnv
from srl_util import runner

common.logger_print()


def main():
    model_path = "./model/game"
    os.makedirs(model_path, exist_ok=True)

    runner.model_summary()

    # --- train
    runner.train(max_episodes=10)
    runner.save_parameter(os.path.join(model_path, "model_param.pkl"))

    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print("mean", np.mean(rewards))

    # --- rendering
    runner.render_window()
    # path = os.path.join(os.path.dirname(__file__), "_game.gif")
    # runner.animation_save_gif(path)


if __name__ == "__main__":
    tracemalloc.start()
    main()
    tracemalloc.stop()
