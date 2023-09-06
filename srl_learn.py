import glob
import os
import tracemalloc

import numpy as np
import srl
from srl.algorithms import ql

# from srl.algorithms import agent57_light
from srl.utils import common

import srlEnv

common.logger_print()


def main():
    model_path = "./model/game"
    os.makedirs(model_path, exist_ok=True)
    max_episode = 25 * 2
    fields = glob.glob("./field_data/*.csv")
    env_config = srl.EnvConfig(
        "TaniJoh-v1",
        kwargs={
            "max_episode_steps": max_episode,
            "csv_path": fields,
            "render_mode": "human",
            "controller": "cli",
        },
    )

    rl_config = ql.Config()
    # rl_config = agent57_light.Config(
    #     # hidden_layer_sizes=(64, 64),
    #     enable_rescale=True,
    #     #
    #     enable_double_dqn=True,
    #     # enable_dueling_network=True,
    #     #
    #     actor_num=4,
    #     enable_intrinsic_reward=True,
    #     input_ext_reward=False,
    #     input_int_reward=False,
    #     input_action=False,
    # )
    # rl_config.memory.capacity = 100_000
    # rl_config.memory.set_replay_memory()

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    tracemalloc.start()
    # --- train
    runner.train(max_episodes=1)
    # runner.save(os.path.join(model_path, "model.pkl"))
    runner.save_parameter(os.path.join(model_path, "model_param.pkl"))
    tracemalloc.stop()

    # --- evaluate
    rewards = runner.evaluate(max_episodes=1)
    print("mean", np.mean(rewards))

    # --- rendering
    runner.render_window()
    # path = os.path.join(os.path.dirname(__file__), "_game.gif")
    # runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
