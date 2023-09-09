import glob

import srl
from srl.algorithms import ql, dqn, stochastic_muzero

import srlEnv

fields = glob.glob("./field_data/*.csv")
max_episode = 25 * 2
env_config = srl.EnvConfig(
    "TaniJoh-v1",
    kwargs={
        "max_episode_steps": max_episode,
        "csv_path": fields,
        "render_mode": "human",
        "controller": "cli",
    },
)

# rl_config = ql.Config()
rl_config = dqn.Config()
rl_config.batch_size = 1
rl_config.memory.capacity = 1000
rl_config.set_config_by_env(env_config.make_env())
rl_config.image_block.set_dqn_image()

# rl_config = stochastic_muzero.Config()
# rl_config.input_image_block.set_muzero_atari_block()

runner = srl.Runner(env_config, rl_config)
