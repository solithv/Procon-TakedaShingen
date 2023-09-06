import glob

import srl
from srl.algorithms import ql

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

rl_config = ql.Config()

runner = srl.Runner(env_config, rl_config)