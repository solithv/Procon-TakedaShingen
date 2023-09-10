import glob

import gymnasium as gym
from gymnasium.utils import env_checker

import MyEnv

fields = glob.glob("./field_data/*.csv")
env = gym.make(
    "TaniJoh-v0",
    # max_steps=100,
    csv_path=fields,
    render_mode="human",
    controller="pygame",
    # first_player=0,
)
env_checker.check_env(env.unwrapped)
