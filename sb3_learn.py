import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import glob
from MyEnv import Game

import os
import shutil
import torch


def learn():
    model_name = "PPO_TaniJoh"
    # buffer_name = "PPO_TaniJoh_replay_buffer"
    fields = glob.glob("./field_data/*.csv")

    env = gym.make("TaniJoh-v0", csv_paths=fields, render_mode="human")

    try:
        # try:
        #     shutil.unpack_archive(f"{buffer_name}.zip", os.getcwd())
        # except:
        #     pass
        model = PPO.load(model_name, env=env, print_system_info=True)
        # model.load_replay_buffer(buffer_name)
    except:
        model = PPO("MlpPolicy", env, verbose=1)
        print("model load failed")
    else:
        print("model loaded")

    for _ in range(1):
        model.learn(total_timesteps=int(1e5), progress_bar=True)
        model.save(model_name)
        # model.save_replay_buffer(buffer_name)
    # shutil.make_archive(
    #     buffer_name, format="zip", root_dir=os.getcwd(), base_dir=f"{buffer_name}.pkl"
    # )

    vec_env = model.get_env()
    obs = vec_env.reset()
    dones = False
    vec_env.render("human")
    for i in range(2 * 1000):
        if dones:
            print("Done!")
            obs = vec_env.reset()
            break
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    learn()
