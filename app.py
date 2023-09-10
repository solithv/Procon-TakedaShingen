import gymnasium as gym
import random
import glob
from MyEnv import Game


def main():
    fields = glob.glob("./field_data/*.csv")
    env = gym.make(
        "TaniJoh-v0",
        # max_episode_steps=10,
        csv_path=fields,
        render_mode="human",
        controller="None",
        # first_player=0,
        use_pyautogui=True,
    )
    observation = env.reset()
    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        env.render()
        actions = env.unwrapped.get_actions()
        # actions = env.unwrapped.random_act()
        actions = env.action_space.sample()
        print(actions)
        observation, reward, terminated, truncated, _ = env.step(actions)
        print(f"reward:{reward}")
    env.close()


if __name__ == "__main__":
    main()
