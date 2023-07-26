import gymnasium as gym
import random
import glob
from MyEnv import Game


def main():
    fields = glob.glob("./field_data/*.csv")
    env = gym.make(
        "TaniJoh-v0",
        # max_episode_steps=10,
        csv_path=random.choice(fields),
        render_mode="human",
        controller="pygame",
    )
    observation = env.reset()
    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        env.render()
        observation, reward, terminated, truncated, _ = env.step(env.get_actions())
        print(f"reward:{reward}")
    env.close()


if __name__ == "__main__":
    main()
