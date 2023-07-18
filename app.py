import gymnasium as gym
import random
import glob
from MyEnv import Game


def main():
    fields = glob.glob("./field_data/*.csv")
    env = gym.make(
        "TaniJoh-v0",
        max_episode_steps=10,
        csv_path=random.choice(fields),
        render_mode="human",
        controller="pygame",
    )
    observation = env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done, *_ = env.step(env.get_actions_from_render())


if __name__ == "__main__":
    main()
