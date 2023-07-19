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
    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        env.render()
        # terminated: エピソード終了フラグ
        # truncated: ステップ数上限での終了フラグ
        observation, reward, terminated, truncated, _ = env.step(
            env.get_actions_from_render()
        )
    env.close()


if __name__ == "__main__":
    main()
