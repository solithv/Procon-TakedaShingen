import glob
import random

import srl

import srlEnv


def main():
    fields = glob.glob("./field_data/*.csv")
    config = srl.EnvConfig(
        "TaniJoh-v1",
        kwargs={
            "max_episode_steps": 2,
            "csv_path": fields,
            "render_mode": "human",
            "controller": "cli",
        },
    )
    env = srl.make_env(config)
    for _ in range(5):
        state = env.reset(render_mode="window")
        # observation = env.reset()
        total_reward = 0
        env.render()

        while not env.done:
            # action = env.env.get_actions()
            action = env.random_act()
            env.step(action)
            total_reward += env.reward
            print(
                f"step {env.step_num}, action {action}, reward {env.reward}, done {env.done}"
            )
            env.render()
    env.close()


if __name__ == "__main__":
    main()
