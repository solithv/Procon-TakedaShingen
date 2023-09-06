import glob
import random

import srl

import srlEnv


def main():
    fields = glob.glob("./field_data/*.csv")
    config = srl.EnvConfig(
        "TaniJoh-v1",
        kwargs={
            "max_episode_steps": 5,
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
            if env.env.controller == "pygame":
                action = env.env.get_actions()
            else:
                action = env.env.random_act()
            env.step(action)
            total_reward += env.reward
            print(
                f"step {env.step_num}, action {action}, reward {env.reward}, done {env.done}"
            )
            env.render()
    env.close()


def test_srl():
    fields = glob.glob("./field_data/*.csv")
    config = srl.EnvConfig(
        "TaniJoh-v1",
        kwargs={
            "max_episode_steps": 100,
            "csv_path": fields,
            "render_mode": "human",
            "controller": "pygame",
        },
    )
    env = srl.make_env(config)
    state = env.reset(render_mode="window")
    # observation = env.reset()
    total_reward = 0
    env.render()

    while not env.done:
        if env.env.controller == "pygame":
            action = env.env.get_actions()
        else:
            action = env.env.random_act()
        env.step(action)
        total_reward += env.reward
        print(
            f"step {env.step_num}, action {action}, reward {env.reward}, done {env.done}"
        )
        env.render()
    env.close()


def test_env():
    fields = glob.glob("./field_data/*.csv")

    # env = srlEnv.Game(csv_path=fields, render_mode="human", controller="pygame")
    env = srlEnv.Game(csv_path=fields, render_mode="ansi", controller="cli")

    observation = env.reset()
    done = False
    print(f"width:{env.width}, height:{env.height}, workers:{env.worker_count}")

    while not done:
        print(
            f"input team {env.current_team} actions (need {env.worker_count} input) : "
        )
        # env.render_rgb_array()
        env.render_terminal()
        observation, reward_A, reward_B, done, info = env.call_step(env.get_actions())
        print(info)


if __name__ == "__main__":
    main()
    # test_srl()
    # test_env()
