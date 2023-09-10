import glob

import gymnasium as gym

import MyEnv


def main():
    fields = glob.glob("./field_data/*.csv")
    env = gym.make(
        "TaniJoh-v0",
        max_steps=100,
        csv_path=fields,
        render_mode="human",
        use_pyautogui=True,
        # first_player=0,
    )
    observation = env.reset()
    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        env.render()
        if env.unwrapped.current_team == "A":
            actions = env.unwrapped.random_act(waste=True)
        else:
            actions = env.unwrapped.get_actions("pygame")
        # actions = env.action_space.sample()
        print(actions)
        observation, reward, terminated, truncated, info = env.step(actions)
        print(f"turn:{info['turn']}, reward:{reward}")
    env.close()


if __name__ == "__main__":
    main()
