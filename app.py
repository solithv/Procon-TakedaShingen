import glob

import gymnasium as gym

import MyEnv
from nn import NNModel


def main():
    fields = glob.glob("./field_data/*.csv")
    model_path = "./model/game"
    env = gym.make(
        "TaniJoh-v0",
        max_steps=100,
        csv_path=fields,
        render_mode="human",
        use_pyautogui=True,
    )
    nn = NNModel(model_path)
    # nn.load_model()
    observation = env.reset()
    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        env.render()
        if env.unwrapped.current_team == "A":
            actions = env.unwrapped.random_act(waste=True)
            # actions = nn.predict(env.unwrapped.get_around_workers())
        else:
            actions = env.unwrapped.get_actions("pygame")
        # actions = env.action_space.sample()
        print(actions)
        observation, reward, terminated, truncated, info = env.step(actions)
        print(f"turn:{info['turn']}, reward:{reward}")
    env.close()


if __name__ == "__main__":
    main()
