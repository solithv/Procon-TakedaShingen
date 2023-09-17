import glob
import time

import gymnasium as gym

import MyEnv
from nn import NNModel
from MyEnv.Field_API import API


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
    nn.make_model()
    # nn.load_model()
    fa = API()
    match = fa.get_match()[0]
    id_ = match["id"]

    observation = env.reset()
    env.unwrapped.reset_from_api(match)

    terminated, truncated = [False] * 2
    while True:
        try:
            field = fa.get_field(id_)
        except:
            time.sleep(0.1)
        else:
            server_turn = field["turn"]
            break
    while not terminated and not truncated:
        env.unwrapped.get_stat_from_api(fa.get_field(id_))
        env.render()
        if env.unwrapped.current_team == "A":
            actions = env.unwrapped.random_act()
            # actions = nn.predict(env.unwrapped.get_around_workers())
            fa.post_actions(env.unwrapped.make_post_data(actions), id_)
        else:
            actions = env.unwrapped.random_act()
            fa.post_actions(env.unwrapped.make_post_data(actions), id_, True)
            # actions = env.unwrapped.get_actions("pygame")
        # actions = env.action_space.sample()
        # env.unwrapped.print_around(env.unwrapped.get_around_workers(side_length=5))
        # print(env.unwrapped.make_post_data(actions))
        # print(actions)
        observation, reward, terminated, truncated, info = env.step(actions)
        print(f"turn:{info['turn']}")
        while server_turn == fa.get_field(id_)["turn"]:
            time.sleep(0.5)
        server_turn = fa.get_field(id_)["turn"]
    env.unwrapped.get_stat_from_api(fa.get_field(id_))
    env.render()
    env.close()


if __name__ == "__main__":
    main()
