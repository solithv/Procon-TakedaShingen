import glob
import time

import MyEnv
from NN import NNModel
from Utils import API


def main():
    fields = glob.glob("./field_data/*.csv")
    model_path = "./model/game"
    env = MyEnv.Game(
        csv_path=fields,
        render_mode="human",
        use_pyautogui=True,
    )

    nn = NNModel(model_path)
    nn.load_model()
    nn.make_model(5)
    nn.model.summary()

    observation = env.reset()

    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        env.render()
        # env.print_around(env.get_around_workers(side_length=5))
        if env.current_team == "A":
            actions = env.get_random_actions()
            # actions = nn.predict(env.get_around_workers(5))
            print(actions)
        else:
            actions = env.get_random_actions()
            # actions = env.random_act()
            # actions = env.get_actions("pygame")
        # print(actions)
        observation, reward, terminated, truncated, info = env.step(actions)
        print(
            f"turn:{info['turn']}, score_A:{info['score_A']}, score_B:{info['score_B']}"
        )
    env.render()
    print("game end")
    input()
    env.close()


def server():
    model_path = "./model/game"
    env = MyEnv.Game(
        max_steps=200,
        render_mode="human",
        use_pyautogui=True,
    )

    nn = NNModel(model_path)
    # nn.make_model()
    nn.load_model()

    fa = API()
    match = fa.get_match()
    if len(match) != 1:
        print("match is not one")
    match = match[0]
    id_ = match["id"]

    observation = env.reset()
    env.reset_from_api(match)

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
        env.get_stat_from_api(fa.get_field(id_))
        print(fa.get_field(id_))
        env.render()
        if env.current_team == "A":
            actions = env.get_random_actions()
            # actions = nn.predict(env.get_around_workers())
            fa.post_actions(env.make_post_data(actions), id_)
        else:
            actions = env.get_random_actions()
            fa.post_actions(env.make_post_data(actions), id_, True)
        # env.print_around(env.get_around_workers(side_length=5))
        print(env.make_post_data(actions))
        # print(actions)
        observation, reward, terminated, truncated, info = env.step(actions)
        print(
            f"turn:{info['turn']}, score_A:{info['score_A']}, score_B:{info['score_B']}"
        )
        while server_turn == fa.get_field(id_)["turn"]:
            time.sleep(0.5)
        server_turn = fa.get_field(id_)["turn"]
    env.get_stat_from_api(fa.get_field(id_))
    env.render()
    print("game end")
    input()
    env.close()


if __name__ == "__main__":
    main()
    # server()
