import time

import MyEnv
from Utils import API

# from NN import NNModel


def main():
    # model_path = "./model"
    # model_name = "game"
    env = MyEnv.Game(
        max_steps=500,
        render_mode="human",
        use_pyautogui=True,
        preset_file=None,
    )

    # nn = NNModel()
    # nn.load_model(model_path, model_name)

    fa = API()
    match = fa.get_match()
    if len(match) != 1:
        print("match is not one")
    match = match[0]
    id_ = match["id"]

    env.reset_from_api(match)

    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        field = fa.get_field(id_)
        print([field[k] for k in ("id", "turn", "logs")])
        server_turn = field["turn"]
        env.get_stat_from_api(field)
        print(f"turn:{server_turn}, score_A:{env.score_A}, score_B:{env.score_B}")
        env.render()
        if env.current_team == "A":
            # actions = nn.predict(env.get_around_workers())
            actions = env.get_random_actions()
            actions = env.check_actions(actions)
            print([env.ACTIONS[action] for action in actions])
            print(env.make_post_data(actions))
            fa.post_actions(env.make_post_data(actions), id_)
            _, _, terminated, truncated, _ = env.step(actions)
        else:
            _, _, terminated, truncated, _ = env.dummy_step()

        while server_turn == fa.get_field(id_)["turn"]:
            time.sleep(0.5)
    time.sleep(match["turnSeconds"])
    env.get_stat_from_api(fa.get_field(id_))
    print("game end")
    print(f"score_A:{env.score_A}, score_B:{env.score_B}")
    env.end_game_render()
    env.close()


if __name__ == "__main__":
    main()
