import time

import MyEnv
from Utils import API

# from NN import NNModel


def main():
    env = MyEnv.Game(
        max_steps=500,
        render_mode="human",
        # use_pyautogui=True,
    )

    # model_path = "./model"
    # model_name = "game"
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
            fa.post_actions(env.make_post_data(actions), id_)

            # manual_actions = env.get_actions_from_pygame()
            # for i, m_action in enumerate(manual_actions):
            #     if m_action != 0:
            #         actions[i] = m_action
            # fa.post_actions(env.make_post_data(actions), id_)

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
