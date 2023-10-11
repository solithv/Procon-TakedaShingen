import glob

import MyEnv

# from NN import NNModel


def main():
    fields = glob.glob("./field_data/C13.csv")
    model_path = "./model"
    model_name = "game"
    env = MyEnv.Game(
        csv_path=fields,
        render_mode="human",
        use_pyautogui=True,
    )

    # nn = NNModel()
    # nn.load_model(model_path, model_name)
    # # nn.make_model(5)
    # nn.model.summary()

    observation, info = env.reset()
    print(info)

    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        env.render()
        if env.current_team == "A":
            actions = env.get_random_actions()
            actions = env.check_actions(actions)
            # actions = [actions[0], 0, 0, 0, 0, 0]
            # print(env.ACTIONS[actions[0]])
        else:
            actions = env.get_random_actions()
            # actions = env.random_act()
            # actions = [0, 0, 0, 0, 0, 0]
            # actions = env.get_actions("pygame")
        # print(actions)
        observation, reward, terminated, truncated, info = env.step(actions)
        # print(
        #     f"turn:{info['turn']}, team:{info['current_team']}, "
        #     + f"score_A:{info['score_A']}, score_B:{info['score_B']}"
        # )
    print("game end")
    print(f"{env.replace_count} action replaced")
    env.end_game_render()
    env.close()


if __name__ == "__main__":
    main()
