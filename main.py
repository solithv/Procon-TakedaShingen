import time

import gymnasium as gym

from MyEnv import API, NNModel


def main():
    model_path = "./model/game"
    env = gym.make(
        "TaniJoh-v0",
        max_steps=200,
        render_mode="human",
        use_pyautogui=True,
    )

    nn = NNModel(model_path)
    nn.make_model()
    # nn.load_model()

    fa = API()
    match = fa.get_match()
    if len(match) != 1:
        print("match is not one")
    match = match[0]
    id_ = match["id"]
    my_turn = int(match["first"])

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
        print(fa.get_field(id_))
        env.render()
        if env.unwrapped.current_team == "A":
            actions = env.unwrapped.get_random_actions()
            # actions = nn.predict(env.unwrapped.get_around_workers())
            fa.post_actions(env.unwrapped.make_post_data(actions), id_)
            print(env.unwrapped.make_post_data(actions))
        else:
            actions = [0 for _ in range(env.unwrapped.worker_count)]
        observation, reward, terminated, truncated, info = env.step(actions)
        print(
            f"turn:{info['turn']}, score_A:{info['score_A']}, score_B:{info['score_B']}"
        )

        while server_turn == fa.get_field(id_)["turn"]:
            time.sleep(0.5)
        server_turn = fa.get_field(id_)["turn"]
    time.sleep(match["turnSeconds"])
    env.unwrapped.get_stat_from_api(fa.get_field(id_))
    env.render()
    print("game end")
    input()
    env.close()


if __name__ == "__main__":
    main()