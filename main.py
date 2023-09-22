import time

<<<<<<< HEAD
import gymnasium as gym

=======
>>>>>>> origin/3x3
import MyEnv
from NN import NNModel
from Utils import API


def main():
    model_path = "./model/game"
<<<<<<< HEAD
    env = gym.make(
        "TaniJoh-v0",
        max_steps=200,
=======
    env = MyEnv.Game(
        max_steps=500,
>>>>>>> origin/3x3
        render_mode="human",
        use_pyautogui=True,
    )

    nn = NNModel(model_path)
    nn.load_model()

    fa = API()
    match = fa.get_match()
    if len(match) != 1:
        print("match is not one")
    match = match[0]
    id_ = match["id"]

<<<<<<< HEAD
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
=======
    _ = env.reset()
    env.reset_from_api(match)

    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        field = fa.get_field(id_)
        server_turn = field["turn"]
        env.get_stat_from_api(field)
        print(f"turn:{server_turn}, score_A:{env.score_A}, score_B:{env.score_B}")
        env.render()
        if env.current_team == "A":
            actions = nn.predict(env.get_around_workers())
            actions = env.check_actions(actions)
            # actions = env.get_random_actions()
            fa.post_actions(env.make_post_data(actions), id_)
            print(env.make_post_data(actions))
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
>>>>>>> origin/3x3
    env.close()


if __name__ == "__main__":
    main()
