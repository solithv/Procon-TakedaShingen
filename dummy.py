import time

import MyEnv
from Utils import API


def main():
    env = MyEnv.Game(
        max_steps=500,
        render_mode=None,
    )

    fa = API()
    match = fa.get_match()
    if len(match) != 1:
        print("match is not one")
    match = match[0]
    id_ = match["id"]

    _ = env.reset()
    env.reset_from_api(match)

    terminated, truncated = [False] * 2
    while not terminated and not truncated:
        field = fa.get_field(id_)
        server_turn = field["turn"]
        env.get_stat_from_api(field)
        print(f"turn:{server_turn}, score_A:{env.score_A}, score_B:{env.score_B}")
        env.render()
        if env.current_team == "B":
            actions = env.random_act()
            fa.post_actions(env.make_post_data(actions), id_, True)
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
    env.close()


if __name__ == "__main__":
    main()
