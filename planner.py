import glob
import json
import numpy as np
import MyEnv
from pathlib import Path


def main():
    fields = glob.glob("./field_data/A15.csv")
    with open("preset.json", "r") as f:
        data = json.load(f)
    for field in fields:
        if Path(field).stem.split("_")[-1] not in data.keys():
            break

    env = MyEnv.Game(
        csv_path=field,
        render_mode="human",
        use_pyautogui=True,
        first_player=0,
        preset_file=None,
    )
    observation, info = env.reset()
    print(info)
    mapName = info["csv_name"].replace("inv_", "")
    plannedActionsA = []
    plannedActionsB = []
    terminated, truncated = [False] * 2
    worker_initial_positions = [
        i.get_coordinate() for worker in env.workers.values() for i in worker
    ]

    while not terminated and not truncated:
        env.render()
        if env.current_team == "A":
            actions = env.get_actions("pygame")
            plannedActionsA.append(actions[: env.worker_count])
        else:
            actions = env.get_actions("pygame")
            plannedActionsB.append(actions[: env.worker_count])

        if env.current_team == "B":
            with open("preset.json", "r") as f:
                data = json.load(f)

            plannedActions = np.concatenate(
                [np.array(plannedActionsA).T, np.array(plannedActionsB).T]
            ).tolist()
            preset = {}
            for position, action in zip(worker_initial_positions, plannedActions):
                preset[str(position)] = [i for i in action if i != 0]

            data[mapName] = preset

            print(data)
            with open("preset.json", "w") as f:
                json.dump(data, f, indent=4)

        observation, reward, terminated, truncated, info = env.step(actions)

    print("game end")
    print(f"{env.replace_count} action replaced")
    env.end_game_render()
    env.close()


if __name__ == "__main__":
    main()
