import json
import numpy as np
import MyEnv
from pathlib import Path

def main():
    map_name = "C25"
    fields = "./field_data/{}.csv"
    with open("preset.json", "r") as f:
        data = json.load(f)
    for field in fields:
        if Path(field).stem.split("_")[-1] not in data.keys():
            break

    for _ in range(2):
        env = MyEnv.Game(
            csv_path=fields.format(map_name),
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
            i.get_coordinate()
            for worker in dict(sorted(env.workers.items(), key=lambda x: x[0])).values()
            for i in worker
        ]
        while not terminated and not truncated:
            env.render()
            if env.current_team == "A":
                actions = env.get_actions("pygame")
                plannedActionsA.append(actions[: env.worker_count])
                a = actions
            else:
                actions = [0 for _ in range(6)]
                plannedActionsB.append(actions[: env.worker_count])

                with open("preset.json", "r") as f:
                    data = json.load(f)
                plannedActions = np.concatenate(
                    [np.array(plannedActionsA).T, np.array(plannedActionsB).T]
                ).tolist()
                preset = data.get(mapName, {})
                for position, action in zip(worker_initial_positions, plannedActions):
                    if any(w.get_coordinate() == position for w in env.workers["B"]):
                        continue
                    preset[str(position)] = [i for i in action if i != 0]

                data[mapName] = preset

                print(data)
                with open("preset.json", "w") as f:
                    json.dump(
                        dict(sorted(data.items(), key=lambda x: x[0])), f, indent=4
                    )

            observation, reward, terminated, truncated, info = env.step(actions)
            if all(i == 0 for i in a):
                break

        map_name = "inv_" + map_name


if __name__ == "__main__":
    main()
