import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("TOKEN")
teams = [{"name": "武田進言", "token": token}, {"name": "相手チーム", "token": "dummy-token"}]
turns = {11: 30, 13: 54, 15: 80, 17: 100, 21: 150, 25: 200}
seconds = {11: 3, 13: 4, 15: 6, 17: 8, 21: 11, 25: 15}
bonus = {"wall": 10, "territory": 30, "castle": 100}


def make_configs():
    csv_dir = "./field_data"
    config_dir = "./server_configs"
    os.makedirs(config_dir, exist_ok=True)
    for filename in Path(config_dir).glob("*.json"):
        filename.unlink()
    for csv_ in Path(csv_dir).glob("*.csv"):
        data = {}
        size = int(re.sub(r"[\D]", "", os.path.normpath(csv_).split(os.path.sep)[-1]))
        name = os.path.normpath(csv_).split(os.path.sep)[-1].split(".")[0]
        id_ = size
        id_ += 100 if "A" in name else 200 if "B" in name else 300
        if "inv_" in name:
            id_ += 300
        structures = np.zeros((size, size), dtype=np.int8)
        masons = np.zeros((size, size), dtype=np.int8)
        with csv_.open() as fi:
            reader = csv.reader(fi)
            a_count, b_count = 0, 0
            for y, row in enumerate(reader):
                for x, item in enumerate(row):
                    if item == "1":
                        structures[y, x] = 1
                    elif item == "2":
                        structures[y, x] = 2
                    elif item == "a":
                        a_count += 1
                        masons[y, x] = a_count
                    elif item == "b":
                        b_count -= 1
                        masons[y, x] = b_count

        data["teams"] = teams
        data["match"] = {
            "id": id_,
            "turns": turns[size],
            "turnSeconds": seconds[size],
            "bonus": bonus,
        }
        data["match"]["board"] = {
            "width": size,
            "height": size,
            "mason": a_count,
            "structures": structures.tolist(),
            "masons": masons.tolist(),
        }
        with open(
            os.path.join(config_dir, f"{csv_.stem}_f.json"), "w", encoding="utf-8"
        ) as fo:
            json.dump(data, fo, indent=4, ensure_ascii=False)
        data["teams"] = list(reversed(teams))
        data["match"]["id"] += 1
        with open(
            os.path.join(config_dir, f"{csv_.stem}_s.json"), "w", encoding="utf-8"
        ) as fo:
            json.dump(data, fo, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    make_configs()
