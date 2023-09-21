import csv
import glob
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pygame
from pygame.locals import *

from MyEnv import Game, Worker
from Utils import Util


class Annotator:
    def __init__(
        self, csv_paths, output_dir, filename="data.dat", size: int = 3, max_steps=None
    ) -> None:
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.basename = self.filename.rsplit(".", 1)[0]
        self.csv_paths = csv_paths
        self.size = size
        self.game = Game(
            self.csv_paths, render_mode="human", use_pyautogui=True, max_steps=max_steps
        )
        self.game.reset()
        self.layers = self.game.CELL[: self.game.CELL.index("worker_A0")] + (
            "worker_A",
            "worker_B",
        )
        self.window_surface = None
        self.cwd = os.getcwd()
        self.display_size_x, self.display_size_y = 600, 600
        self.cell_size = min(
            self.display_size_x * 0.9 // self.size,
            self.display_size_y * 0.8 // self.size,
        )
        self.window_size = self.size * self.cell_size
        self.window_size_x = self.size * self.cell_size
        self.window_size_y = self.size * self.cell_size
        self.reset_render()
        self.GRAY = (125, 125, 125)

        if not self.output_dir.joinpath(self.filename).exists() and list(
            self.output_dir.glob("*.zip.[0-9][0-9][0-9]")
        ):
            Util.combine_split_zip(
                self.output_dir.joinpath(self.basename),
                f"{self.output_dir.joinpath(self.basename)}.zip",
            )
            shutil.unpack_archive(
                f"{self.output_dir.joinpath(self.basename)}.zip", self.output_dir
            )

    def reset(self):
        self.load_from_csv(self.csv_paths)

    def finish(self):
        if self.output_dir.joinpath(self.filename).stat().st_size > 100 * (1024**2):
            shutil.make_archive(
                self.output_dir.joinpath(self.basename),
                format="zip",
                root_dir=self.output_dir,
                base_dir=self.filename,
            )
            Util.split_zip(
                f"{self.output_dir.joinpath(self.basename)}.zip",
                self.output_dir.joinpath(self.basename),
            )
            self.output_dir.joinpath(f"{self.basename}.zip").unlink()
            self.output_dir.joinpath(self.filename).unlink()

    def do_annotate(self, only=False):
        features = []
        targets = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[self.layers.index("pond"), y, x]:
                    continue
                feature = self.get_around(y, x, self.size)
                # self.env.print_around([self.update_blank_around(feature)])
                target = np.identity(len(self.game.ACTIONS), dtype=np.int8)[
                    self.get_action(feature)
                ]
                features.append(feature)
                targets.append(target)
                if not only:
                    feature = self.make_random_feature(feature, 0.1)
                    target = np.identity(len(self.game.ACTIONS), dtype=np.int8)[
                        self.get_action(feature)
                    ]
                    features.append(feature)
                    targets.append(target)

                self.save_dataset(features, targets)
                features = []
                targets = []

    def make_random_feature(self, origin: np.ndarray, rate: float):
        conflict_list = (
            {"castle", "rampart_A"},
            {"castle", "rampart_A"},
            {"pond", "worker_A"},
            {"pond", "worker_B"},
            {"worker_A", "worker_B"},
            {"worker_A", "rampart_B"},
            {"worker_B", "rampart_A"},
            {"rampart_A", "territory_A"},
            {"rampart_B", "territory_B"},
            {"rampart_A", "open_territory_A"},
            {"rampart_B", "open_territory_B"},
            {"rampart_A", "open_territory_B"},
            {"rampart_B", "open_territory_A"},
            {"territory_A", "open_territory_A"},
            {"territory_B", "open_territory_B"},
            {"territory_A", "open_territory_B"},
            {"territory_B", "open_territory_A"},
            {"open_territory_A", "open_territory_B"},
        )
        noise = np.random.rand(*origin.shape)
        noise = np.where(noise < rate, 1, 0)
        noise[self.layers.index("pond")] = np.where(
            noise[self.layers.index("pond")] >= 0, 0, noise[self.layers.index("pond")]
        )
        noise[self.layers.index("castle")] = np.where(
            noise[self.layers.index("castle")] >= 0,
            0,
            noise[self.layers.index("castle")],
        )
        center = (self.size // 2 for _ in range(2))
        for y in range(self.size):
            for x in range(self.size):
                is_ok = False
                while not is_ok:
                    for conflict in random.sample(conflict_list, len(conflict_list)):
                        if conflict.issubset(
                            {
                                self.layers[i]
                                for i, value in enumerate(
                                    origin[:, y, x] + noise[:, y, x]
                                )
                                if value > 0
                            }
                        ):
                            item = random.choice(list(conflict))
                            if (y, x) == center and item == "worker_A":
                                break
                            noise[self.layers.index(item), y, x] = 0
                            break
                    else:
                        is_ok = True
        # print(noise)
        feature = np.where(origin >= 0, origin + noise, origin)
        feature = np.where(feature > 0, 1, feature)
        feature = self.update_blank_around(feature)
        return feature

    def save_dataset(self, features, targets):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, self.filename), "a") as f:
            if isinstance(targets, Iterable):
                [
                    print(
                        json.dumps(
                            {
                                "X": x.tolist(),
                                "Y": y.tolist(),
                            }
                        ),
                        file=f,
                    )
                    for x, y in zip(features, targets)
                ]
            else:
                print(
                    json.dumps(
                        {
                            "X": features.tolist(),
                            "Y": targets.tolist(),
                        }
                    ),
                    file=f,
                )

    def make_augmentation(self, feature, target):
        def rotate_augment(feature_, target_, count):
            feature_ = np.rot90(feature_, count, axes=(1, 2))
            target_ = np.argmax(target_)
            target_name = self.game.ACTIONS[target_]
            for _ in range(count):
                if target_name == "stay":
                    continue
                split_name = target_name.split("_")
                target_name = f"{split_name[0]}_{rotate_trans[split_name[-1]]}"
            target_ = self.game.ACTIONS.index(target_name)
            target_ = np.identity(len(self.game.ACTIONS), dtype=np.int8)[target_]
            return feature_, target_

        def horizontal_augment(feature_, target_):
            feature_ = np.flip(feature_, 1)
            target_ = np.argmax(target_)
            target_ = self.game.ACTIONS.index(
                self.game.ACTIONS[target_].translate(horizontal_trans)
            )
            target_ = np.identity(len(self.game.ACTIONS), dtype=np.int8)[target_]
            return feature_, target_

        def vertical_augment(feature_, target_):
            feature_ = np.flip(feature_, 2)
            target_ = np.argmax(target_)
            target_ = self.game.ACTIONS.index(
                self.game.ACTIONS[target_].translate(vertical_trans)
            )
            target_ = np.identity(len(self.game.ACTIONS), dtype=np.int8)[target_]
            return feature_, target_

        rotate_trans = {
            "N": "W",
            "W": "S",
            "S": "E",
            "E": "N",
            "NW": "SW",
            "SW": "SE",
            "SE": "NE",
            "NE": "NW",
        }
        horizontal_trans = str.maketrans({"N": "S", "S": "N"})
        vertical_trans = str.maketrans({"W": "E", "E": "W"})

        data = [rotate_augment(feature, target, i + 1) for i in range(3)]
        data.append(horizontal_augment(feature, target))
        data.append(vertical_augment(feature, target))
        features, targets = [list(x) for x in zip(*data)]
        return features, targets

    def load_from_csv(self, path: Union[str, list[str]]):
        if isinstance(path, (list, tuple)):
            path = random.choice(path)
        size = int(re.sub(r"[\D]", "", os.path.normpath(path).split(os.path.sep)[-1]))
        self.board = np.zeros((len(self.layers), size, size), dtype=np.int8)
        self.width, self.height = [size] * 2

        a_count = 0
        with open(path, "r") as f:
            reader = csv.reader(f)
            for y, row in enumerate(reader):
                for x, item in enumerate(row):
                    if item == "0":
                        self.board[self.layers.index("blank"), y, x] = 1
                    elif item == "1":
                        self.board[self.layers.index("pond"), y, x] = 1
                    elif item == "2":
                        self.board[self.layers.index("castle"), y, x] = 1
                    elif item == "a":
                        a_count += 1
        self.worker_count = a_count
        self.update_blank()

    def update_blank(self):
        self.board[self.layers.index("blank")] = np.where(
            self.board[1:].any(axis=0), 0, 1
        )
        self.board[self.layers.index("blank"), self.height :, :] = -1
        self.board[self.layers.index("blank"), :, self.width :] = -1

    def update_blank_around(self, field: np.ndarray):
        assert self.layers.index("blank") != len(self.layers) - 1
        field[self.layers.index("blank")] = np.where(
            field[self.layers.index("blank") + 1 :].any(axis=0), 0, 1
        )
        field[self.layers.index("blank")] = np.where(
            np.any(field[self.layers.index("blank") + 1 :] < 0, axis=0),
            -1,
            field[self.layers.index("blank")],
        )
        return field

    def get_around(self, y: int, x: int, side_length: int = 3):
        if side_length % 2 == 0:
            raise ValueError("need to input an odd number")
        length_ = side_length // 2
        field = np.pad(
            self.board,
            [(0, 0), (length_, length_), (length_, length_)],
            "constant",
            constant_values=-1,
        )
        front = length_ * 2 + 1
        field = field[:, y : y + front, x : x + front]
        field[self.layers.index("worker_A"), length_, length_] = 1
        self.update_blank()
        return field

    def reset_render(self):
        self.IMG_SCALER = np.array((self.cell_size, self.cell_size))
        self.BLANK_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/blank.png"), self.IMG_SCALER
        )
        self.POND_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/pond.png"), self.IMG_SCALER
        )
        self.CASTLE_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/castle.png"), self.IMG_SCALER
        )
        self.RAMPART_A_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/rampart_A.png"), self.IMG_SCALER
        )
        self.RAMPART_B_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/rampart_B.png"), self.IMG_SCALER
        )
        self.WORKER_A_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_A.png"), self.IMG_SCALER
        )
        self.WORKER_B_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_B.png"), self.IMG_SCALER
        )
        self.WORKER_A_HOVER_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_A_hover.png"), self.IMG_SCALER
        )
        self.WORKER_B_HOVER_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_B_hover.png"), self.IMG_SCALER
        )

    def drawGrids(self):
        # 縦線描画
        for i in range(1, self.size):
            pygame.draw.line(
                self.window_surface,
                self.game.BLACK,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size_y),
                1,
            )
        # 横線描画
        for i in range(1, self.size):
            pygame.draw.line(
                self.window_surface,
                self.game.BLACK,
                (0, i * self.cell_size),
                (self.window_size_x, i * self.cell_size),
                1,
            )

    def placeImage(self, img, i, j, scale=1.0):
        """
        i, j番目に画像描画する関数
        workerNumber: str 職人番号
        scale: float 画像の倍率
        """
        placement = (
            (
                self.cell_size * (j + (1 - scale) / 2),
                self.cell_size * (i + (1 - scale) / 2),
            )
            if scale != 1.0
            else (j * self.cell_size, i * self.cell_size)
        )
        img = pygame.transform.scale(img, self.IMG_SCALER * scale)
        self.window_surface.blit(img, placement)

    def drawAll(self, view):
        for i in range(self.size):
            for j in range(self.size):
                self.placeImage(self.BLANK_IMG, i, j)
                cellInfo = view[i][j]
                if not cellInfo:
                    pygame.draw.rect(
                        self.window_surface,
                        self.GRAY,
                        (
                            j * self.cell_size,
                            i * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
                worker_A_exist = "worker_A" in cellInfo
                worker_B_exist = "worker_B" in cellInfo

                if "castle" in cellInfo and worker_A_exist:
                    self.placeImage(self.CASTLE_IMG, i, j)
                    self.placeImage(
                        self.WORKER_A_IMG,
                        i,
                        j,
                        scale=0.7,
                    )
                elif "castle" in cellInfo and worker_B_exist:
                    self.placeImage(self.CASTLE_IMG, i, j)
                    self.placeImage(
                        self.WORKER_B_IMG,
                        i,
                        j,
                        scale=0.7,
                    )
                elif "rampart_A" in cellInfo and worker_A_exist:
                    self.placeImage(self.RAMPART_A_IMG, i, j)
                    self.placeImage(
                        self.WORKER_A_IMG,
                        i,
                        j,
                        scale=0.8,
                    )
                elif "rampart_B" in cellInfo and worker_B_exist:
                    self.placeImage(self.RAMPART_B_IMG, i, j)
                    self.placeImage(
                        self.WORKER_B_IMG,
                        i,
                        j,
                        scale=0.8,
                    )
                elif "pond" in cellInfo and "rampart_A" in cellInfo:
                    self.placeImage(self.POND_IMG, i, j)
                    self.placeImage(self.RAMPART_A_IMG, i, j, scale=0.8)
                elif "pond" in cellInfo and "rampart_B" in cellInfo:
                    self.placeImage(self.POND_IMG, i, j)
                    self.placeImage(self.RAMPART_B_IMG, i, j, scale=0.8)
                elif "castle" in cellInfo:
                    self.placeImage(self.CASTLE_IMG, i, j)
                elif worker_A_exist:
                    self.placeImage(self.WORKER_A_IMG, i, j)
                elif worker_B_exist:
                    self.placeImage(self.WORKER_B_IMG, i, j)
                elif "pond" in cellInfo:
                    self.placeImage(self.POND_IMG, i, j)
                elif "rampart_A" in cellInfo:
                    self.placeImage(self.RAMPART_A_IMG, i, j)
                elif "rampart_B" in cellInfo:
                    self.placeImage(self.RAMPART_B_IMG, i, j)
                elif "blank" in cellInfo:
                    self.placeImage(self.BLANK_IMG, i, j)
        self.drawGrids()

    def get_action(self, board):
        def nonAllowedMovements(x, y, directions):
            coordinates = np.round(
                np.array(
                    [
                        [
                            np.cos(2 * n * np.pi / directions) + x,
                            np.sin(2 * n * np.pi / directions) + y,
                        ]
                        for n in range(directions)
                    ]
                )
            )

            return coordinates

        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            self.window_surface = pygame.display.set_mode(
                (self.window_size_x, self.window_size_y + self.cell_size * 2)
            )
            pygame.display.set_caption("annotator")
        view = [
            [
                [self.layers[i] for i, item in enumerate(board[:, y, x]) if item > 0]
                for x in range(self.size)
            ]
            for y in range(self.size)
        ]

        self.drawAll(view)
        showTerritory = False
        action = None
        while action is None:
            for event in pygame.event.get():
                mouseX, mouseY = pygame.mouse.get_pos()
                cellX = int(mouseX // self.cell_size)
                cellY = int(mouseY // self.cell_size)
                workerY, workerX = [self.size // 2] * 2
                if event.type == KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if showTerritory:
                            self.drawAll(view)
                        showTerritory = not showTerritory

                if showTerritory:
                    territoryALayer = self.game.compile_layers(
                        board, "territory_A", one_hot=True
                    )
                    territoryBLayer = self.game.compile_layers(
                        board, "territory_B", one_hot=True
                    )
                    openTerritoryALayer = self.game.compile_layers(
                        board, "open_territory_A", one_hot=True
                    )
                    openTerritoryBLayer = self.game.compile_layers(
                        board, "open_territory_B", one_hot=True
                    )
                    pygame.display.update()
                    for i in range(self.size):
                        for j in range(self.size):
                            if territoryALayer[i][j] == territoryBLayer[i][j] == 1:
                                color = self.game.PURPLE
                            elif territoryALayer[i][j] == 1:
                                color = self.game.RED
                            elif territoryBLayer[i][j] == 1:
                                color = self.game.BLUE
                            elif openTerritoryALayer[i][j] == 1:
                                color = self.game.PINK
                            elif openTerritoryBLayer[i][j] == 1:
                                color = self.game.SKY
                            elif territoryALayer[i][j] < 0:
                                pygame.draw.rect(
                                    self.window_surface,
                                    self.GRAY,
                                    (
                                        j * self.cell_size,
                                        i * self.cell_size,
                                        self.cell_size,
                                        self.cell_size,
                                    ),
                                )
                                continue
                            else:
                                self.placeImage(self.BLANK_IMG, i, j)
                                continue

                            pygame.draw.rect(
                                self.window_surface,
                                color,
                                (
                                    j * self.cell_size,
                                    i * self.cell_size,
                                    self.cell_size,
                                    self.cell_size,
                                ),
                            )
                    self.drawGrids()
                    continue

                self.placeImage(
                    self.WORKER_A_HOVER_IMG,
                    workerY,
                    workerX,
                    scale=1.0,
                )
                pygame.display.update()

                if event.type == KEYDOWN:
                    if event.key == pygame.K_1:
                        if not np.any(
                            np.all(
                                np.append(
                                    nonAllowedMovements(workerX, workerY, 8),
                                    [[workerX, workerY]],
                                    axis=0,
                                )
                                == np.array([cellX, cellY]),
                                axis=1,
                            )
                        ):
                            continue
                        directionVector = np.array([cellX - workerX, workerY - cellY])
                        if directionVector[0] == directionVector[1] == 0:
                            action = 0
                        else:
                            action = int(
                                (
                                    (
                                        np.round(
                                            np.degrees(
                                                np.arctan2(
                                                    directionVector[0],
                                                    directionVector[1],
                                                )
                                            )
                                        )
                                        / 45
                                    )
                                    % 8
                                )
                                + 1
                            )
                        self.placeImage(self.BLANK_IMG, workerY, workerX)
                        self.placeImage(
                            self.WORKER_A_IMG,
                            cellY,
                            cellX,
                        )
                        self.drawGrids()
                        pygame.display.update()
                    # build
                    elif event.key == pygame.K_2:
                        if not np.any(
                            np.all(
                                nonAllowedMovements(workerX, workerY, 4)
                                == np.array([cellX, cellY]),
                                axis=1,
                            )
                        ):
                            continue
                        directionVector = np.array([cellX - workerX, workerY - cellY])
                        action = int(
                            (
                                np.round(
                                    np.degrees(
                                        np.arctan2(
                                            -directionVector[1],
                                            directionVector[0],
                                        )
                                    )
                                )
                                / 90
                            )
                            + 10
                        )
                        self.placeImage(self.BLANK_IMG, workerY, workerX)
                        self.placeImage(
                            self.WORKER_A_IMG,
                            workerY,
                            workerX,
                        )
                        self.placeImage(self.RAMPART_A_IMG, cellY, cellX)
                        self.drawGrids()
                        pygame.display.update()

                    # break
                    elif event.key == pygame.K_3:
                        if not np.any(
                            np.all(
                                nonAllowedMovements(workerX, workerY, 4)
                                == np.array([cellX, cellY]),
                                axis=1,
                            )
                        ):
                            continue
                        directionVector = np.array([cellX - workerX, workerY - cellY])
                        action = int(
                            (
                                np.round(
                                    np.degrees(
                                        np.arctan2(
                                            -directionVector[1],
                                            directionVector[0],
                                        )
                                    )
                                )
                                / 90
                            )
                            + 14
                        )

                        self.placeImage(self.BLANK_IMG, workerY, workerX)
                        self.placeImage(
                            self.WORKER_A_IMG,
                            workerY,
                            workerX,
                        )
                        self.placeImage(self.BLANK_IMG, cellY, cellX)
                        self.drawGrids()
                        pygame.display.update()
        return action

    def play_game_annotator(self, enemy=""):
        self.game = Game(self.csv_paths, render_mode="human", use_pyautogui=True)
        observation = self.game.reset()

        terminated, truncated = [False] * 2
        while not terminated and not truncated:
            self.game.render()
            if self.game.current_team == "A":
                workers = self.game.workers["A"]
                actions = self.game.get_actions("pygame")
                features, targets = self.game_dataset_maker(
                    self.game.board, actions, workers
                )
                self.save_dataset(features, targets)
            else:
                if enemy == "smart":
                    actions = self.game.get_random_actions()
                elif enemy == "human":
                    actions = self.game.get_actions("pygame")
                else:
                    actions = self.game.random_act()
            observation, reward, terminated, truncated, info = self.game.step(actions)
            print(
                f"turn:{info['turn']}, score_A:{info['score_A']}, score_B:{info['score_B']}"
            )
        self.game.end_game_render()
        self.game.close()

    def game_dataset_maker(
        self, board: np.ndarray, actions: list[int], workers: list[Worker]
    ):
        features = []
        targets = []
        for worker, action in zip(workers, actions):
            feature = self.game.get_around(board, *worker.get_coordinate(), self.size)
            target = np.identity(len(self.game.ACTIONS), dtype=np.int8)[action]
            features.append(feature)
            targets.append(target)
        return features, targets


def main():
    output_dir = "./dataset"
    csv_dir = "./field_data"
    filename = "data.dat"
    # random: ランダム
    # smart: 強強ランダム
    # human: 手動
    enemy = "smart"
    # 最大ターン数を指定
    # 数値入力で指定可能
    # Noneでマップサイズに応じて可変
    max_steps = None
    annotator = Annotator(
        glob.glob(os.path.join(csv_dir, "*.csv")),
        output_dir,
        filename,
        size=5,
        max_steps=max_steps,
    )
    for _ in range(1):
        annotator.reset()
        annotator.play_game_annotator(enemy)
        # annotator.do_annotate()
    annotator.finish()


if __name__ == "__main__":
    main()
