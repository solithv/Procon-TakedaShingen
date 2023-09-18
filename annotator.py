import copy
import csv
import glob
import json
import os
import random
import re
from typing import Iterable, Union

import numpy as np
import pygame
from pygame.locals import *

from MyEnv import Game


class Annotator:
    def __init__(self, csv_paths, output_dir, size: int = 3) -> None:
        self.output_dir = output_dir
        self.csv_paths = csv_paths
        self.size = size
        self.env = Game()
        self.env.reset()
        self.layers = self.env.CELL[: self.env.CELL.index("worker_A0")] + (
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

    def reset(self):
        self.load_from_csv(self.csv_paths)

    def do_annotate(self):
        features = []
        targets = []
        for y in range(self.height):
            for x in range(self.width):
                if self.board[self.layers.index("pond"), y, x]:
                    continue
                feature = self.get_around(y, x, self.size)
                target = np.identity(len(self.env.ACTIONS), dtype=np.int8)[
                    self.get_action(feature)
                ]
                features.append(feature)
                targets.append(target)

        self.save_dataset(features, targets)
        features = []
        targets = []

    def save_dataset(self, features, targets):
        with open(os.path.join(self.output_dir, "data.dat"), "a") as f:
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

    def make_annotation(self, feature, target):
        def rotate_annotate(feature_, target_, count):
            feature_ = np.rot90(feature_, count, axes=(1, 2))
            target_ = np.argmax(target_)
            target_name = self.env.ACTIONS[target_]
            for _ in range(count):
                if target_name == "stay":
                    continue
                elif len(target_name.split("_")[-1]) == 1:
                    target_name = target_name.translate(str.maketrans(rotate_trans))
                else:
                    split_name = target_name.split("_")
                    target_name = f"{split_name[0]}_{rotate_trans2[split_name[-1]]}"
            target_ = self.env.ACTIONS.index(target_name)
            target_ = np.identity(len(self.env.ACTIONS), dtype=np.int8)[target_]
            return feature_, target_

        def horizontal_annotate(feature_, target_):
            feature_ = np.flip(copy.deepcopy(feature_), 1).copy()
            target_ = np.argmax(target_)
            target_ = self.env.ACTIONS.index(
                self.env.ACTIONS[target_].translate(horizontal_trans)
            )
            target_ = np.identity(len(self.env.ACTIONS), dtype=np.int8)[target_]
            return feature_, target_

        def vertical_annotate(feature_, target_):
            feature_ = np.flip(copy.deepcopy(feature_), 2).copy()
            target_ = np.argmax(target_)
            target_ = self.env.ACTIONS.index(
                self.env.ACTIONS[target_].translate(vertical_trans)
            )
            target_ = np.identity(len(self.env.ACTIONS), dtype=np.int8)[target_]
            return feature_, target_

        rotate_trans = str.maketrans({"N": "W", "W": "S", "S": "E", "E": "N"})
        rotate_trans2 = {"NW": "SW", "SW": "SE", "SE": "NE", "NE": "NW"}
        horizontal_trans = str.maketrans({"N": "S", "S": "N"})
        vertical_trans = str.maketrans({"W": "E", "E": "W"})

        data = [rotate_annotate(feature, target, i + 1) for i in range(3)]
        data.append(horizontal_annotate(feature, target))
        data.append(vertical_annotate(feature, target))
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
                self.env.BLACK,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size_y),
                1,
            )
        # 横線描画
        for i in range(1, self.size):
            pygame.draw.line(
                self.window_surface,
                self.env.BLACK,
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
                worker_A_exist = any(
                    f"worker_A{k}" in cellInfo for k in range(self.env.WORKER_MAX)
                )
                worker_B_exist = any(
                    f"worker_B{k}" in cellInfo for k in range(self.env.WORKER_MAX)
                )

                if "castle" in cellInfo and worker_A_exist:
                    self.placeImage(self.CASTLE_IMG, i, j)
                    self.placeImage(
                        self.WORKER_A_IMG,
                        i,
                        j,
                        workerNumber=cellInfo[-1][-1],
                        scale=0.7,
                    )
                elif "castle" in cellInfo and worker_B_exist:
                    self.placeImage(self.CASTLE_IMG, i, j)
                    self.placeImage(
                        self.WORKER_B_IMG,
                        i,
                        j,
                        workerNumber=cellInfo[-1][-1],
                        scale=0.7,
                    )
                elif "rampart_A" in cellInfo and worker_A_exist:
                    self.placeImage(self.RAMPART_A_IMG, i, j)
                    self.placeImage(
                        self.WORKER_A_IMG,
                        i,
                        j,
                        workerNumber=cellInfo[-1][-1],
                        scale=0.8,
                    )
                elif "rampart_B" in cellInfo and worker_B_exist:
                    self.placeImage(self.RAMPART_B_IMG, i, j)
                    self.placeImage(
                        self.WORKER_B_IMG,
                        i,
                        j,
                        workerNumber=cellInfo[-1][-1],
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
                    self.placeImage(
                        self.WORKER_A_IMG, i, j, workerNumber=cellInfo[-1][-1]
                    )
                elif worker_B_exist:
                    self.placeImage(
                        self.WORKER_B_IMG, i, j, workerNumber=cellInfo[-1][-1]
                    )
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
                [
                    self.layers[i]
                    for i, item in enumerate(board[:, y, x].astype(bool))
                    if item
                ]
                for x in range(self.size)
            ]
            for y in range(self.size)
        ]

        self.drawAll(view)
        if "action" in locals():
            del action
        while "action" not in locals():
            for event in pygame.event.get():
                mouseX, mouseY = pygame.mouse.get_pos()
                cellX = int(mouseX // self.cell_size)
                cellY = int(mouseY // self.cell_size)
                workerY, workerX = [self.size // 2] * 2

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


def main():
    output_dir = "./dataset"
    os.makedirs(output_dir, exist_ok=True)
    csv_dir = "./field_data"
    annotator = Annotator(glob.glob(os.path.join(csv_dir, "[ABC]*.csv")), output_dir, 5)
    annotator.reset()
    annotator.do_annotate()


if __name__ == "__main__":
    main()
