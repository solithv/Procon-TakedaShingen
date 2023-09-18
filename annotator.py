from collections import defaultdict
import csv
import glob
import os
import random
import re
from typing import Union
import numpy as np
import pygame
from pygame.locals import *

from MyEnv import Game


class Annotator:
    def __init__(self, csv_paths, size: int = 3) -> None:
        self.csv_paths = csv_paths
        self.size = size
        self.env = Game()
        self.env.reset()
        self.window_surface = None
        self.clock = None
        self.cwd = os.getcwd()
        self.display_size_x, self.display_size_y = 400, 400

    def reset(self):
        self.load_from_csv(self.csv_paths)
        self.cell_size = min(
            self.display_size_x * 0.9 // self.size,
            self.display_size_y * 0.8 // self.size,
        )
        self.window_size = self.size * self.cell_size
        self.window_size_x = self.size * self.cell_size
        self.window_size_y = self.size * self.cell_size
        self.reset_render()

    def load_from_csv(self, path: Union[str, list[str]]):
        if isinstance(path, (list, tuple)):
            path = random.choice(path)
        size = int(re.sub(r"[\D]", "", os.path.normpath(path).split(os.path.sep)[-1]))
        self.board = np.zeros((len(self.env.CELL), size, size), dtype=np.int8)
        self.width, self.height = [size] * 2

        a_count = 0
        with open(path, "r") as f:
            reader = csv.reader(f)
            for y, row in enumerate(reader):
                for x, item in enumerate(row):
                    if item == "0":
                        self.board[self.env.CELL.index("blank"), y, x] = 1
                    elif item == "1":
                        self.board[self.env.CELL.index("pond"), y, x] = 1
                    elif item == "2":
                        self.board[self.env.CELL.index("castle"), y, x] = 1
                    elif item == "a":
                        self.board[self.env.CELL.index(f"worker_A{a_count}"), y, x] = 1
                        a_count += 1
        self.worker_count = a_count
        self.update_blank()

    def update_blank(self):
        self.board[self.env.CELL.index("blank")] = np.where(
            self.board[1:].any(axis=0), 0, 1
        )
        self.board[self.env.CELL.index("blank"), self.height :, :] = -1
        self.board[self.env.CELL.index("blank"), :, self.width :] = -1

    def get_around(self, y: int, x: int, side_length: int = 3):
        if side_length % 2 == 0:
            raise ValueError("need to input an odd number")
        length_ = side_length // 2
        field = np.pad(
            self.board,
            [(0, 0), (length_, 0), (length_, 0)],
            "constant",
            constant_values=-1,
        )
        front = length_ * 2 + 1
        return field[:, y : y + front, x : x + front]

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

    def drawTurnInfo(self, actingWorker=None):
        pygame.draw.rect(
            self.window_surface,
            self.env.BLACK,
            (
                0,
                self.cell_size * self.height,
                self.cell_size * self.width,
                self.cell_size * 2,
            ),
        )
        font = pygame.font.SysFont(None, 60)
        if actingWorker:
            text = font.render(
                f"{self.current_team}'s turn {actingWorker}/{self.worker_count}",
                False,
                self.env.WHITE,
            )
        else:
            text = font.render(f"{self.current_team}'s turn", False, self.env.WHITE)

        text_rect = text.get_rect(
            center=(
                self.cell_size * self.width / 2,
                self.cell_size * (self.height + 1),
            )
        )
        self.window_surface.blit(text, text_rect)
        pygame.display.update()

    def render_rgb_array(self):
        """
        描画を行う
        """

        view = [
            [
                [
                    self.CELL[i]
                    for i, item in enumerate(self.board[:, y, x].astype(bool))
                    if item
                ]
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            self.window_surface = pygame.display.set_mode(
                (self.window_size_x, self.window_size_y + self.cell_size * 2)
            )
            pygame.display.set_caption("game")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.drawAll(view)
        self.clock.tick(self.metadata["render_fps"])

        self.drawTurnInfo()
        # if self.controller != "pygame":
        #     return

    def get_actions_from_pygame(self):
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

        view = [
            [
                [
                    self.CELL[i]
                    for i, item in enumerate(self.board[:, y, x].astype(bool))
                    if item
                ]
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]
        showTerritory = False
        actions = []
        actingWorker = 0
        while actingWorker < self.worker_count:
            for event in pygame.event.get():
                if actingWorker >= self.worker_count:
                    break
                mouseX, mouseY = pygame.mouse.get_pos()
                cellX = int(mouseX // self.cell_size)
                cellY = int(mouseY // self.cell_size)
                workerY, workerX = self.workers[self.current_team][
                    actingWorker
                ].get_coordinate()

                if event.type == KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if showTerritory:
                            self.drawAll(view)
                        showTerritory = not showTerritory

                self.placeImage(
                    eval(f"self.WORKER_{self.current_team}_HOVER_IMG"),
                    workerY,
                    workerX,
                    workerNumber=str(actingWorker),
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
                            actions.append(0)
                        else:
                            actions.append(
                                int(
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
                            )
                        self.placeImage(self.BLANK_IMG, workerY, workerX)
                        self.placeImage(
                            eval(f"self.WORKER_{self.current_team}_IMG"),
                            cellY,
                            cellX,
                            workerNumber=str(actingWorker),
                        )
                        self.drawGrids()
                        actingWorker += 1
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
                        actions.append(
                            int(
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
                        )
                        actingWorker += 1
                        self.placeImage(self.BLANK_IMG, workerY, workerX)
                        self.placeImage(
                            eval(f"self.WORKER_{self.current_team}_IMG"),
                            workerY,
                            workerX,
                            workerNumber=str(actingWorker - 1),
                        )
                        self.placeImage(
                            eval(f"self.RAMPART_{self.current_team}_IMG"), cellY, cellX
                        )
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
                        actions.append(
                            int(
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
                        )
                        actingWorker += 1

                        self.placeImage(self.BLANK_IMG, workerY, workerX)
                        self.placeImage(
                            eval(f"self.WORKER_{self.current_team}_IMG"),
                            workerY,
                            workerX,
                            workerNumber=str(actingWorker - 1),
                        )
                        self.placeImage(self.BLANK_IMG, cellY, cellX)
                        self.drawGrids()
                        pygame.display.update()
            self.drawTurnInfo(actingWorker=actingWorker + 1)
        return actions


def main():
    csv_dir = "./field_data"
    annotator = Annotator(glob.glob(csv_dir, "*.csv"))


if __name__ == "__main__":
    main()
