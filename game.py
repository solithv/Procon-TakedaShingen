import os
import copy
from typing import Iterable, Optional, Union

import gymnasium as gym
import numpy as np
import pyautogui
import pygame
from pygame.locals import *

from worker import Worker


class Game(gym.Env):
    metadata = {"render.modes": ["human", "console"]}
    SCORE_MULTIPLIER = {"castle": 100, "position": 50, "rampart": 10}
    TEAM = ("A", "B")
    POND_MIN, POND_MAX = 1, 5
    FIELD_MIN, FIELD_MAX = 11, 25
    WORKER_MIN, WORKER_MAX = 2, 6
    CELL = (
        "blank",  # 論理反転
        "position_A",
        "position_B",
        "open_position_A",
        "open_position_B",
        "rampart_A",
        "rampart_B",
        "castle",
        "pond",
        *[f"worker_A{i}" for i in range(WORKER_MAX)],
        *[f"worker_B{i}" for i in range(WORKER_MAX)],
    )
    ACTIONS = (
        "stay",
        "move_N",
        "move_NE",
        "move_E",
        "move_SE",
        "move_S",
        "move_SW",
        "move_W",
        "move_NW",
        "build_N",
        "build_E",
        "build_S",
        "build_W",
        "break_N",
        "break_E",
        "break_S",
        "break_W",
    )
    DIRECTIONS = {
        # [y, x]
        "N": np.array([-1, 0]),
        "E": np.array([0, 1]),
        "S": np.array([1, 0]),
        "W": np.array([0, -1]),
    }
    SCORE_MULTIPLIER = {"castle": 100, "position": 50, "rampart": 10}
    CASTLE_MIN, CASTLE_MAX = 1, 1
    POND_MIN, POND_MAX = 1, 5
    FIELD_MIN, FIELD_MAX = 11, 25
    WORKER_MIN, WORKER_MAX = 2, 6

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    SKY = (127, 176, 255)
    PINK = (255, 127, 127)

    def __init__(
        self,
        end_turn: int = 10,
        width: Optional[int] = None,
        height: Optional[int] = None,
        castle: Optional[int] = None,
        pond: Optional[int] = None,
        worker: Optional[int] = None,
        controller: str = "cli"
    ):
        super().__init__()
        self.end_turn = end_turn * 2
        self.width = width or np.random.randint(self.FIELD_MIN, self.FIELD_MAX)
        self.height = height or np.random.randint(self.FIELD_MIN, self.FIELD_MAX)

        self.castle_count = (
            castle or self.CASTLE_MIN
            if self.CASTLE_MIN == self.CASTLE_MAX
            else np.random.randint(self.CASTLE_MIN, self.CASTLE_MAX)
        )
        self.pond_count = pond or np.random.randint(self.POND_MIN, self.POND_MAX)
        self.worker_count = worker or np.random.randint(
            self.WORKER_MIN, self.WORKER_MAX
        )
        self.controller = controller
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.CELL), self.height, self.width),
            dtype=np.int8,
        )
        self.reward_range = [np.NINF, np.inf]
        self.cwd = os.getcwd()
        self.display_size_x, self.display_size_y = pyautogui.size()
        self.cell_size = min(
            self.display_size_x * 0.9 // self.width,
            self.display_size_y * 0.8 // self.height,
        )
        self.window_size = max(self.width, self.height) * self.cell_size
        self.window_size_x = self.width * self.cell_size
        self.window_size_y = self.height * self.cell_size

        self.reset()

    def change_player(self, no_change: bool = False):
        if not no_change:
            self.current_player = 1 - self.current_player
        self.current_team = self.TEAM[self.current_player]
        self.opponent_team = self.TEAM[1 - self.current_player]

    def set_cell_property(
        self, target: str, coordinate: Optional[Iterable[int]] = None
    ):
        """
        内部関数
        セルに要素を配置
        """
        if not coordinate:
            while True:
                y = np.random.randint(0, self.height - 1)
                x = np.random.randint(0, self.width - 1)
                if (y, x) not in self.used:
                    break
        else:
            y, x = coordinate
        self.board[self.CELL.index(target), y, x] = 1
        self.used.append((y, x))
        return y, x

    def set_worker_position(
        self, target: str, coordinates: Optional[Iterable[int]] = None
    ):
        """
        内部関数
        セルに職人を配置
        """
        y, x = self.set_cell_property(target, coordinates)
        return Worker(target, y, x)

    def update_blank(self):
        """
        内部関数
        blank層を更新
        """
        self.board[0] = 1 - self.board[1:].any(axis=0)

    def reset(
        self,
        first_player: Optional[int] = None,
        castle: Optional[Iterable[Iterable[int]]] = None,
        pond: Optional[Iterable[Iterable[int]]] = None,
        worker_A: Optional[Iterable[Iterable[int]]] = None,
        worker_B: Optional[Iterable[Iterable[int]]] = None,
    ):
        """
        gymの必須関数
        環境の初期化
        first_player: str("A" or "B") 先攻のチーム名を指定
        castle: list[y,x] 城の座標を指定
        pond: list[list[y,x]] 池の座標を指定
        worker_A: list[list[y,x]] Aチームの職人の座標を指定
        worker_B: list[list[y,x]] Aチームの職人の座標を指定
        """
        self.current_player = (
            self.TEAM.index(first_player) if first_player else np.random.randint(0, 2)
        )
        self.change_player(no_change=True)
        self.score_A, self.score_B = 0, 0
        self.previous_score_A, self.previous_score_B = 0, 0
        self.turn = 1
        self.done = False
        self.board = np.zeros((len(self.CELL), self.height, self.width))
        self.workers: dict[str, Worker] = {}
        self.used = []

        if castle:
            assert self.castle_count == len(castle), "castle input error"
            [
                self.set_cell_property("castle", coordinate=coordinate)
                for coordinate in castle
            ]
        else:
            [self.set_cell_property("castle") for _ in range(self.castle_count)]

        if pond:
            assert self.pond_count == len(pond), "pond input error"
            [
                self.set_cell_property("pond", coordinate=coordinate)
                for coordinate in pond
            ]
        else:
            [self.set_cell_property("pond") for _ in range(self.pond_count)]

        if worker_A:
            assert self.worker_count == len(worker_A), "worker_A input error"
        self.workers["A"] = (
            [
                self.set_worker_position(f"worker_A{i}", coordinate)
                for i, coordinate in enumerate(worker_A)
            ]
            if worker_A
            else [
                self.set_worker_position(f"worker_A{i}")
                for i in range(self.worker_count)
            ]
        )

        if worker_B:
            assert self.worker_count == len(worker_B), "worker_B input error"
        self.workers["B"] = (
            [
                self.set_worker_position(f"worker_B{i}", coordinate)
                for i, coordinate in enumerate(worker_B)
            ]
            if worker_B
            else [
                self.set_worker_position(f"worker_B{i}")
                for i in range(self.worker_count)
            ]
        )

        self.update_blank()
        return self.board

    def compile_layers(self, *layers: tuple[str], one_hot: bool = True):
        """
        入力された層を合成した2次元配列を返す
        one_hot: bool 返り値の各要素を1,0のみにする (default=False)
        """
        compiled = np.sum(
            [self.board[self.CELL.index(layer)] for layer in layers], axis=0
        )
        if one_hot:
            return np.where(compiled, 1, 0)
        else:
            return compiled

    def get_team_worker_coordinate(
        self, team: str, actioned: bool = True, worker: Worker = None
    ):
        """
        内部関数
        行動済みの職人の座標を取得
        team: str("A" or "B") 取得するチームを指定
        """
        if actioned:
            result = [
                worker.get_coordinate()
                for worker in self.workers[team]
                if worker.is_action
            ]
        else:
            result = [
                worker.get_coordinate()
                for worker in self.workers[team]
                if not worker.is_action
            ]
        if worker:
            result.remove(worker.get_coordinate())
        return result

    def is_movable(self, worker: Worker, y: int, x: int):
        """
        内部関数
        行動可能判定
        """
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and not self.compile_layers(
                f"rampart_{worker.opponent_team}",
                "pond",
                *[f"worker_{worker.opponent_team}{i}" for i in range(self.WORKER_MAX)],
            )[y, x]
            and (y, x) not in self.get_team_worker_coordinate(worker.team)
        ):
            return True
        else:
            return False

    def is_buildable(self, worker: Worker, y: int, x: int):
        """
        内部関数
        建築可能判定
        """
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and not self.compile_layers(
                "rampart_A",
                "rampart_B",
                "castle",
                *[f"worker_{worker.opponent_team}{i}" for i in range(self.WORKER_MAX)],
            )[y, x]
        ):
            return True
        else:
            return False

    def is_breakable(self, worker: Worker, y: int, x: int):
        """
        内部関数
        破壊可能判定
        """
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and self.compile_layers("rampart_A", "rampart_B")[y, x]
        ):
            return True
        else:
            return False

    def get_direction(self, action: int):
        """
        内部関数
        入力行動に対する方向を取得
        """
        direction = np.zeros(2)
        for key, value in self.DIRECTIONS.items():
            if key in self.ACTIONS[action]:
                direction += value
        return direction

    def action_workers(self, workers: Iterable[tuple[Worker, int]]):
        """
        内部関数
        職人を行動させる
        """
        for _ in range(self.worker_count):
            worker, action = workers.pop(0)
            y, x = map(
                int, np.array(worker.get_coordinate()) + self.get_direction(action)
            )
            if "stay" in self.ACTIONS[action] and self.is_movable(
                worker, worker.y, worker.x
            ):
                worker.stay()
                self.successful.append(True)

            elif "move" in self.ACTIONS[action] and self.is_movable(worker, y, x):
                if (y, x) in self.get_team_worker_coordinate(
                    worker.team, actioned=False, worker=worker
                ):
                    workers.append((worker, action))
                    continue
                self.board[self.CELL.index(worker.name), worker.y, worker.x] = 0
                self.board[self.CELL.index(worker.name), y, x] = 1
                worker.move(y, x)
                self.successful.append(True)

            elif "build" in self.ACTIONS[action] and self.is_buildable(worker, y, x):
                self.board[self.CELL.index(f"rampart_{worker.team}"), y, x] = 1
                worker.build(y, x)
                self.successful.append(True)

            elif "break" in self.ACTIONS[action] and self.is_breakable(worker, y, x):
                if self.board[self.CELL.index("rampart_A"), y, x]:
                    self.board[self.CELL.index("rampart_A"), y, x] = 0
                else:
                    self.board[self.CELL.index("rampart_B"), y, x] = 0
                worker.break_(y, x)
                self.successful.append(True)

            else:
                print("行動できない入力です")
                worker.stay()
                self.successful.append(False)

            if not workers:
                break

        for worker, action in workers:
            print(f"{worker.name}: 行動できませんでした。待機します。")
            worker.stay()
            self.successful.append(False)

    def fill_area(self, array: np.ndarray):
        """
        内部関数
        囲まれている領域を取得
        """
        array = np.where(array, 0, 1)
        # 配列の形状を取得
        rows, cols = array.shape

        def dfs(row, col):
            # 配列の外周に達した場合、再帰を終了
            if not (0 <= row < rows and 0 <= col < cols):
                return

            # すでに探索済みの要素や外周の要素は処理しない
            if array[row, col] != 1:
                return

            # 1を2に置換
            array[row, col] = 2

            # 上下左右の要素を再帰的に探索
            dfs(row - 1, col)  # 上
            dfs(row + 1, col)  # 下
            dfs(row, col - 1)  # 左
            dfs(row, col + 1)  # 右

        # 外周の上下左右の要素を探索
        for i in range(cols):
            dfs(0, i)  # 上辺
            dfs(rows - 1, i)  # 下辺
        for i in range(rows):
            dfs(i, 0)  # 左辺
            dfs(i, cols - 1)  # 右辺

        return np.where(array == 1, 1, 0)

    def update_position(self):
        """
        内部関数
        陣地を更新
        """
        self.previous_position_A = copy.deepcopy(
            self.board[self.CELL.index("position_A")]
        )
        self.previous_position_B = copy.deepcopy(
            self.board[self.CELL.index("position_B")]
        )
        self.board[self.CELL.index("position_A")] = self.fill_area(
            self.board[self.CELL.index("rampart_A")]
        )
        self.board[self.CELL.index("position_B")] = self.fill_area(
            self.board[self.CELL.index("rampart_B")]
        )

    def update_open_position(self):
        """
        内部関数
        開放陣地を更新
        """
        self.previous_open_position_A = copy.deepcopy(
            self.board[self.CELL.index("open_position_A")]
        )
        self.previous_open_position_B = copy.deepcopy(
            self.board[self.CELL.index("open_position_B")]
        )
        self.board[self.CELL.index("open_position_A")] = np.where(
            np.where(
                (
                    self.previous_position_A
                    - self.board[self.CELL.index("position_A")]
                    + self.previous_open_position_A
                )
                > 0,
                1,
                0,
            )
            - self.compile_layers("rampart_B", "position_B")
            > 0,
            1,
            0,
        )
        self.board[self.CELL.index("open_position_B")] = np.where(
            np.where(
                (
                    self.previous_position_B
                    - self.board[self.CELL.index("position_B")]
                    + self.previous_open_position_B
                )
                > 0,
                1,
                0,
            )
            - self.compile_layers("rampart_A", "position_A")
            > 0,
            1,
            0,
        )

    def calculate_score(self):
        """
        内部関数
        得点を計算
        """
        self.previous_score_A, self.previous_score_B = self.score_A, self.score_B

        self.score_A = (
            np.sum(
                self.board[self.CELL.index("castle")]
                * self.compile_layers("position_A", "open_position_A")
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_A += (
            np.sum(
                (1 - self.board[self.CELL.index("castle")])
                * self.compile_layers("position_A", "open_position_A")
            )
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_A += (
            np.sum(self.board[self.CELL.index("rampart_A")])
            * self.SCORE_MULTIPLIER["rampart"]
        )

        self.score_B = (
            np.sum(
                self.board[self.CELL.index("castle")]
                * self.compile_layers("position_B", "open_position_B")
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_B += (
            np.sum(
                (1 - self.board[self.CELL.index("castle")])
                * self.compile_layers("position_B", "open_position_B")
            )
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_B += (
            np.sum(self.board[self.CELL.index("rampart_B")])
            * self.SCORE_MULTIPLIER["rampart"]
        )

        print(f"score_A:{self.score_A}, score_B:{self.score_B}")

    def get_reward(self):
        """報酬更新処理実装予定"""
        if not all(self.successful):
            return np.NINF

    def is_done(self):
        """ゲーム終了判定実装予定"""
        if self.turn >= self.end_turn:
            self.done = True

    def step(self, actions: Iterable[int]):
        """
        gymの必須関数
        1ターン進める処理を実行
        """
        assert self.worker_count == len(
            actions
        ), f"input length error {self.worker_count}, {len(actions)}"
        current_workers = (
            self.workers["A"] if not self.current_player else self.workers["B"]
        )
        [worker.turn_init() for worker in current_workers]
        sorted_workers = [
            (worker, action)
            for worker, action in zip(current_workers, actions)
            if "break" in self.ACTIONS[action]
        ]
        sorted_workers += [
            (worker, action)
            for worker, action in zip(current_workers, actions)
            if "break" not in self.ACTIONS[action]
        ]

        self.successful = []
        self.action_workers(sorted_workers)
        self.update_position()
        self.update_open_position()
        self.update_blank()
        print(
            self.compile_layers(
                f"position_{self.current_team}", f"open_position_{self.current_team}"
            )
        )
        self.change_player()
        self.turn += 1
        self.calculate_score()
        reward = self.get_reward()
        self.is_done()
        return self.board, reward, self.done, {}

    def render(self, mode: str = "human", input_with: str = "cli"):
        """
        gymの必須関数
        描画を行う
        mode: str("human" or "console") pygameかcliどちらで描画するか選択
        input_with: str("pygame" or "cli") pygame上で職人を操作するか、cli上で行動番号を入力するか選択
        """
        IMG_SCALER = np.array((self.cell_size, self.cell_size))
        BLANK_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/blank.png"), IMG_SCALER
        )
        UNAVAILABLE_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/unavailable.png"), IMG_SCALER
        )
        POND_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/pond.png"), IMG_SCALER
        )
        CASTLE_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/castle.png"), IMG_SCALER
        )
        RAMPART_A_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/rampart_A.png"), IMG_SCALER
        )
        RAMPART_B_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/rampart_B.png"), IMG_SCALER
        )
        WORKER_A_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_A.png"), IMG_SCALER
        )
        WORKER_B_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_B.png"), IMG_SCALER
        )
        WORKER_A_HOVER_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_A_hover.png"), IMG_SCALER
        )
        WORKER_B_HOVER_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/worker_B_hover.png"), IMG_SCALER
        )

        def drawGrids():
            # 縦線描画
            for i in range(1, self.width):
                pygame.draw.line(
                    window_surface,
                    self.BLACK,
                    (i * self.cell_size, 0),
                    (i * self.cell_size, self.window_size_y),
                    1,
                )
            # 横線描画
            for i in range(1, self.height):
                pygame.draw.line(
                    window_surface,
                    self.BLACK,
                    (0, i * self.cell_size),
                    (self.window_size_x, i * self.cell_size),
                    1,
                )

        def placeImage(img, i, j, workerNumber=None, scale=1.0):
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
            img = pygame.transform.scale(img, IMG_SCALER * scale)
            window_surface.blit(img, placement)

            if workerNumber:
                font = pygame.font.SysFont(None, 30)
                text = font.render(workerNumber, False, self.BLACK)
                text_rect = text.get_rect(
                    center=((j + 0.5) * self.cell_size, (i + 0.2) * self.cell_size)
                )
                window_surface.blit(text, text_rect)

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

        def drawAll():
            for i in range(self.height):
                for j in range(self.width):
                    placeImage(BLANK_IMG, i, j)
                    cellInfo = view[i][j]
                    worker_A_exist = any(
                        f"worker_A{k}" in cellInfo for k in range(self.WORKER_MAX)
                    )
                    worker_B_exist = any(
                        f"worker_B{k}" in cellInfo for k in range(self.WORKER_MAX)
                    )

                    if "castle" in cellInfo and worker_A_exist:
                        placeImage(CASTLE_IMG, i, j)
                        placeImage(
                            WORKER_A_IMG, i, j, workerNumber=cellInfo[-1][-1], scale=0.7
                        )
                    elif "castle" in cellInfo and worker_B_exist:
                        placeImage(CASTLE_IMG, i, j)
                        placeImage(
                            WORKER_B_IMG, i, j, workerNumber=cellInfo[-1][-1], scale=0.7
                        )
                    elif "rampart_A" in cellInfo and worker_A_exist:
                        placeImage(RAMPART_A_IMG, i, j)
                        placeImage(
                            WORKER_A_IMG, i, j, workerNumber=cellInfo[-1][-1], scale=0.8
                        )
                    elif "rampart_B" in cellInfo and worker_B_exist:
                        placeImage(RAMPART_B_IMG, i, j)
                        placeImage(
                            WORKER_B_IMG, i, j, workerNumber=cellInfo[-1][-1], scale=0.8
                        )
                    elif "pond" in cellInfo and "rampart_A" in cellInfo:
                        placeImage(POND_IMG, i, j)
                        placeImage(RAMPART_A_IMG, i, j, scale=0.8)
                    elif "pond" in cellInfo and "rampart_B" in cellInfo:
                        placeImage(POND_IMG, i, j)
                        placeImage(RAMPART_B_IMG, i, j, scale=0.8)
                    elif "castle" in cellInfo:
                        placeImage(CASTLE_IMG, i, j)
                    elif worker_A_exist:
                        placeImage(WORKER_A_IMG, i, j, workerNumber=cellInfo[-1][-1])
                    elif worker_B_exist:
                        placeImage(WORKER_B_IMG, i, j, workerNumber=cellInfo[-1][-1])
                    elif "pond" in cellInfo:
                        placeImage(POND_IMG, i, j)
                    elif "rampart_A" in cellInfo:
                        placeImage(RAMPART_A_IMG, i, j)
                    elif "rampart_B" in cellInfo:
                        placeImage(RAMPART_B_IMG, i, j)
                    elif "blank" in cellInfo:
                        placeImage(BLANK_IMG, i, j)
            drawGrids()

        def drawTurnInfo(actingWorker=None):
            pygame.draw.rect(
                window_surface,
                self.BLACK,
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
                    self.WHITE,
                )
            else:
                text = font.render(f"{self.current_team}'s turn", False, self.WHITE)

            text_rect = text.get_rect(
                center=(
                    self.cell_size * self.width / 2,
                    self.cell_size * (self.height + 1),
                )
            )
            window_surface.blit(text, text_rect)
            pygame.display.update()

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
        pygame.init()
        if mode == "console":
            [print(row) for row in view]
        elif mode == "human":
            window_surface = pygame.display.set_mode(
                (self.window_size_x, self.window_size_y + self.cell_size * 2)
            )
            pygame.display.set_caption("game")

            drawAll()
            print(self.compile_layers("position_A", one_hot=True))

            if input_with != "pygame":
                drawTurnInfo()
                return
            showPosition = False
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
                            if showPosition:
                                drawAll()
                            showPosition = not showPosition

                    if showPosition:
                        positionALayer = self.compile_layers("position_A", one_hot=True)
                        positionBLayer = self.compile_layers("position_B", one_hot=True)
                        for i in range(self.height):
                            for j in range(self.width):
                                if positionALayer[i][j] == 1:
                                    pygame.draw.rect(
                                        window_surface,
                                        self.RED,
                                        (
                                            j * self.cell_size,
                                            i * self.cell_size,
                                            self.cell_size,
                                            self.cell_size,
                                        ),
                                    )
                                elif positionBLayer[i][j] == 1:
                                    pygame.draw.rect(
                                        window_surface,
                                        self.BLUE,
                                        (
                                            j * self.cell_size,
                                            i * self.cell_size,
                                            self.cell_size,
                                            self.cell_size,
                                        ),
                                    )
                                else:
                                    placeImage(BLANK_IMG, i, j)
                        drawGrids()
                        continue

                    placeImage(
                        eval(f"WORKER_{self.current_team}_HOVER_IMG"),
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
                            directionVector = np.array(
                                [cellX - workerX, workerY - cellY]
                            )
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
                            placeImage(BLANK_IMG, workerY, workerX)
                            placeImage(
                                eval(f"WORKER_{self.current_team}_IMG"),
                                cellY,
                                cellX,
                                workerNumber=str(actingWorker),
                            )
                            drawGrids()
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
                            directionVector = np.array(
                                [cellX - workerX, workerY - cellY]
                            )
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
                            placeImage(BLANK_IMG, workerY, workerX)
                            placeImage(
                                eval(f"WORKER_{self.current_team}_IMG"),
                                workerY,
                                workerX,
                                workerNumber=str(actingWorker - 1),
                            )
                            placeImage(
                                eval(f"RAMPART_{self.current_team}_IMG"), cellY, cellX
                            )
                            drawGrids()
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
                            directionVector = np.array(
                                [cellX - workerX, workerY - cellY]
                            )
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

                            placeImage(BLANK_IMG, workerY, workerX)
                            placeImage(
                                eval(f"WORKER_{self.current_team}_IMG"),
                                workerY,
                                workerX,
                                workerNumber=str(actingWorker - 1),
                            )
                            placeImage(BLANK_IMG, cellY, cellX)
                            drawGrids()
                            pygame.display.update()
                drawTurnInfo(actingWorker=actingWorker + 1)
            return actions


if __name__ == "__main__":

    def turn():
        env.render(input_with="cli")

        [print(f"{i:2}: {action}") for i, action in enumerate(env.ACTIONS)]
        print(
            f"input team {env.current_team} actions (need {env.worker_count} input) : "
        )
        actions = [int(input()) for _ in range(env.worker_count)]

        return env.step(actions)

    env = Game(controller="pygame")

    print(f"width:{env.width}, height:{env.height}, workers:{env.worker_count}")
    observation = env.reset()
    done = False

    if env.controller == "pygame":
        while not done:
            print(
                f"input team {env.current_team} actions (need {env.worker_count} input) : "
            )
            actions = env.render(input_with="pygame")
            observation, reward, done, _ = env.step(actions)

            print(
                f"input team {env.current_team} actions (need {env.worker_count} input) : "
            )
            actions = env.render(input_with="pygame")
            observation, reward, done, _ = env.step(actions)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
    elif env.controller == "cli":
        while not done:
            observation, reward, done, _ = turn()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
