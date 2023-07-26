import copy
import csv
import glob
import os
import random
import re
from collections import defaultdict
from typing import Iterable, Optional, Union

import gymnasium as gym
import numpy as np
import pyautogui
import pygame
from pygame.locals import *

try:
    from .worker import Worker
except:
    from worker import Worker


class Game(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}
    SCORE_MULTIPLIER = {"castle": 100, "position": 30, "rampart": 10}
    TEAM = ("A", "B")
    FIELD_MIN, FIELD_MAX = 11, 25
    WORKER_MIN, WORKER_MAX = 2, 6
    CELL = (
        "blank",
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
        csv_paths: Union[str, list[str]],
        render_mode="ansi",
        max_episode_steps=100,
        first_player: Optional[int] = None,
        controller: str = "cli",
    ):
        super().__init__()
        self.csv_paths = csv_paths
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.ACTIONS)] * self.WORKER_MAX
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(len(self.CELL), self.FIELD_MAX, self.FIELD_MAX),
            dtype=np.uint8,
        )
        self.reward_range = [np.NINF, np.inf]

        self.first_player = first_player
        self.controller = controller
        self.cwd = os.getcwd()
        self.display_size_x, self.display_size_y = pyautogui.size()

    def change_player(self, no_change: bool = False):
        """
        内部関数
        操作対象のチームを更新する
        Args:
            no_change (bool, optional): チームの変更は行わずに変数の更新のみを行う (default=False)
        """
        if not no_change:
            self.current_player = 1 - self.current_player
        self.current_team = self.TEAM[self.current_player]
        self.opponent_team = self.TEAM[1 - self.current_player]

    def update_blank(self):
        """
        内部関数
        blank層を更新
        """
        self.board[0] = np.where(self.board[1:].any(axis=0), 0, 1)
        self.board[0, self.height :, :] = -1
        self.board[0, :, self.width :] = -1

    def load_from_csv(self, path: str):
        """
        内部関数
        csvデータからフィールドを作成する
        Args:
            path (str): csvデータのパス
        """
        size = int(re.sub(r"[\D]", "", os.path.normpath(path).split(os.path.sep)[-1]))
        self.board = np.zeros((len(self.CELL), size, size), dtype=np.uint8)
        self.workers: defaultdict[str, list[Worker]] = defaultdict(list)
        self.width, self.height = [size] * 2

        a_count, b_count = 0, 0
        with open(path, "r") as f:
            reader = csv.reader(f)
            for y, row in enumerate(reader):
                for x, item in enumerate(row):
                    if item == "0":
                        self.board[self.CELL.index("blank"), y, x] = 1
                    elif item == "1":
                        self.board[self.CELL.index("pond"), y, x] = 1
                    elif item == "2":
                        self.board[self.CELL.index("castle"), y, x] = 1
                    elif item == "a":
                        self.board[self.CELL.index(f"worker_A{a_count}"), y, x] = 1
                        self.workers["A"].append(Worker(f"worker_A{a_count}", y, x))
                        a_count += 1
                    elif item == "b":
                        self.board[self.CELL.index(f"worker_B{b_count}"), y, x] = 1
                        self.workers["B"].append(Worker(f"worker_B{b_count}", y, x))
                        b_count += 1
        self.board = np.pad(
            self.board,
            [(0, 0), (0, self.FIELD_MAX - size), (0, self.FIELD_MAX - size)],
            "constant",
            constant_values=-1,
        )
        assert a_count == b_count, "チーム間の職人数が不一致"
        self.worker_count = a_count
        self.update_blank()

    def reset(self, seed=None, options=None):
        """
        gymの必須関数
        環境の初期化
        Args:
            csv_path (str): 使用するフィールドのcsvデータのパスを入力
            end_turn (int, optional): 終了ターン数を指定. Defaults to 500.
            first_player (Optional[int], optional): 先攻のチーム名を指定. Defaults to None.
        """
        super().reset(seed=seed, options=options)
        self.current_player = (
            self.TEAM.index(self.first_player)
            if self.first_player
            else np.random.randint(0, 2)
        )
        self.change_player(no_change=True)
        self.score_A, self.score_B = 0, 0
        self.previous_score_A, self.previous_score_B = 0, 0
        self.turn = 1
        self.reward = 0
        self.terminated = False
        self.truncated = False

        if isinstance(self.csv_paths, list):
            self.load_from_csv(random.choice(self.csv_paths))
        else:
            self.load_from_csv(self.csv_paths)

        self.cell_size = min(
            self.display_size_x * 0.9 // self.width,
            self.display_size_y * 0.8 // self.height,
        )
        self.window_size = max(self.width, self.height) * self.cell_size
        self.window_size_x = self.width * self.cell_size
        self.window_size_y = self.height * self.cell_size

        self.update_blank()
        return self.board, {}

    def compile_layers(self, *layers: tuple[str], one_hot: bool = True):
        """
        入力された層を合成した2次元配列を返す
        one_hot: bool 返り値の各要素を1,0のみにする (default=True)
        """
        compiled = np.sum(
            [self.board[self.CELL.index(layer)] for layer in layers],
            axis=0,
            dtype=np.uint8,
        )
        if one_hot:
            compiled = np.where(compiled, 1, 0)
        compiled[self.height :, :] = -1
        compiled[:, self.width :] = -1
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

    def check_stack_workers(self, workers: list[tuple[Worker, int]]):
        destinations = defaultdict(int)
        for worker, action in workers:
            if "move" in self.ACTIONS[action]:
                destinations[
                    tuple(
                        (
                            np.array(worker.get_coordinate())
                            + self.get_direction(action)
                        ).tolist(),
                    )
                ] += 1
        if any(value > 1 for value in destinations.values()):
            stack_destinations = [
                key for key, value in destinations.items() if value > 1
            ]
            for worker, action in workers.copy():
                if "move" in self.ACTIONS[action] and (
                    tuple(
                        (
                            np.array(worker.get_coordinate())
                            + self.get_direction(action)
                        ).tolist(),
                    )
                    in stack_destinations
                ):
                    # print(f"{worker.name}: 行動できませんでした。待機します。")
                    worker.stay()
                    self.successful.append(False)
                    workers.remove((worker, action))
        return workers

    def action_workers(self, workers: list[tuple[Worker, int]]):
        """
        内部関数
        職人を行動させる
        """
        workers = self.check_stack_workers(workers)
        if not workers:
            return
        for _ in range(self.worker_count):
            worker, action = workers.pop(0)
            y, x = map(
                int, np.array(worker.get_coordinate()) + self.get_direction(action)
            )
            if "stay" in self.ACTIONS[action] and self.is_movable(
                worker, worker.y, worker.x
            ):
                worker.stay()
                self.successful.append(False)

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
                # print(f"{worker.name}: 行動できませんでした。待機します。")
                worker.stay()
                self.successful.append(False)

            if not workers:
                break

        for worker, action in workers:
            # print(f"{worker.name}: 行動できませんでした。待機します。")
            worker.stay()
            self.successful.append(False)

    def fill_area(self, array: np.ndarray):
        """
        内部関数
        囲まれている領域を取得
        """
        array = np.where(array == 1, 0, 1)
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

        array = np.where(array == 1, 1, 0)
        array[:, self.width :] = -1
        array[self.height :, :] = -1
        return array

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
            (self.previous_position_A + self.previous_open_position_A),
            1,
            0,
        ) - self.compile_layers("rampart_A", "rampart_B", "position_A", "position_B")
        self.board[self.CELL.index("open_position_A")] = np.where(
            self.board[self.CELL.index("open_position_A")] == np.uint8(-1),
            0,
            self.board[self.CELL.index("open_position_A")],
        )
        self.board[self.CELL.index("open_position_A"), :, self.width :] = -1
        self.board[self.CELL.index("open_position_A"), self.height :, :] = -1

        self.board[self.CELL.index("open_position_B")] = np.where(
            (self.previous_position_B + self.previous_open_position_B),
            1,
            0,
        ) - self.compile_layers("rampart_A", "rampart_B", "position_A", "position_B")
        self.board[self.CELL.index("open_position_B")] = np.where(
            self.board[self.CELL.index("open_position_B")] == np.uint8(-1),
            0,
            self.board[self.CELL.index("open_position_B")],
        )
        self.board[self.CELL.index("open_position_B"), :, self.width :] = -1
        self.board[self.CELL.index("open_position_B"), self.height :, :] = -1

    def calculate_score(self):
        """
        内部関数
        得点を計算
        """
        self.previous_score_A, self.previous_score_B = self.score_A, self.score_B

        self.score_A = (
            np.sum(
                self.board[self.CELL.index("castle"), : self.height, : self.width]
                * self.compile_layers("position_A", "open_position_A")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_A += (
            np.sum(
                (1 - self.board[self.CELL.index("castle"), : self.height, : self.width])
                * self.compile_layers("position_A", "open_position_A")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_A += (
            np.sum(
                self.board[self.CELL.index("rampart_A"), : self.height, : self.width]
            )
            * self.SCORE_MULTIPLIER["rampart"]
        )

        self.score_B = (
            np.sum(
                self.board[self.CELL.index("castle"), : self.height, : self.width]
                * self.compile_layers("position_B", "open_position_B")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_B += (
            np.sum(
                (1 - self.board[self.CELL.index("castle"), : self.height, : self.width])
                * self.compile_layers("position_B", "open_position_B")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_B += (
            np.sum(
                self.board[self.CELL.index("rampart_B"), : self.height, : self.width]
            )
            * self.SCORE_MULTIPLIER["rampart"]
        )

        # print(f"score_A:{self.score_A}, score_B:{self.score_B}")

    def get_reward(self):
        """報酬更新処理実装予定"""
        reward = self.score_A - self.score_B
        reward += np.abs(reward) * (
            (self.score_A - self.previous_score_A)
            - (self.score_B - self.previous_score_B)
        )
        if self.score_A == self.previous_score_A:
            reward *= 0.75 if reward > 0 else 1.25
        if not all(self.successful):
            reward -= 10000
        return float(reward)

    def is_done(self):
        """ゲーム終了判定実装予定"""
        # terminated : エピソード終了フラグ
        # truncated  : ステップ数上限での終了フラグ
        if self.turn >= self.max_episode_steps:
            self.truncated = True
        # self.terminated = True

    def step(self, actions: Iterable[int]):
        """
        gymの必須関数
        1ターン進める処理を実行
        """
        current_workers = (
            self.workers["A"] if not self.current_player else self.workers["B"]
        )
        [worker.turn_init() for worker in current_workers]
        sorted_workers = [
            (worker, action)
            for worker, action in zip(current_workers, actions[: self.worker_count])
            if "break" in self.ACTIONS[action]
        ]
        sorted_workers += [
            (worker, action)
            for worker, action in zip(current_workers, actions[: self.worker_count])
            if "break" not in self.ACTIONS[action]
        ]

        self.successful = []
        self.action_workers(sorted_workers)
        self.update_position()
        self.update_open_position()
        self.update_blank()
        self.calculate_score()
        self.reward = self.get_reward()
        self.is_done()
        self.change_player()
        self.turn += 1
        return self.board, self.reward, self.terminated, self.truncated, {}

    def render(self):
        """
        gymの必須関数
        描画を行う
        mode: str("human" or "ansi") pygameかcliどちらで描画するか選択
        input_with: str("pygame" or "cli") pygame上で職人を操作するか、cli上で行動番号を入力するか選択
        """
        IMG_SCALER = np.array((self.cell_size, self.cell_size))
        BLANK_IMG = pygame.transform.scale(
            pygame.image.load(self.cwd + "/assets/blank.png"), IMG_SCALER
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
        if self.render_mode == "ansi":
            rendering = str([row for row in view])
            print(rendering)
            return rendering
        elif self.render_mode == "human":
            window_surface = pygame.display.set_mode(
                (self.window_size_x, self.window_size_y + self.cell_size * 2)
            )
            pygame.display.set_caption("game")

            drawAll()

            if self.controller != "pygame":
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
                        openPositionALayer = self.compile_layers(
                            "open_position_A", one_hot=True
                        )
                        openPositionBLayer = self.compile_layers(
                            "open_position_B", one_hot=True
                        )
                        for i in range(self.height):
                            for j in range(self.width):
                                if positionALayer[i][j] == 1:
                                    color = self.RED
                                elif positionBLayer[i][j] == 1:
                                    color = self.BLUE
                                elif openPositionALayer[i][j] == 1:
                                    color = self.PINK
                                elif openPositionBLayer[i][j] == 1:
                                    color = self.SKY
                                else:
                                    placeImage(BLANK_IMG, i, j)
                                    continue

                                pygame.draw.rect(
                                    window_surface,
                                    color,
                                    (
                                        j * self.cell_size,
                                        i * self.cell_size,
                                        self.cell_size,
                                        self.cell_size,
                                    ),
                                )
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
            self.actions = actions

    def get_actions(self):
        if self.controller == "pygame":
            return self.actions
        elif self.controller == "cli":
            [print(f"{i:2}: {action}") for i, action in enumerate(env.ACTIONS)]
            print(
                f"input team {env.current_team} actions (need {env.worker_count} input) : "
            )
            actions = [int(input()) for _ in range(env.worker_count)]
            return actions

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    fields = glob.glob(os.path.normpath("./field_data/*.csv"))

    env = Game(
        csv_paths=random.choice(fields), render_mode="human", controller="pygame"
    )

    observation = env.reset()
    terminated, truncated = [False] * 2
    print(f"width:{env.width}, height:{env.height}, workers:{env.worker_count}")

    while not terminated and not truncated:
        print(
            f"input team {env.current_team} actions (need {env.worker_count} input) : "
        )
        env.render()
        observation, reward, terminated, truncated, _ = env.step(env.get_actions())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
