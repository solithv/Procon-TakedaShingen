import copy
import csv
import glob
import os
import random
import re
from collections import defaultdict
from typing import Union, Optional

import gymnasium as gym
import numpy as np
import pygame
from pygame.locals import *

try:
    from .worker import Worker
except:
    from worker import Worker


class Game(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}
    SCORE_MULTIPLIER = {"castle": 100, "territory": 30, "rampart": 10}
    TEAM = ("A", "B")
    FIELD_MIN, FIELD_MAX = 11, 25
    WORKER_MIN, WORKER_MAX = 2, 6
    CELL = (
        "blank",
        "territory_A",
        "territory_B",
        "open_territory_A",
        "open_territory_B",
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
    ICONS = {
        "blank": "  ",
        "castle": "C ",
        "pond": "P ",
        "territory_A": "Ta",
        "territory_B": "Tb",
        "open_territory_A": "ta",
        "open_territory_B": "tb",
        "rampart_A": "Ra",
        "rampart_B": "Rb",
        "worker_A": "Wa",
        "worker_B": "Wb",
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
        csv_path: Union[str, list[str]],
        render_mode="ansi",
        max_episode_steps=100,
        first_player: Optional[int] = None,
        controller: str = "cli",
        use_pyautogui: bool = False,
    ):
        """init

        Args:
            csv_path (Union[str, list[str]]): フィールドデータのパス
            render_mode (str, optional): 描画方法. Defaults to "ansi".
            max_episode_steps (int, optional): 最大ステップ数. Defaults to 100.
            first_player (Optional[int], optional): 先行プレイヤーの番号. Defaults to None.
            controller (str, optional): 操作入力方法. Defaults to "cli".
        """
        super().__init__()
        self.csv_path = csv_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.first_player = first_player
        self.controller = controller

        self.cwd = os.getcwd()
        if use_pyautogui:
            import pyautogui

            self.display_size_x, self.display_size_y = pyautogui.size()
        else:
            self.display_size_x, self.display_size_y = 960, 960

        self.action_space = gym.spaces.Tuple(
            gym.spaces.Discrete(len(self.ACTIONS)) for _ in range(self.WORKER_MAX)
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(len(self.CELL), self.FIELD_MAX, self.FIELD_MAX),
            dtype=np.uint8,
        )
        self.reward_range = [np.NINF, np.inf]

    def get_observation(self):
        """状態を整形して観測空間として返す"""
        return self.board

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

    def load_from_csv(self, path: Union[str, list[str]]):
        """
        内部関数
        csvデータからフィールドを作成する
        Args:
            path (Union[str, list[str]]): csvデータのパス
        """
        if isinstance(path, (list, tuple)):
            path = random.choice(path)
        size = int(re.sub(r"[\D]", "", os.path.normpath(path).split(os.path.sep)[-1]))
        name = os.path.normpath(path).split(os.path.sep)[-1].split(".")[0]
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

        return name

    def reset(self, seed=None, options=None):
        """
        gymの必須関数
        環境の初期化
        """
        super().reset(seed=seed, options=options)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.current_player = (
            self.first_player
            if self.first_player is not None
            else np.random.randint(0, 2)
        )
        self.change_player(no_change=True)
        self.score_A, self.score_B = 0, 0
        self.previous_score_A, self.previous_score_B = 0, 0
        self.turn = 1
        name = self.load_from_csv(self.csv_path)

        self.cell_size = min(
            self.display_size_x * 0.9 // self.width,
            self.display_size_y * 0.8 // self.height,
        )
        self.window_size = max(self.width, self.height) * self.cell_size
        self.window_size_x = self.width * self.cell_size
        self.window_size_y = self.height * self.cell_size

        self.update_blank()
        info = {"csv_name": name}
        return self.get_observation(), info

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
            and (y, x) not in self.worker_positions
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
        """移動先が競合している職人を待機させる

        Args:
            workers (list[tuple[Worker, int]]): 職人と行動のリスト

        Returns:
            list[tuple[Worker, int]]: 職人と行動のリスト
        """
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
                    self.stayed_workers.append(worker.name)
                    workers.remove((worker, action))
        return workers

    def action_workers(self, workers: list[tuple[Worker, int]]):
        """
        内部関数
        職人を行動させる
        """
        self.worker_positions = [worker.get_coordinate() for worker, _ in workers]
        workers = self.check_stack_workers(workers)
        if not workers:
            return
        for _ in range(self.worker_count):
            worker, action = workers.pop(0)
            y, x = map(
                int, np.array(worker.get_coordinate()) + self.get_direction(action)
            )
            if "stay" in self.ACTIONS[action]:
                worker.stay()
                self.successful.append(False)
                self.stayed_workers.append(worker.name)

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
                self.stayed_workers.append(worker.name)

            if not workers:
                break

        for worker, action in workers:
            # print(f"{worker.name}: 行動できませんでした。待機します。")
            worker.stay()
            self.successful.append(False)
            self.stayed_workers.append(worker.name)

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

    def update_territory(self):
        """
        内部関数
        陣地を更新
        """
        self.previous_territory_A = copy.deepcopy(
            self.board[self.CELL.index("territory_A")]
        )
        self.previous_territory_B = copy.deepcopy(
            self.board[self.CELL.index("territory_B")]
        )
        self.board[self.CELL.index("territory_A")] = self.fill_area(
            self.board[self.CELL.index("rampart_A")]
        )
        self.board[self.CELL.index("territory_B")] = self.fill_area(
            self.board[self.CELL.index("rampart_B")]
        )

    def update_open_territory(self):
        """
        内部関数
        開放陣地を更新
        """
        self.previous_open_territory_A = copy.deepcopy(
            self.board[self.CELL.index("open_territory_A")]
        )
        self.previous_open_territory_B = copy.deepcopy(
            self.board[self.CELL.index("open_territory_B")]
        )

        self.board[self.CELL.index("open_territory_A")] = np.where(
            (self.previous_territory_A + self.previous_open_territory_A),
            1,
            0,
        ) - self.compile_layers("rampart_A", "rampart_B", "territory_A", "territory_B")
        self.board[self.CELL.index("open_territory_A")] = np.where(
            self.board[self.CELL.index("open_territory_A")] == np.uint8(-1),
            0,
            self.board[self.CELL.index("open_territory_A")],
        )
        self.board[self.CELL.index("open_territory_A"), :, self.width :] = -1
        self.board[self.CELL.index("open_territory_A"), self.height :, :] = -1

        self.board[self.CELL.index("open_territory_B")] = np.where(
            (self.previous_territory_B + self.previous_open_territory_B),
            1,
            0,
        ) - self.compile_layers("rampart_A", "rampart_B", "territory_A", "territory_B")
        self.board[self.CELL.index("open_territory_B")] = np.where(
            self.board[self.CELL.index("open_territory_B")] == np.uint8(-1),
            0,
            self.board[self.CELL.index("open_territory_B")],
        )
        self.board[self.CELL.index("open_territory_B"), :, self.width :] = -1
        self.board[self.CELL.index("open_territory_B"), self.height :, :] = -1

    def calculate_score(self):
        """
        内部関数
        得点を計算
        """
        self.previous_score_A, self.previous_score_B = self.score_A, self.score_B

        self.score_A = (
            np.sum(
                self.board[self.CELL.index("castle"), : self.height, : self.width]
                * self.compile_layers("territory_A", "open_territory_A")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_A += (
            np.sum(
                (1 - self.board[self.CELL.index("castle"), : self.height, : self.width])
                * self.compile_layers("territory_A", "open_territory_A")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["territory"]
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
                * self.compile_layers("territory_B", "open_territory_B")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_B += (
            np.sum(
                (1 - self.board[self.CELL.index("castle"), : self.height, : self.width])
                * self.compile_layers("territory_B", "open_territory_B")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["territory"]
        )
        self.score_B += (
            np.sum(
                self.board[self.CELL.index("rampart_B"), : self.height, : self.width]
            )
            * self.SCORE_MULTIPLIER["rampart"]
        )

        # print(f"score_A:{self.score_A}, score_B:{self.score_B}")

    def get_reward(self):
        if self.current_team == "A":
            return self.get_reward_A()
        elif self.current_team == "B":
            return self.get_reward_B()
        else:
            raise RuntimeError

    def get_reward_A(self):
        """Aチーム報酬更新処理実装予定"""
        reward = self.score_A - self.score_B
        reward += np.abs(reward) * (
            (self.score_A - self.previous_score_A)
            - (self.score_B - self.previous_score_B)
        )
        if self.score_A == self.previous_score_A:
            reward *= 0.75 if reward > 0 else 1.25
        if self.current_player == self.TEAM.index("A") and not all(self.successful):
            reward -= 10000 * sum(self.successful)
        return float(reward)

    def get_reward_B(self):
        """Bチーム報酬更新処理実装予定"""
        reward = self.score_B - self.score_A
        reward += np.abs(reward) * (
            (self.score_B - self.previous_score_B)
            - (self.score_A - self.previous_score_A)
        )
        if self.score_B == self.previous_score_B:
            reward *= 0.75 if reward > 0 else 1.25
        if self.current_player == self.TEAM.index("B") and not all(self.successful):
            reward -= 10000 * sum(self.successful)
        return float(reward)

    def is_done(self):
        """ゲーム終了判定実装予定"""
        # terminated : エピソード終了フラグ
        # truncated  : ステップ数上限での終了フラグ
        terminated, truncated = False, False
        if self.turn >= self.max_episode_steps:
            truncated = True
        # self.terminated = True
        return terminated, truncated

    def step(self, actions: Union[list[int], tuple[int]]):
        """
        gymの必須関数
        1ターン進める処理を実行
        """
        actions = actions[: self.worker_count]
        [worker.turn_init() for worker in self.workers[self.current_team]]
        sorted_workers = [
            (worker, action)
            for worker, action in zip(self.workers[self.current_team], actions)
            if "break" in self.ACTIONS[action]
        ]
        sorted_workers += [
            (worker, action)
            for worker, action in zip(self.workers[self.current_team], actions)
            if "break" not in self.ACTIONS[action]
        ]

        # if self.ACTIONS.index("stay") in actions:
        #     print("stay in actions", actions)
        self.successful = []
        self.stayed_workers = []
        self.action_workers(sorted_workers)
        self.update_territory()
        self.update_open_territory()
        self.update_blank()
        self.calculate_score()
        reward = self.get_reward()
        terminated, truncated = self.is_done()
        self.change_player()
        info = {
            "turn": self.turn,
            "current_team": self.current_team,
            "actions": actions,
            "stayed_workers": self.stayed_workers,
            "score_A": self.score_A,
            "score_B": self.score_B,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        self.turn += 1
        return self.get_observation(), reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        """
        gymの必須関数
        描画を行う
        """
        if self.render_mode == "human":
            return self.render_rgb_array(*args, **kwargs)
        elif self.render_mode == "ansi":
            return self.render_terminal(*args, **kwargs)

    def render_terminal(self):
        view = ""
        icon_base = len(list(self.ICONS.values())[0])
        item_num = int(
            np.max(np.sum(self.board[:, : self.height, : self.width], axis=0))
        )
        cell_num = icon_base * item_num + (item_num - 1)
        for y in range(self.height):
            line = []
            for x in range(self.width):
                # cell = []
                # for i, item in enumerate(self.board[:, y, x]):
                #     if item:
                #         for key, value in self.ICONS.items():
                #             if key in self.CELL[i]:
                #                 cell.append(value)
                # line.append(f"{','.join(cell):^{cell_num}}")
                line.append(
                    f"{','.join([value for i, item in enumerate(self.board[:,y,x]) if item for key,value in self.ICONS.items() if key in self.CELL[i]]):^{cell_num}}"
                )
            # print("|".join(line))
            view += "|".join(line) + "\n"
            if y < self.height - 1:
                # print("-" * (self.width * cell_num + self.width - 1))
                view += "-" * (self.width * cell_num + self.width - 1) + "\n"
        print(view, end="")
        return view

    def render_rgb_array(self):
        """
        描画を行う
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
                                drawAll()
                            showTerritory = not showTerritory

                    if showTerritory:
                        territoryALayer = self.compile_layers(
                            "territory_A", one_hot=True
                        )
                        territoryBLayer = self.compile_layers(
                            "territory_B", one_hot=True
                        )
                        openTerritoryALayer = self.compile_layers(
                            "open_territory_A", one_hot=True
                        )
                        openTerritoryBLayer = self.compile_layers(
                            "open_territory_B", one_hot=True
                        )
                        for i in range(self.height):
                            for j in range(self.width):
                                if territoryALayer[i][j] == 1:
                                    color = self.RED
                                elif territoryBLayer[i][j] == 1:
                                    color = self.BLUE
                                elif openTerritoryALayer[i][j] == 1:
                                    color = self.PINK
                                elif openTerritoryBLayer[i][j] == 1:
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

    def get_actions_from_cli(self):
        [print(f"{i:2}: {action}") for i, action in enumerate(self.ACTIONS)]
        print(
            f"input team {self.current_team} actions (need {self.worker_count} input) : "
        )
        actions = [int(input()) for _ in range(self.worker_count)]
        return actions

    def get_actions(self):
        if self.controller == "pygame":
            actions = self.actions
        elif self.controller == "cli":
            actions = self.get_actions_from_cli()
        else:
            actions = self.random_act()
        while self.WORKER_MAX > len(actions):
            actions.append(0)
        return actions

    def close(self):
        pygame.quit()

    def random_act(self):
        """
        行動可能な範囲でランダムな行動を返す
        """
        [worker.turn_init() for worker in self.workers[self.current_team]]
        self.worker_positions = [
            worker.get_coordinate() for worker in self.workers[self.current_team]
        ]
        act = []
        for worker in self.workers[self.current_team]:
            act_able = []
            pos = worker.get_coordinate()

            for w, action in enumerate(self.ACTIONS):
                direction = self.get_direction(w)
                act_pos = (np.array(pos) + np.array(direction)).astype(int)
                if (
                    ("break" in action and self.is_breakable(worker, *act_pos))
                    or ("move" in action and self.is_movable(worker, *act_pos))
                    or ("build" in action and self.is_buildable(worker, *act_pos))
                    # or w == 0
                ):
                    act_able.append(w)

            act.append(random.choice(act_able))
        while self.WORKER_MAX > len(act):
            act.append(0)
        return act


if __name__ == "__main__":
    fields = glob.glob(os.path.normpath("./field_data/*.csv"))

    env = Game(csv_path=random.choice(fields), render_mode="human", controller="random")

    observation = env.reset()
    terminated, truncated = [False] * 2
    print(f"width:{env.width}, height:{env.height}, workers:{env.worker_count}")

    while not terminated and not truncated:
        print(
            f"input team {env.current_team} actions (need {env.worker_count} input) : "
        )
        env.render()
        observation, reward, terminated, truncated, _ = env.step(env.random_act())
        input()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
