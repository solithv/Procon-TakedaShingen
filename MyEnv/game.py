import copy
import csv
import os
import random
import re
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import pygame
from pygame.locals import *

from .worker import Worker


class Game:
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
    POST_DIRS = {
        "NW": 1,
        "N": 2,
        "NE": 3,
        "E": 4,
        "SE": 5,
        "S": 6,
        "SW": 7,
        "W": 8,
    }
    ICONS = {
        "blank": " ",
        "castle": "C",
        "pond": "P",
        "territory_A": "Ta",
        "territory_B": "Tb",
        "open_territory_A": "ta",
        "open_territory_B": "tb",
        "rampart_A": "Ra",
        "rampart_B": "Rb",
        "worker_A": "Wa",
        "worker_B": "Wb",
        "outside": "X",
    }
    turns = {11: 30, 13: 54, 15: 80, 17: 100, 21: 150, 25: 200}

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    SKY = (127, 176, 255)
    PINK = (255, 127, 127)
    PURPLE = (128, 0, 128)

    def __init__(
        self,
        csv_path: Union[str, list[str]] = None,
        render_mode="ansi",
        max_steps: int = None,
        first_player: Optional[int] = None,
        use_pyautogui: bool = False,
    ):
        """init

        Args:
            csv_path (Union[str, list[str]]): フィールドデータのパス
            render_mode (str, optional): 描画方法. Defaults to "ansi".
            max_steps (int, optional): 最大ステップ数 未指定でマップサイズに応じて変動. Defaults to None.
            first_player (Optional[int], optional): 先行プレイヤーの番号. Defaults to None.
            use_pyautogui (bool): PyAutoGUIを使って描画windowサイズを指定するか. Defaults to False.
        """
        self.csv_path = csv_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.first_player = first_player
        self.board = np.zeros(
            (len(self.CELL), self.FIELD_MAX, self.FIELD_MAX), dtype=np.int8
        )

        self.window_surface = None
        self.clock = None
        self.cwd = os.getcwd()
        if use_pyautogui:
            import pyautogui

            self.display_size_x, self.display_size_y = pyautogui.size()
        else:
            self.display_size_x, self.display_size_y = 960, 960

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

    def update_blank(self, board: np.ndarray):
        """
        内部関数
        blank層を更新
        """
        assert self.CELL.index("blank") != len(self.CELL) - 1
        board[self.CELL.index("blank")] = np.where(
            board[self.CELL.index("blank") + 1 :].any(axis=0), 0, 1
        )
        board = np.where(np.any(board < 0, axis=0), -1, board)
        return board

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
        self.board = np.zeros((len(self.CELL), size, size), dtype=np.int8)
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
        self.board = self.update_blank(self.board)

        return name

    def reset(self, seed=None):
        """
        gymの必須関数
        環境の初期化
        """
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
        if self.csv_path is not None:
            name = self.load_from_csv(self.csv_path)
            info = {"mode": "csv", "csv_name": name}
        else:
            self.board = np.zeros(
                (len(self.CELL), self.FIELD_MAX, self.FIELD_MAX), dtype=np.int8
            )
            self.width, self.height = [self.FIELD_MAX] * 2
            info = {"mode": "api"}
        self.max_turn = (
            self.max_steps if self.max_steps is not None else self.turns[self.width]
        )

        self.cell_size = min(
            self.display_size_x * 0.9 // self.width,
            self.display_size_y * 0.8 // self.height,
        )
        self.window_size = max(self.width, self.height) * self.cell_size
        self.window_size_x = self.width * self.cell_size
        self.window_size_y = self.height * self.cell_size
        if self.render_mode == "human":
            self.reset_render()

        self.board = self.update_blank(self.board)
        return self.get_observation(), info

    def compile_layers(
        self, board: np.ndarray, *layers: tuple[str], one_hot: bool = True
    ):
        """入力された層を合成した2次元配列を返す

        Args:
            board (np.ndarray): 盤面
            layers (tuple[str]): 結合したい層の名称
            one_hot (bool, optional): 返り値の各要素を1,0のみにする. Defaults to True.
        """
        compiled = np.sum(
            [board[self.CELL.index(layer)] for layer in layers],
            axis=0,
            dtype=np.int8,
        )
        if one_hot:
            compiled = np.where(compiled > 0, 1, compiled)
            compiled = np.where(compiled < 0, -1, compiled)
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

    def is_movable(self, worker: Worker, y: int, x: int, smart: bool = False):
        """
        内部関数
        行動可能判定
        """
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and not self.compile_layers(
                self.board,
                f"rampart_{worker.opponent_team}",
                "pond",
                *[f"worker_{worker.opponent_team}{i}" for i in range(self.WORKER_MAX)],
            )[y, x]
            and (y, x) not in self.get_team_worker_coordinate(worker.team)
            and (y, x) not in self.worker_positions
        ):
            if smart:
                if (
                    worker.action_log
                    and worker.action_log[-1][0] == "move"
                    and worker.action_log[-1][1] == (y, x)
                ):
                    return False
                field = self.get_around(self.board, y, x, side_length=3)
                compiled: np.ndarray = np.sum(
                    [
                        field[self.CELL.index(layer)]
                        for layer in (
                            f"rampart_{worker.team}",
                            f"territory_{worker.team}",
                        )
                    ],
                    axis=0,
                )
                return not compiled.all()
            return True
        else:
            return False

    def is_buildable(self, worker: Worker, y: int, x: int, smart: bool = False):
        """
        内部関数
        建築可能判定
        """
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and not self.compile_layers(
                self.board,
                "rampart_A",
                "rampart_B",
                "castle",
                *[f"worker_{worker.opponent_team}{i}" for i in range(self.WORKER_MAX)],
            )[y, x]
        ):
            if smart:
                territory = copy.deepcopy(
                    self.board[self.CELL.index(f"rampart_{worker.team}")]
                )
                territory[y, x] = 1
                res = (
                    self.fill_area(territory).sum()
                    >= self.fill_area(
                        self.board[self.CELL.index(f"rampart_{worker.team}")]
                    ).sum()
                )
                return res
            else:
                return True
        else:
            return False

    def is_breakable(self, worker: Worker, y: int, x: int, smart: bool = False):
        """
        内部関数
        破壊可能判定
        """
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and self.compile_layers(self.board, "rampart_A", "rampart_B")[y, x]
        ):
            if (
                smart
                and self.compile_layers(self.board, f"rampart_{worker.team}")[y, x]
            ):
                territory = copy.deepcopy(
                    self.board[self.CELL.index(f"rampart_{worker.team}")]
                )
                territory[y, x] = 0
                res = (
                    self.fill_area(territory).sum()
                    > self.fill_area(
                        self.board[self.CELL.index(f"rampart_{worker.team}")]
                    ).sum()
                )
                return res
            else:
                return True
        else:
            return False

    def is_actionable(
        self,
        worker: Worker,
        action: str,
        stay=False,
        smart: bool = False,
        smart_move: bool = False,
    ):
        """行動が有効か判定

        Args:
            worker (Worker): 職人
            action (str): 行動名
            stay (bool, optional): 待機を許容するか. Defaults to False.
            smart (bool, optional): 建築、破壊が点数向上するか判定する. Defaults to False.
            move_smart (bool, optional): 移動先が城壁と陣地で埋まっているか判定する. Defaults to False.
        """
        position = worker.get_coordinate()
        direction = self.get_direction(self.ACTIONS.index(action))
        act_pos = (np.array(position) + np.array(direction)).astype(int)
        if "break" in action and self.is_breakable(worker, *act_pos, smart):
            return True
        elif "build" in action and self.is_buildable(worker, *act_pos, smart):
            return True
        elif "move" in action and self.is_movable(worker, *act_pos, smart_move):
            return True
        elif stay and action == "stay":
            return True
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
        ) - self.compile_layers(
            self.board, "rampart_A", "rampart_B", "territory_A", "territory_B"
        )
        self.board[self.CELL.index("open_territory_A")] = np.where(
            self.board[self.CELL.index("open_territory_A")] == np.int8(-1),
            0,
            self.board[self.CELL.index("open_territory_A")],
        )
        self.board[self.CELL.index("open_territory_A"), :, self.width :] = -1
        self.board[self.CELL.index("open_territory_A"), self.height :, :] = -1

        self.board[self.CELL.index("open_territory_B")] = np.where(
            (self.previous_territory_B + self.previous_open_territory_B),
            1,
            0,
        ) - self.compile_layers(
            self.board, "rampart_A", "rampart_B", "territory_A", "territory_B"
        )
        self.board[self.CELL.index("open_territory_B")] = np.where(
            self.board[self.CELL.index("open_territory_B")] == np.int8(-1),
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
                * self.compile_layers(self.board, "territory_A", "open_territory_A")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_A += (
            np.sum(
                (1 - self.board[self.CELL.index("castle"), : self.height, : self.width])
                * self.compile_layers(self.board, "territory_A", "open_territory_A")[
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
                * self.compile_layers(self.board, "territory_B", "open_territory_B")[
                    : self.height, : self.width
                ]
            )
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_B += (
            np.sum(
                (1 - self.board[self.CELL.index("castle"), : self.height, : self.width])
                * self.compile_layers(self.board, "territory_B", "open_territory_B")[
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
        if self.turn >= self.max_turn:
            truncated = True
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

        self.successful = []
        self.stayed_workers = []
        self.action_workers(sorted_workers)
        self.update_territory()
        self.update_open_territory()
        self.board = self.update_blank(self.board)
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

    def dummy_step(self):
        [worker.turn_init() for worker in self.workers[self.current_team]]
        workers = [
            (worker, self.ACTIONS.index("stay"))
            for worker in self.workers[self.current_team]
        ]
        self.successful = []
        self.stayed_workers = []
        self.action_workers(workers)
        self.update_territory()
        self.update_open_territory()
        self.board = self.update_blank(self.board)
        self.calculate_score()
        reward = self.get_reward()
        terminated, truncated = self.is_done()
        self.change_player()
        info = {
            "turn": self.turn,
            "current_team": self.current_team,
            "actions": "dummy",
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
        """コンソールに描画"""
        view = ""
        icon_base = max(len(value) for value in self.ICONS.values())
        item_num = int(
            np.max(np.sum(self.board[:, : self.height, : self.width], axis=0))
        )
        cell_num = icon_base * item_num + (item_num - 1)
        for y in range(self.height):
            view += (
                "|".join(
                    [
                        f"{','.join([value for i, item in enumerate(self.board[:,y,x]) if item for key,value in self.ICONS.items() if key in self.CELL[i]]):^{cell_num}}"
                        for x in range(self.width)
                    ]
                )
                + "\n"
            )
            if y < self.height - 1:
                view += "-" * (self.width * cell_num + self.width - 1) + "\n"
        print(view, end="")
        return view

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
        for i in range(1, self.width):
            pygame.draw.line(
                self.window_surface,
                self.BLACK,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size_y),
                1,
            )
        # 横線描画
        for i in range(1, self.height):
            pygame.draw.line(
                self.window_surface,
                self.BLACK,
                (0, i * self.cell_size),
                (self.window_size_x, i * self.cell_size),
                1,
            )

    def placeImage(self, img, i, j, workerNumber=None, scale=1.0):
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

        if workerNumber:
            font = pygame.font.SysFont(None, 30)
            text = font.render(workerNumber, False, self.BLACK)
            text_rect = text.get_rect(
                center=((j + 0.5) * self.cell_size, (i + 0.2) * self.cell_size)
            )
            self.window_surface.blit(text, text_rect)

    def drawAll(self, view):
        for i in range(self.height):
            for j in range(self.width):
                self.placeImage(self.BLANK_IMG, i, j)
                cellInfo = view[i][j]
                worker_A_exist = any(
                    f"worker_A{k}" in cellInfo for k in range(self.WORKER_MAX)
                )
                worker_B_exist = any(
                    f"worker_B{k}" in cellInfo for k in range(self.WORKER_MAX)
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
            self.BLACK,
            (
                0,
                self.cell_size * self.height,
                self.cell_size * self.width,
                self.cell_size * 2,
            ),
        )
        font = pygame.font.SysFont(None, 60)
        text = font.render(
            f"{self.turn}/{self.max_turn} turn : {self.current_team}'s turn",
            False,
            self.WHITE,
        )

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

        [worker.turn_init() for worker in self.workers[self.current_team]]
        self.worker_positions = [
            worker.get_coordinate() for worker in self.workers[self.current_team]
        ]
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

                if showTerritory:
                    territoryALayer = self.compile_layers(
                        self.board, "territory_A", one_hot=True
                    )
                    territoryBLayer = self.compile_layers(
                        self.board, "territory_B", one_hot=True
                    )
                    openTerritoryALayer = self.compile_layers(
                        self.board, "open_territory_A", one_hot=True
                    )
                    openTerritoryBLayer = self.compile_layers(
                        self.board, "open_territory_B", one_hot=True
                    )
                    for i in range(self.height):
                        for j in range(self.width):
                            if territoryALayer[i][j] == territoryBLayer[i][j] == 1:
                                color = self.PURPLE
                            elif territoryALayer[i][j] == 1:
                                color = self.RED
                            elif territoryBLayer[i][j] == 1:
                                color = self.BLUE
                            elif openTerritoryALayer[i][j] == 1:
                                color = self.PINK
                            elif openTerritoryBLayer[i][j] == 1:
                                color = self.SKY
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
                            if not self.is_actionable(
                                self.workers[self.current_team][actingWorker],
                                self.ACTIONS[action],
                                stay=True,
                            ):
                                continue
                            actions.append(action)
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
                        if not self.is_actionable(
                            self.workers[self.current_team][actingWorker],
                            self.ACTIONS[action],
                            stay=True,
                        ):
                            continue
                        actions.append(action)
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
                        if not self.is_actionable(
                            self.workers[self.current_team][actingWorker],
                            self.ACTIONS[action],
                            stay=True,
                        ):
                            continue
                        actions.append(action)
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

    def end_game_render(self):
        """終了状態描画"""
        if self.window_surface is None:
            self.reset_render()
            pygame.init()
            pygame.display.init()
            self.window_surface = pygame.display.set_mode(
                (self.window_size_x, self.window_size_y + self.cell_size * 2)
            )
            pygame.display.set_caption("game")

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
        self.drawAll(view)
        pygame.draw.rect(
            self.window_surface,
            self.BLACK,
            (
                0,
                self.cell_size * self.height,
                self.cell_size * self.width,
                self.cell_size * 2,
            ),
        )
        font = pygame.font.SysFont(None, 60)
        self.update_territory()
        self.update_open_territory()
        self.board = self.update_blank(self.board)
        self.calculate_score()
        text = font.render(
            f"A team:{self.score_A}, B team:{self.score_B}",
            False,
            self.WHITE,
        )
        text_rect = text.get_rect(
            center=(
                self.cell_size * self.width / 2,
                self.cell_size * (self.height + 1),
            )
        )
        self.window_surface.blit(text, text_rect)
        pygame.display.update()
        while True:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        showTerritory = not showTerritory
                        self.drawAll(view)
                    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return

                    if showTerritory:
                        territoryALayer = self.compile_layers(
                            self.board, "territory_A", one_hot=True
                        )
                        territoryBLayer = self.compile_layers(
                            self.board, "territory_B", one_hot=True
                        )
                        openTerritoryALayer = self.compile_layers(
                            self.board, "open_territory_A", one_hot=True
                        )
                        openTerritoryBLayer = self.compile_layers(
                            self.board, "open_territory_B", one_hot=True
                        )
                        for i in range(self.height):
                            for j in range(self.width):
                                if territoryALayer[i][j] == territoryBLayer[i][j] == 1:
                                    color = self.PURPLE
                                elif territoryALayer[i][j] == 1:
                                    color = self.RED
                                elif territoryBLayer[i][j] == 1:
                                    color = self.BLUE
                                elif openTerritoryALayer[i][j] == 1:
                                    color = self.PINK
                                elif openTerritoryBLayer[i][j] == 1:
                                    color = self.SKY
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
                pygame.display.update()

    def close(self):
        """描画終了処理"""
        self.window_surface = None
        self.clock = None
        pygame.display.quit()
        pygame.quit()

    def get_actions_from_cli(self):
        """コンソールから行動を入力"""
        [print(f"{i:2}: {action}") for i, action in enumerate(self.ACTIONS)]
        print(
            f"input team {self.current_team} actions (need {self.worker_count} input) : "
        )
        actions = [int(input()) for _ in range(self.worker_count)]
        return actions

    def get_actions(self, controller: Optional[str] = None):
        """操作入力 入力方法が指定されていない場合ランダム行動

        Args:
            controller (str, optional): 操作入力方法("cli", "pygame"). Defaults to None.
        """
        if controller == "pygame":
            while True:
                actions = self.get_actions_from_pygame()
                [worker.turn_init() for worker in self.workers[self.current_team]]
                self.worker_positions = [
                    worker.get_coordinate()
                    for worker in self.workers[self.current_team]
                ]
                if all(
                    self.is_actionable(worker, self.ACTIONS[action], stay=True)
                    for worker, action in zip(self.workers[self.current_team], actions)
                ):
                    break
                self.render_rgb_array()

        elif controller == "cli":
            actions = self.get_actions_from_cli()
        else:
            actions = self.get_random_actions()
        while self.WORKER_MAX > len(actions):
            actions.append(self.ACTIONS.index("stay"))
        return actions

    def random_act(self):
        """行動可能な範囲でランダムな行動を返す"""
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
                    or action == "stay"
                ):
                    act_able.append(w)

            if len(act_able) > 0:
                act.append(random.choice(act_able))
            else:
                act.append(self.ACTIONS.index("stay"))

        while self.WORKER_MAX > len(act):
            act.append(self.ACTIONS.index("stay"))
        return act

    def get_random_action(self, worker: Worker):
        """職人ごとに有効な行動をランダムに返す

        Args:
            worker (Worker): 職人
        """
        worker.turn_init()
        actionable = []
        group = "break"
        for action in reversed(self.ACTIONS):
            if group != action.split("_")[0] and len(actionable):
                break
            group = action.split("_")[0]
            if self.is_actionable(worker, action, smart=True, smart_move=True):
                actionable.append(self.ACTIONS.index(action))
        if len(actionable) == 0:
            for action in self.ACTIONS:
                if self.is_actionable(worker, action, smart=True, smart_move=False):
                    actionable.append(self.ACTIONS.index(action))

        if len(actionable) > 0:
            return random.choice(actionable)
        else:
            return self.ACTIONS.index("stay")

    def get_random_actions(self, team: str = None):
        """有効な行動をランダムで返す

        Args:
            team (str, optional): チームを指定. Defaults to None.
        """
        team = team if team is not None else self.current_team
        [worker.turn_init() for worker in self.workers[team]]
        self.worker_positions = [
            worker.get_coordinate() for worker in self.workers[team]
        ]
        act = []
        for worker in self.workers[team]:
            act.append(self.get_random_action(worker))

        while self.WORKER_MAX > len(act):
            act.append(self.ACTIONS.index("stay"))
        return act

    def check_actions(self, actions: list[int], team: str = None, stay=False):
        """AIの行動を確認して無効なら上書き

        Args:
            actions (list[int]): 行動のリスト
            team (str, optional): チームを指定. Defaults to None.
            stay (bool, optional): 待機を許容するか. Defaults to False.
        """
        team = team if team is not None else self.current_team
        [worker.turn_init() for worker in self.workers[team]]
        self.worker_positions = [
            worker.get_coordinate() for worker in self.workers[team]
        ]
        for i, (worker, action) in enumerate(zip(self.workers[team], actions)):
            if not self.is_actionable(
                worker, self.ACTIONS[action], stay=stay, smart=True
            ):
                actions[i] = self.get_random_action(worker)
        return actions

    def get_around(
        self, board: np.ndarray, y: int, x: int, side_length: int = 5, raw=False
    ):
        """指定マスの周囲を取得

        Args:
            board (ndarray): 盤面
            y (int): y座標
            x (int): x座標
            side_length (int, optional): 取得領域. Defaults to 5.
            raw (bool, optional): Falseなら職人の層を結合. Defaults to False.
        """
        if side_length % 2 == 0:
            raise ValueError("need to input an odd number")
        length_ = side_length // 2
        field = np.pad(
            board,
            [(0, 0), (length_, length_), (length_, length_)],
            "constant",
            constant_values=-1,
        )
        front = length_ * 2 + 1
        field = field[:, y : y + front, x : x + front]

        return field if raw else self.compile_worker_layers(field)

    def compile_worker_layers(self, board: np.ndarray):
        """職人の層を結合

        Args:
            board (ndarray): 盤面
        """
        a = np.sum(
            [
                board[self.CELL.index(layer)]
                for layer in self.CELL
                if "worker_A" in layer
            ],
            axis=0,
        )[np.newaxis, :, :]
        a = np.where(a < 0, -1, a)
        b = np.sum(
            [
                board[self.CELL.index(layer)]
                for layer in self.CELL
                if "worker_B" in layer
            ],
            axis=0,
        )[np.newaxis, :, :]
        b = np.where(b < 0, -1, b)
        board = np.concatenate([board[: self.CELL.index("worker_A0")], a, b], axis=0)
        return board

    def get_around_workers(
        self, side_length: int = 5, team: str = None
    ) -> list[np.ndarray]:
        """職人の周囲を取得する

        Args:
            side_length (int, optional): 1辺の長さ(奇数で指定). Defaults to 3.
            team (str, optional): チーム名("A" or "B") 未指定で現在のチーム. Defaults to None.

        Returns:
            list[np.ndarray]: 職人の周囲
        """
        if team is None:
            team = self.current_team
        elif team == "opponent":
            team = self.opponent_team
        around_workers = [
            self.get_around(self.board, *worker.get_coordinate(), side_length)
            for worker in self.workers[team]
        ]
        return around_workers

    def print_around(self, around_workers: list[np.ndarray]):
        """職人の周囲を表示する(debug用)

        Args:
            around_workers (list[np.ndarray]): 職人の周囲
        """
        layers = self.CELL[: self.CELL.index("worker_A0")] + ("worker_A", "worker_B")
        for around in around_workers:
            view = ""
            icon_base = max(len(value) for value in self.ICONS.values())
            item_num = int(np.max(np.sum(around, axis=0)))
            cell_num = icon_base * item_num + item_num - 1
            _, height, width = around.shape
            for y in range(height):
                line = []
                for x in range(width):
                    cell = []
                    for i, item in enumerate(around[:, y, x]):
                        if item < 0:
                            cell.append(self.ICONS["outside"])
                            break
                        elif item:
                            cell.append(self.ICONS[layers[i]])
                    line.append(f"{','.join(cell):^{cell_num}}")

                view += "|".join(line) + "\n"
                if y < height - 1:
                    view += "-" * (width * cell_num + width - 1) + "\n"
            print(view)

    def reset_from_api(self, data: dict[str, Any]):
        """APIから環境を初期化

        Args:
            data (dict[str, Any]): 試合一覧取得APIから受け取った1試合分のデータ
        """
        self.id = data["id"]
        self.current_player = 0 if data["first"] else 1
        self.change_player(no_change=True)
        self.max_steps = data["turns"]
        self.SCORE_MULTIPLIER["castle"] = data["bonus"]["castle"]
        self.SCORE_MULTIPLIER["territory"] = data["bonus"]["territory"]
        self.SCORE_MULTIPLIER["rampart"] = data["bonus"]["wall"]
        self.turn = 1
        # self.turn_seconds=data["turnSeconds"]
        self.height, self.width = data["board"]["height"], data["board"]["width"]
        self.worker_count = data["board"]["mason"]
        self.workers: defaultdict[str, list[Worker]] = defaultdict(list)
        structures = np.pad(
            np.array(data["board"]["structures"]),
            [
                (0, self.FIELD_MAX - self.height),
                (0, self.FIELD_MAX - self.width),
            ],
        )
        masons = np.pad(
            np.array(data["board"]["masons"]),
            [
                (0, self.FIELD_MAX - self.height),
                (0, self.FIELD_MAX - self.width),
            ],
        )
        self.board = np.zeros(
            (len(self.CELL), self.FIELD_MAX, self.FIELD_MAX), dtype=np.int8
        )
        self.board[self.CELL.index("castle")] = np.where(structures == 2, 1, 0)
        self.board[self.CELL.index("pond")] = np.where(structures == 1, 1, 0)
        for i in range(self.worker_count):
            self.board[self.CELL.index(f"worker_A{i}")] = np.where(
                masons == i + 1, 1, 0
            )
            self.board[self.CELL.index(f"worker_B{i}")] = np.where(
                masons == -(i + 1), 1, 0
            )
            self.workers["A"].append(
                Worker(f"worker_A{i}", *np.argwhere(masons == i + 1)[0])
            )
            self.workers["B"].append(
                Worker(f"worker_B{i}", *np.argwhere(masons == -(i + 1))[0])
            )
        self.board[:, self.height :, :] = -1
        self.board[:, :, self.width :] = -1
        self.board = self.update_blank(self.board)
        self.update_territory()

        self.max_turn = (
            self.max_steps if self.max_steps is not None else self.turns[self.width]
        )
        self.cell_size = min(
            self.display_size_x * 0.9 // self.width,
            self.display_size_y * 0.8 // self.height,
        )
        self.window_size = max(self.width, self.height) * self.cell_size
        self.window_size_x = self.width * self.cell_size
        self.window_size_y = self.height * self.cell_size
        if self.render_mode == "human":
            self.reset_render()

    def get_stat_from_api(self, data: dict[str, Any]):
        """APIから環境状態を更新

        Args:
            data (dict[str, Any]): 試合状態取得APIから受け取ったデータ
        """
        assert self.id == data["id"], f"self.id:{self.id}, data['id']:{data['id']}"
        assert (
            self.worker_count == data["board"]["mason"]
        ), f"self.worker_count:{self.worker_count}, data['board']['mason']:{data['board']['mason']}"
        assert (
            self.turn - 1 == data["turn"]
        ), f"self.turn:{self.turn}, data['turn']:{data['turn']}"
        # self.turn = data["turn"] + 1
        structures = np.pad(
            np.array(data["board"]["structures"]),
            [
                (0, self.FIELD_MAX - self.height),
                (0, self.FIELD_MAX - self.width),
            ],
        )
        masons = np.pad(
            np.array(data["board"]["masons"]),
            [
                (0, self.FIELD_MAX - self.height),
                (0, self.FIELD_MAX - self.width),
            ],
        )
        walls = np.pad(
            np.array(data["board"]["walls"]),
            [
                (0, self.FIELD_MAX - self.height),
                (0, self.FIELD_MAX - self.width),
            ],
        )
        territories = np.pad(
            np.array(data["board"]["territories"]),
            [
                (0, self.FIELD_MAX - self.height),
                (0, self.FIELD_MAX - self.width),
            ],
        )
        self.board[self.CELL.index("castle")] = np.where(structures == 2, 1, 0)
        self.board[self.CELL.index("pond")] = np.where(structures == 1, 1, 0)
        self.board[self.CELL.index("rampart_A")] = np.where(walls == 1, 1, 0)
        self.board[self.CELL.index("rampart_B")] = np.where(walls == 2, 1, 0)
        self.board[self.CELL.index("territory_A")] = np.where(
            (territories == 1) | (territories == 3), 1, 0
        )
        self.board[self.CELL.index("territory_B")] = np.where(
            (territories == 2) | (territories == 3), 1, 0
        )
        for i in range(self.worker_count):
            self.board[self.CELL.index(f"worker_A{i}")] = np.where(
                masons == i + 1, 1, 0
            )
            self.board[self.CELL.index(f"worker_B{i}")] = np.where(
                masons == -(i + 1), 1, 0
            )
            self.workers["A"][i].update_coordinate(*np.argwhere(masons == i + 1)[0])
            self.workers["B"][i].update_coordinate(*np.argwhere(masons == -(i + 1))[0])
        self.board[:, self.height :, :] = -1
        self.board[:, :, self.width :] = -1
        self.update_open_territory()
        self.board = self.update_blank(self.board)
        self.calculate_score()

    def make_post_data(self, actions: list[int]):
        """行動計画APIで送るデータを作成

        Args:
            actions (list[int]): 行動のリスト
        """

        def get_type(action: int):
            if "stay" in self.ACTIONS[action]:
                return 0
            elif "move" in self.ACTIONS[action]:
                return 1
            elif "build" in self.ACTIONS[action]:
                return 2
            elif "break" in self.ACTIONS[action]:
                return 3

        def get_dir(action: int):
            return self.POST_DIRS.get(self.ACTIONS[action].split("_")[-1], 0)

        data = {"turn": self.turn}
        data["actions"] = [
            {"type": get_type(action), "dir": get_dir(action)}
            for action in actions[: self.worker_count]
        ]
        return data
