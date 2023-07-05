import gymnasium as gym
import numpy as np
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


class Worker:
    TEAMS = ("A", "B")

    def __init__(self, name, y, x):
        self.name = name
        self.team = name[-2]
        self.num = name[-1]
        self.another_team = self.TEAMS[1 - self.TEAMS.index(self.team)]
        self.y = y
        self.x = x
        self.is_action = False
        self.action_log = []

    def move(self, y, x):
        if self.x - 1 <= x <= self.x + 1 and self.y - 1 <= y <= self.y + 1:
            self.x = x
            self.y = y
            self.action_log.append(("move", (y, x)))
            self.is_action = True
            return True
        else:
            return False

    def build(self, y, x):
        self.action_log.append(("build", (y, x)))
        self.is_action = True

    def break_(self, y, x):
        self.action_log.append(("break", (y, x)))
        self.is_action = True

    def get_coordinate(self):
        return self.y, self.x

    def turn_init(self):
        self.is_action = False


class Game(gym.Env):
    metadata = {"render.modes": ["human", "console"]}
    CELL_SIZE = 32
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
        "worker_A0",
        "worker_A1",
        "worker_A2",
        "worker_A3",
        "worker_A4",
        "worker_A5",
        "worker_B0",
        "worker_B1",
        "worker_B2",
        "worker_B3",
        "worker_B4",
        "worker_B5",
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
        # y, x
        "N": np.array([-1, 0]),
        "E": np.array([0, 1]),
        "S": np.array([1, 0]),
        "W": np.array([0, -1]),
    }
    SCORE_MULTIPLIER = {"castle": 100, "position": 50, "rampart": 10}
    POND_MIN, POND_MAX = 1, 5
    FIELD_MIN, FIELD_MAX = 11, 25
    WORKER_MIN, WORKER_MAX = 2, 6

    def __init__(self, end_turn=10, width=None, height=None, pond=None, worker=None):
        super().__init__()
        self.end_turn = end_turn
        self.width = width or np.random.randint(self.FIELD_MIN, self.FIELD_MAX)
        self.height = height or np.random.randint(self.FIELD_MIN, self.FIELD_MAX)
        self.pond_count = pond or np.random.randint(self.POND_MIN, self.POND_MAX)
        self.worker_count = worker or np.random.randint(
            self.WORKER_MIN, self.WORKER_MAX
        )
        self.current_player = 1
        self.score_A, self.score_B = 0, 0
        self.previous_score_A, self.previous_score_B = 0, 0
        self.turn = 0
        self.done = False
        self.board = np.zeros((len(self.CELL), self.height, self.width))

        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.CELL), self.height, self.width),
            dtype=np.int8,
        )
        self.reward_range = [np.NINF, np.inf]
        self.window_size = max(self.width, self.height) * self.CELL_SIZE
        self.window_size_x = self.width * self.CELL_SIZE
        self.window_size_y = self.height * self.CELL_SIZE

    def set_cell_property(self, target, coordinates=None):
        if not coordinates:
            while True:
                y = np.random.randint(0, self.height - 1)
                x = np.random.randint(0, self.width - 1)
                if (y, x) not in self.used:
                    break
        else:
            y, x = coordinates
        self.board[self.CELL.index(target), y, x] = 1
        self.used.append((y, x))
        return y, x

    def set_worker_position(self, target, coordinates=None):
        y, x = self.set_cell_property(target, coordinates)
        return Worker(target, y, x)

    def update_blank(self):
        self.board[0] = 1 - self.board[1:].any(axis=0)

    def reset(self, castle=None, pond=None, worker_A=None, worker_B=None):
        self.current_player = 1
        self.score_A, self.score_B = 0, 0
        self.previous_score_A, self.previous_score_B = 0, 0
        self.turn = 0
        self.done = False
        self.board = np.zeros((len(self.CELL), self.height, self.width))
        self.used = []
        # self.width = width or np.random.randint(self.FIELD_MIN, self.FIELD_MAX)
        # self.height = height or np.random.randint(self.FIELD_MIN, self.FIELD_MAX)
        # self.pond_count = pond or np.random.randint(self.POND_MIN, self.POND_MAX)
        # self.worker_count = np.random.randint(self.WORKER_MIN, self.WORKER_MAX)

        self.set_cell_property("castle", coordinates=castle)

        if pond:
            assert self.pond_count == len(pond), "pond input error"
            [
                self.set_cell_property("pond", coordinates=coordinate)
                for coordinate in pond
            ]
        else:
            [self.set_cell_property("pond") for _ in range(self.pond_count)]

        if worker_A:
            assert self.worker_count == len(worker_A), "worker_A input error"
            self.workers_A = [
                self.set_worker_position(f"worker_A{i}", coordinate)
                for i, coordinate in enumerate(worker_A)
            ]
        else:
            self.workers_A = [
                self.set_worker_position(f"worker_A{i}")
                for i in range(self.worker_count)
            ]

        if worker_B:
            assert self.worker_count == len(worker_B), "worker_B input error"
            self.workers_B = [
                self.set_worker_position(f"worker_B{i}", coordinate)
                for i, coordinate in enumerate(worker_B)
            ]
        else:
            self.workers_B = [
                self.set_worker_position(f"worker_B{i}")
                for i in range(self.worker_count)
            ]

        self.update_blank()
        return self.board

    def compile_layers(self, *layers, one_hot=False):
        compiled = np.sum(
            [self.board[self.CELL.index(layer)] for layer in layers], axis=0
        )
        if one_hot:
            return np.where(compiled, 1, 0)
        else:
            return compiled

    def get_team_worker_coordinate(self, team):
        return [
            worker.get_coordinate()
            for worker in eval(f"self.workers_{team}")
            if worker.is_action
        ]

    def is_movable(self, worker: Worker, y, x):
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and not self.compile_layers(
                "rampart_A",
                "rampart_B",
                "pond",
                *[f"worker_{worker.another_team}{i}" for i in range(self.WORKER_MAX)],
            )[y, x]
            and (y, x) not in self.get_team_worker_coordinate(worker.team)
        ):
            return True
        else:
            return False

    def is_buildable(self, worker: Worker, y, x):
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and not self.compile_layers(
                "rampart_A",
                "rampart_B",
                "castle",
                *[f"worker_{worker.another_team}{i}" for i in range(self.WORKER_MAX)],
            )[y, x]
            and (y, x) not in self.get_team_worker_coordinate(worker.team)
        ):
            return True
        else:
            return False

    def is_breakable(self, worker: Worker, y, x):
        if (
            not worker.is_action
            and 0 <= y < self.height
            and 0 <= x < self.width
            and self.compile_layers("rampart_A", "rampart_B")[y, x]
        ):
            return True
        else:
            return False

    def get_direction(self, action):
        direction = np.zeros(2)
        if "N" in self.ACTIONS[action]:
            direction += self.DIRECTIONS["N"]
        if "E" in self.ACTIONS[action]:
            direction += self.DIRECTIONS["E"]
        if "S" in self.ACTIONS[action]:
            direction += self.DIRECTIONS["S"]
        if "W" in self.ACTIONS[action]:
            direction += self.DIRECTIONS["W"]
        return direction

    def worker_action(self, worker: Worker, action):
        if "stay" == self.ACTIONS[action]:
            return True

        direction = self.get_direction(action)
        y, x = map(int, np.array(worker.get_coordinate()) + direction)

        if "move" in self.ACTIONS[action] and self.is_movable(worker, y, x):
            self.board[self.CELL.index(worker.name), worker.y, worker.x] = 0
            self.board[self.CELL.index(worker.name), y, x] = 1
            worker.move(y, x)

        elif "build" in self.ACTIONS[action] and self.is_buildable(worker, y, x):
            self.board[self.CELL.index(f"rampart_{worker.team}"), y, x] = 1
            worker.build(y, x)

        elif "break" in self.ACTIONS[action] and self.is_breakable(worker, y, x):
            if self.board[self.CELL.index("rampart_A"), y, x]:
                self.board[self.CELL.index("rampart_A"), y, x] = 0
            else:
                self.board[self.CELL.index("rampart_B"), y, x] = 0
            worker.break_(y, x)

        else:
            print("行動できない入力です")
            return False

        self.update_blank()
        return True

    def fill_area(self, array):
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

    def update_open_position(self):
        self.previous_open_position_A = self.board[self.CELL.index("open_position_A")]
        self.previous_open_position_B = self.board[self.CELL.index("open_position_B")]
        self.board[self.CELL.index("open_position_A")] = np.where(
            self.fill_area(self.board[self.CELL.index("position_A")])
            + self.previous_open_position_A,
            1,
            0,
        ) - self.compile_layers("rampart_B", "position_B")
        self.board[self.CELL.index("open_position_B")] = np.where(
            self.fill_area(self.board[self.CELL.index("position_B")])
            + self.previous_open_position_A,
            1,
            0,
        ) - self.compile_layers("rampart_A", "position_A")
        self.update_blank()

    def calculate_score(self):
        self.previous_score_A, self.previous_score_B = self.score_A, self.score_B

        self.score_A = np.sum(
            self.board[self.CELL.index("castle")]
            * self.compile_layers("position_A", "open_position_A", one_hot=True)
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_A += np.sum(
            (1 - self.board[self.CELL.index("castle")])
            * self.compile_layers("position_A", "open_position_A", one_hot=True)
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_A += np.sum(
            self.board[self.CELL.index("rampart_A")] * self.SCORE_MULTIPLIER["rampart"]
        )

        self.score_B = np.sum(
            self.board[self.CELL.index("castle")]
            * self.compile_layers("position_B", "open_position_B", one_hot=True)
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_B += np.sum(
            (1 - self.board[self.CELL.index("castle")])
            * self.compile_layers("position_B", "open_position_B", one_hot=True)
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_B += np.sum(
            self.board[self.CELL.index("rampart_B")] * self.SCORE_MULTIPLIER["rampart"]
        )

        print(f"score_A:{self.score_A}, score_B:{self.score_B}")

    def get_reward(self, successful):
        """報酬更新処理実装予定"""
        if not successful:
            return np.NINF

    def is_done(self):
        """ゲーム終了判定実装予定"""
        if self.turn >= self.end_turn:
            self.done = True

    def step(self, actions):
        assert self.worker_count == len(actions), "input length error"
        [worker.turn_init() for worker in self.workers_A]
        [worker.turn_init() for worker in self.workers_B]
        successful = all(
            [
                self.worker_action(worker, action)
                for worker, action in zip(
                    self.workers_A if self.current_player > 0 else self.workers_B,
                    actions,
                )
            ]
        )
        self.update_open_position()
        self.current_player = -self.current_player
        self.turn += 1
        self.calculate_score()
        reward = self.get_reward(successful)
        self.is_done()
        return self.board, reward, self.done, {}

    def render(self, mode="human"):
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
                (self.window_size_x, self.window_size_y)
            )
            pygame.display.set_caption("game")

            window_surface.fill(WHITE)

            for i in range(self.height):
                for j in range(self.width):
                    cellPlacement = (
                        j * self.CELL_SIZE,
                        i * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    )
                    cellInfo = view[i][j]

                    if "castle" in cellInfo:
                        color = YELLOW
                    elif "worker_A" in cellInfo:
                        color = RED
                    elif "worker_B" in cellInfo:
                        color = BLUE
                    elif "pond" in cellInfo:
                        color = GREEN
                    else:
                        color = WHITE

                    pygame.draw.rect(window_surface, color, cellPlacement)

            # 縦線描画
            for i in range(1, self.width):
                pygame.draw.line(
                    window_surface,
                    BLACK,
                    (i * self.CELL_SIZE, 0),
                    (i * self.CELL_SIZE, self.window_size_y),
                    1,
                )
            # 横線描画
            for i in range(1, self.height):
                pygame.draw.line(
                    window_surface,
                    BLACK,
                    (0, i * self.CELL_SIZE),
                    (self.window_size_x, i * self.CELL_SIZE),
                    1,
                )

            pygame.display.update()


env = Game()

print(f"width:{env.width}, height:{env.height}, workers:{env.worker_count}")

observation = env.reset()
done = False

while not done:
    env.render()
    print(env.board[env.CELL.index("worker_A0")])

    print(f"input team A actions (need {env.worker_count} input) : ")
    actions = [int(input()) for _ in range(env.worker_count)]
    observation, reward, done, _ = env.step(actions)

    print(env.board[env.CELL.index("worker_A0")])
    env.render()

    print(f"input team B actions (need {env.worker_count} input) : ")
    actions = [int(input()) for _ in range(env.worker_count)]
    observation, reward, done, _ = env.step(actions)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

env.render()
