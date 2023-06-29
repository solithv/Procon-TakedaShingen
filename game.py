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
    def __init__(
        self,
        name,
        x,
        y,
        num,
    ):
        self.name = name
        self.team = name[-1]
        self.x = x
        self.y = y
        self.num = num
        self.is_action = False
        self.action_log = []

    def move(self, y, x):
        if self.x - 1 <= x <= self.x + 1 and self.y - 1 <= y <= self.y + 1:
            self.x = x
            self.y = y
            self.action_log.append("move", (y, x))
            self.is_action = True
            return True
        else:
            return False

    def build(self, y, x):
        self.action_log.append("build", (y, x))
        self.is_action = True

    def break_(self, y, x):
        self.action_log.append("break", (y, x))
        self.is_action = True

    def turn_init(self):
        self.is_action = False


class Game(gym.Env):
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
        "worker_A",
        "worker_B",
    )
    ACTIONS = (
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

    def __init__(self, end_turn=10, width=None, height=None, pond=None, worker=None):
        super().__init__()
        self.end_turn = end_turn
        self.width = width or np.random.randint(11, 25)
        self.height = height or np.random.randint(11, 25)
        self.pond_count = pond or np.random.randint(1, 5)
        self.worker_count = worker or np.random.randint(2, 6)
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

    def set_cell_property(self, target, count=1, coordinates=None):
        if not coordinates:
            while True:
                y, x = np.random.randint(0, self.height - 1), np.random.randint(
                    0, self.width - 1
                )
                if (y, x) not in self.used:
                    break
        else:
            y, x = coordinates
        self.board[self.CELL.index(target), y, x] = count
        self.used.append((y, x))
        return y, x

    def set_worker_position(self, target, count, coordinates=None):
        y, x = self.set_cell_property(target, count, coordinates)
        return Worker(target, y, x, count)

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
                self.set_worker_position("worker_A", i + 1, coordinate)
                for i, coordinate in enumerate(worker_A)
            ]
        else:
            self.workers_A = [
                self.set_worker_position("worker_A", i + 1)
                for i in range(self.worker_count)
            ]

        if worker_B:
            assert self.worker_count == len(worker_B), "worker_B input error"
            self.workers_B = [
                self.set_worker_position("worker_B", i + 1, coordinate)
                for i, coordinate in enumerate(worker_B)
            ]
        else:
            self.workers_B = [
                self.set_worker_position("worker_B", i + 1)
                for i in range(self.worker_count)
            ]

        self.update_blank()
        return self.board

    def compile_layers(self, *layers):
        return np.sum([self.board[self.CELL.index(layer)] for layer in layers], axis=0)

    def is_movable(self, worker, action):
        pass

    def is_buildable(self, worker, action):
        pass

    def is_breakable(self, worker, action):
        pass

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
        direction = self.get_direction(action)
        y, x = np.array([worker.y, worker.x]) + direction
        if "move" in self.ACTIONS[action] and self.is_movable(worker, action):
            self.board[self.CELL.index(worker.name), worker.y, worker.x] = 0
            self.board[
                self.CELL.index(worker.name),
                y,
                x,
            ] = worker.num
            worker.move(y, x)

        elif "build" in self.ACTIONS[action] and self.is_buildable(worker, action):
            self.board[
                self.CELL.index(f"rampart_{worker.team}"),
                y,
                x,
            ] = 1
            worker.build(y, x)

        elif "break" in self.ACTIONS[action] and self.is_breakable(worker, action):
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

    def update_open_position(self):
        pass

    def calculate_score(self):
        self.score_A = np.sum(
            self.board[self.CELL.index("castle")]
            * self.compile_layers("position_A", "open_position_A")
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_A += np.sum(
            (1 - self.board[self.CELL.index("castle")])
            * self.compile_layers("position_A", "open_position_A")
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_A += np.sum(
            self.board[self.CELL.index("rampart_A")] * self.SCORE_MULTIPLIER["rampart"]
        )
        self.score_B = np.sum(
            self.board[self.CELL.index("castle")]
            * self.compile_layers("position_B", "open_position_B")
            * self.SCORE_MULTIPLIER["castle"]
        )
        self.score_B += np.sum(
            (1 - self.board[self.CELL.index("castle")])
            * self.compile_layers("position_B", "open_position_B")
            * self.SCORE_MULTIPLIER["position"]
        )
        self.score_B += np.sum(
            self.board[self.CELL.index("rampart_B")] * self.SCORE_MULTIPLIER["rampart"]
        )
        print(f"score_A:{self.score_A}, score_B:{self.score_B}")

    def get_reward(self, success_actions):
        pass

    def is_done(self):
        if self.turn >= self.end_turn:
            self.done = True

    def step(self, actions):
        assert self.worker_count == len(actions), "input error"
        success_actions = all(
            [
                self.worker_action(worker, action)
                for worker, action in zip(
                    self.workers_A if self.current_player > 0 else self.workers_B,
                    actions,
                )
            ]
        )
        self.current_player = -self.current_player
        self.turn += 1
        self.calculate_score()
        reward = self.get_reward(success_actions)
        self.is_done()
        return self.board, reward, self.done, {}

    def render(self):
        view = [["" for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                view[y][x] = [
                    self.CELL[i]
                    for i, item in enumerate(self.board[:, y, x])
                    if item >= 1
                ]

        view = np.array(view)
        # print(view)

        pygame.init()
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

                if cellInfo == "castle":
                    color = YELLOW
                elif cellInfo == "worker_A":
                    color = RED
                elif cellInfo == "worker_B":
                    color = BLUE
                elif cellInfo == "pond":
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

observation = env.reset()
done = False

# while not done:
#     env.render()

#     action = int(input("Choose an action (0-8): "))
#     observation, reward, done, _ = env.step(action)

#     if reward == -10:
#         print("Invalid move. Try again.")

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()

print(f"width:{env.width}, height:{env.height}, workers:{env.worker_count}")

while not done:
    env.render()

    print(f"input team A actions (need {env.worker_count} input) : ")
    actions = [int(input()) for _ in range(env.worker_count)]
    observation, reward, done, _ = env.step(actions)

    print(f"input team B actions (need {env.worker_count} input) : ")
    actions = [int(input()) for _ in range(env.worker_count)]
    observation, reward, done, _ = env.step(actions)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

# env.render()
