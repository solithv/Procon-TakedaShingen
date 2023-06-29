import gymnasium as gym
import numpy as np
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


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

    def move(self, x, y):
        if self.x - 1 <= x <= self.x + 1 and self.y - 1 <= y <= self.y + 1:
            self.x = x
            self.y = y
            self.action_log.append("move", (x, y))
            self.is_action = True
            return True
        else:
            return False

    def build(self, x, y):
        self.action_log.append("build", (x, y))
        self.is_action = True

    def break_(self, x, y):
        self.action_log.append("break", (x, y))
        self.is_action = True

    def turn_init(self):
        self.is_action = False


class Game(gym.Env):
    CELL_SIZE = 100
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

    def __init__(self, end_turn=10, width=None, height=None, worker=None, pond=None):
        super().__init__()
        self.end_turn = end_turn
        self.width = width or np.random.randint(11, 25)
        self.height = height or np.random.randint(11, 25)
        self.worker_count = worker or np.random.randint(2, 6)
        self.pond_count = pond or np.random.randint(1, 5)
        self.current_player = 1
        self.board = np.zeros((len(self.CELL), self.height, self.width))
        self.score_A, self.score_B = 0, 0
        self.turn = 0
        self.done = False

        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.CELL), self.height, self.width),
            dtype=np.int8,
        )
        self.reward_range = [np.NINF, np.inf]
        self.window_size = max(self.width, self.height) * self.CELL_SIZE

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
        return Worker(target, x, y, count)

    def update_blank(self):
        self.board[0] = 1 - self.board[1:].any(axis=0)

    def reset(self, castle=None, pond=None, worker_A=None, worker_B=None):
        self.current_player = 1
        self.turn = 0
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

        elif "build" in self.ACTIONS[action] and self.is_buildable(worker, action):
            self.board[
                self.CELL.index(f"rampart_{worker.team}"),
                y,
                x,
            ] = 1

        elif "break" in self.ACTIONS[action] and self.is_breakable(worker, action):
            if self.board[self.CELL.index("rampart_A"), y, x]:
                self.board[self.CELL.index("rampart_A"), y, x] = 0
            else:
                self.board[self.CELL.index("rampart_B"), y, x] = 0

        self.update_blank()

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

    def is_done(self):
        if not self.turn >= self.end_turn:
            self.done = True

    def step(self, actions):
        assert self.worker_count == len(actions), "input error"
        for worker, action in zip(
            self.workers_A if self.current_player > 0 else self.workers_B, actions
        ):
            self.worker_action(worker, action)
        self.current_player = -self.current_player
        self.turn += 1
        self.calculate_score()
        self.is_done()

    def render(self):
        view = [["" for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                view[y][x] = [
                    self.CELL[i]
                    for i, item in enumerate(self.board[:, y, x])
                    if item >= 1
                ]
        print(np.array(view))


env = Game()

observation = env.reset()
done = False

while not done:
    env.render()

    action = int(input("Choose an action (0-8): "))
    observation, reward, done, _ = env.step(action)

    if reward == -10:
        print("Invalid move. Try again.")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

print(f"width:{env.width}, height:{env.height}, workers:{env.worker_count}")
# env.render()
print(env.board[0])
print(env.compile_layers("pond", "worker_A", "worker_B"))
