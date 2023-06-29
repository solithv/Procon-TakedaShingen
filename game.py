import gymnasium as gym
import numpy as np
import pygame
import random
import pprint

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


class Worker:
    def __init__(
        self,
        team,
        x,
        y,
        num,
    ):
        self.team = team
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
        "N": np.array([-1, 0]),
        "E": np.array([0, 1]),
        "S": np.array([1, 0]),
        "W": np.array([0, -1]),
    }

    def __init__(self):
        super().__init__()
        self.width = random.randint(11, 25)
        self.height = random.randint(11, 25)
        self.worker_count = random.randint(2, 6)
        self.pond_count = np.random.randint(1, 5)
        self.current_player = 1
        self.board = np.zeros((len(self.CELL), self.height, self.width))
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.CELL), self.height, self.width),
            dtype=np.int8,
        )
        self.window_size = max(self.width, self.height) * self.CELL_SIZE

    def set_cell_property(self, target, count=1):
        while True:
            x, y = np.random.randint(0, self.width - 1), np.random.randint(
                0, self.height - 1
            )
            if (x, y) not in self.used:
                break
        self.board[self.CELL.index(target), y, x] = count
        # if self.board[1:, y, x].any():
        #     self.board[0, y, x] = 0
        self.used.append((x, y))
        return x, y

    def set_worker_position(self, target, count):
        x, y = self.set_cell_property(target, count)
        return Worker(target, x, y, count)

    def update_blank(self):
        self.board[0] = 1 - self.board[1:].any(axis=0)

    def reset(self):
        self.current_player = 1
        self.board = np.zeros((len(self.CELL), self.height, self.width))
        self.used = []
        self.set_cell_property("castle")
        [self.set_cell_property("pond") for _ in range(self.pond_count)]
        self.workers_A = [
            self.set_worker_position("worker_A", i + 1)
            for i in range(self.worker_count)
        ]
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
        if "move" in self.ACTIONS[action]:
            if self.is_movable(worker, action):
                direction = self.get_direction(action)
                self.board[self.CELL.index(worker.team), worker.y, worker.x] = 0
                self.board[
                    self.CELL.index(worker.team),
                    worker.y + direction[0],
                    worker.x + direction[1],
                ] = worker.num

        elif "build" in self.ACTIONS[action]:
            if self.is_buildable(worker, action):
                direction = self.get_direction(action)

        elif "break" in self.ACTIONS[action]:
            if self.is_breakable(worker, action):
                direction = self.get_direction(action)

        self.update_blank()

    def step(self, action):
        pass

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
# env.render()
print(env.board[0])
print(env.compile_layers("pond", "worker_A", "worker_B"))
