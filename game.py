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
    ):
        self.team = team
        self.x = x
        self.y = y
        self.is_action = False
        self.action_log = []

    def move(self, x, y):
        if self.x - 1 <= x <= self.x + 1 and self.y - 1 <= y <= self.y + 1:
            self.x = x
            self.y = y
            self.action_log.append("move", (x, y))
            return True
        else:
            return False

    def build(self, x, y):
        self.action_log.append("build", (x, y))

    def break_(self, x, y):
        self.action_log.append("build", (x, y))


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

    def __init__(self):
        super().__init__()
        self.width = random.randint(11, 25)
        self.height = random.randint(11, 25)
        self.worker_count = random.randint(2, 6)
        self.current_player = 1
        self.board = np.dstack(
            [
                np.ones((self.width, self.height)),
                np.zeros((self.width, self.height, len(self.CELL) - 1)),
            ]
        )
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.width, self.height, len(self.CELL)),
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
        self.board[x][y][self.CELL.index(target)] = count
        if any(self.board[x][y][1:]):
            self.board[x][y][0] = 0
        self.used.append((x, y))
        return x,y

    def set_worker_position(self, target, count):
        x,y=self.set_cell_property(target,count)
        return Worker(target, x, y)

    def reset(self):
        self.current_player = 1
        self.board = np.dstack(
            [
                np.ones((self.width, self.height)),
                np.zeros((self.width, self.height, len(self.CELL) - 1)),
            ]
        )
        pond_count = np.random.randint(1, 5)
        self.used = []
        self.set_cell_property("castle")
        [self.set_cell_property("pond") for _ in range(pond_count)]
        [self.set_worker_position("worker_A", i + 1) for i in range(self.worker_count)]
        [self.set_worker_position("worker_B", i + 1) for i in range(self.worker_count)]
        return self.board

    def step(self, action):
        pass

    def render(self):
        view = [["" for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                view[y][x] = [
                    self.CELL[i] for i, item in enumerate(self.board[x][y]) if item >= 1
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
env.render()
