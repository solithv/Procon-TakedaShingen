import gymnasium as gym
import numpy as np
import pygame
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


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

    def set_cell_property(self, target):
        while True:
            x, y = np.random.randint(0, self.width - 1), np.random.randint(
                0, self.height - 1
            )
            if (x, y) not in self.used:
                break
        self.board[x][y][self.CELL.index(target)] = 1
        if 1 in self.board[x][y][1:]:
            self.board[x][y][0] = 0
        self.used.append((x, y))

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
        [self.set_cell_property("worker_A") for _ in range(self.worker_count)]
        [self.set_cell_property("worker_B") for _ in range(self.worker_count)]
        return self.board

    def step(self, action):
        row = action // self.width
        col = action % self.height

        if self.board[row][col] != 0:
            return self.board, -10, True, {}  # Invalid move

        self.board[row][col] = self.current_player

        if self._check_win(self.current_player):
            return self.board, 10, True, {}  # Current player wins

        if np.count_nonzero(self.board) == self.width * self.height:
            return self.board, 0, True, {}  # Draw

        self.current_player = -self.current_player
        return self.board, 0, False, {}

    def render(self):
        view = [["" for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                view[y][x] = [
                    self.CELL[i] for i, item in enumerate(self.board[x][y]) if item == 1
                ]
        print(view)

    def __render(self):
        pygame.init()
        window_surface = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("n乗谷城")

        window_surface.fill(WHITE)

        # ゲームボードの描画
        for i in range(1, self.width):
            pygame.draw.line(
                window_surface,
                BLACK,
                (i * self.CELL_SIZE, 0),
                (i * self.CELL_SIZE, self.window_size),
                3,
            )
        for i in range(1, self.height):
            pygame.draw.line(
                window_surface,
                BLACK,
                (0, i * self.CELL_SIZE),
                (self.window_size, i * self.CELL_SIZE),
                3,
            )

        # マーカーの描画
        for i in range(self.width):
            for j in range(self.height):
                if self.board[i][j] == 1:
                    pygame.draw.circle(
                        window_surface,
                        RED,
                        (
                            int((j + 0.5) * self.CELL_SIZE),
                            int((i + 0.5) * self.CELL_SIZE),
                        ),
                        int(self.CELL_SIZE * 0.4),
                        3,
                    )
                elif self.board[i][j] == -1:
                    pygame.draw.line(
                        window_surface,
                        BLACK,
                        (j * self.CELL_SIZE + 10, i * self.CELL_SIZE + 10),
                        ((j + 1) * self.CELL_SIZE - 10, (i + 1) * self.CELL_SIZE - 10),
                        3,
                    )
                    pygame.draw.line(
                        window_surface,
                        BLACK,
                        ((j + 1) * self.CELL_SIZE - 10, i * self.CELL_SIZE + 10),
                        (j * self.CELL_SIZE + 10, (i + 1) * self.CELL_SIZE - 10),
                        3,
                    )

        pygame.display.update()

    def _check_win(self, player):
        # for i in range(BOARD_SIZE):
        #     if np.all(self.board[i] == player) or np.all(self.board[:, i] == player):
        #         return True
        # if np.all(np.diag(self.board) == player) or np.all(
        #     np.diag(np.fliplr(self.board)) == player
        # ):
        #     return True
        return False


# メインプログラム
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

env.render()
