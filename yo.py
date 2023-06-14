import gym
import numpy as np
import pygame

# ゲームボードの大きさとセルの大きさを定義
BOARD_SIZE = 3
CELL_SIZE = 100
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE

# 色の定義
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.action_space = gym.spaces.Discrete(BOARD_SIZE**2)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int32
        )

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        return self.board

    def step(self, action):
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE

        if self.board[row][col] != 0:
            return self.board, -10, True, {}  # Invalid move

        self.board[row][col] = self.current_player

        if self._check_win(self.current_player):
            return self.board, 10, True, {}  # Current player wins

        if np.count_nonzero(self.board) == BOARD_SIZE**2:
            return self.board, 0, True, {}  # Draw

        self.current_player = -self.current_player
        return self.board, 0, False, {}

    def render(self):
        pygame.init()
        window_surface = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Tic Tac Toe")

        window_surface.fill(WHITE)

        # ゲームボードの描画
        for i in range(1, BOARD_SIZE):
            pygame.draw.line(
                window_surface,
                BLACK,
                (i * CELL_SIZE, 0),
                (i * CELL_SIZE, WINDOW_SIZE),
                3,
            )
            pygame.draw.line(
                window_surface,
                BLACK,
                (0, i * CELL_SIZE),
                (WINDOW_SIZE, i * CELL_SIZE),
                3,
            )

        # マーカーの描画
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 1:
                    pygame.draw.circle(
                        window_surface,
                        RED,
                        (int((j + 0.5) * CELL_SIZE), int((i + 0.5) * CELL_SIZE)),
                        int(CELL_SIZE * 0.4),
                        3,
                    )
                elif self.board[i][j] == -1:
                    pygame.draw.line(
                        window_surface,
                        BLACK,
                        (j * CELL_SIZE + 10, i * CELL_SIZE + 10),
                        ((j + 1) * CELL_SIZE - 10, (i + 1) * CELL_SIZE - 10),
                        3,
                    )
                    pygame.draw.line(
                        window_surface,
                        BLACK,
                        ((j + 1) * CELL_SIZE - 10, i * CELL_SIZE + 10),
                        (j * CELL_SIZE + 10, (i + 1) * CELL_SIZE - 10),
                        3,
                    )

        pygame.display.update()

    def _check_win(self, player):
        for i in range(BOARD_SIZE):
            if np.all(self.board[i] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(
            np.diag(np.fliplr(self.board)) == player
        ):
            return True
        return False


# メインプログラム
env = TicTacToeEnv()

observation = env.reset()
done = False

while not done:
    env.render()

    action = int(input("Choose an action (0-8): "))
    observation, reward, done, _ = env.step(action)
    print(observation)
    print(env.observation_space)

    if reward == -10:
        print("Invalid move. Try again.")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

env.render()
