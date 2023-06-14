import gym
import numpy as np
import pygame

# ボードのサイズとセルのサイズを定義
BOARD_SIZE = 9
CELL_SIZE = 50
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE

# 色の定義
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BEIGE = (255, 240, 220)

class GoEnv(gym.Env):
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.action_space = gym.spaces.Discrete(BOARD_SIZE ** 2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

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

        self.current_player = -self.current_player
        return self.board, 0, False, {}

    def render(self):
        pygame.init()
        window_surface = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Go")

        pygame.display.update()

        # ボードの描画
        window_surface.fill(BEIGE)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                pygame.draw.rect(window_surface, BEIGE, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                font = pygame.font.Font(None, 20)
                text = font.render(f"{i * BOARD_SIZE + j}", True, BLACK)
                text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2))
                window_surface.blit(text, text_rect)

        # 石の描画
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 1:
                    pygame.draw.circle(window_surface, BLACK, (int((j + 0.5) * CELL_SIZE), int((i + 0.5) * CELL_SIZE)), int(CELL_SIZE * 0.4), 0)
                elif self.board[i][j] == -1:
                    pygame.draw.circle(window_surface, WHITE, (int((j + 0.5) * CELL_SIZE), int((i + 0.5) * CELL_SIZE)), int(CELL_SIZE * 0.4), 0)

        pygame.display.flip()

    def get_valid_moves(self):
        valid_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    valid_moves.append(i * BOARD_SIZE + j)
        return valid_moves


# メインプログラム
env = GoEnv()

observation = env.reset()
done = False

while not done:
    env.render()

    valid_moves = env.get_valid_moves()
    print("Valid moves:", valid_moves)

    action = int(input("Choose an action: "))
    observation, reward, done, _ = env.step(action)

    if reward == -10:
        print("Invalid move. Try again.")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

env.render()
