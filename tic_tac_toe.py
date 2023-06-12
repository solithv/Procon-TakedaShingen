import gym
from gym import spaces
import numpy as np


class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.int32)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board

    def step(self, action):
        row = action // 3
        col = action % 3

        if self.board[row][col] != 0:
            return self.board, -10, True, {}  # Invalid move

        self.board[row][col] = self.current_player

        if self._check_win(self.current_player):
            return self.board, 10, True, {}  # Current player wins

        if np.count_nonzero(self.board) == 9:
            return self.board, 0, True, {}  # Draw

        self.current_player = -self.current_player
        return self.board, 0, False, {}

    def render(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 1:
                    print("X", end=" ")
                elif self.board[i][j] == -1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()

    def _check_win(self, player):
        for i in range(3):
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

    if reward == -10:
        print("Invalid move. Try again.")

env.render()
