import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tensorflow import keras


class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3, 3), dtype=np.float32
        )

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        return self.board

    def step(self, action):
        row = action // 3
        col = action % 3

        if self.board[row, col] != 0:
            return self.board, -1, True, {}  # Invalid move, penalize and end the game

        self.board[row, col] = self.current_player

        if self._check_winner(self.current_player):
            reward = 1  # Current player wins
            done = True
        elif np.count_nonzero(self.board) == 9:
            reward = 0  # Draw
            done = True
        else:
            reward = 0  # Game in progress
            done = False
            self.current_player = -self.current_player

        return self.board, reward, done, {}

    def _check_winner(self, player):
        # Check rows
        for row in range(3):
            if np.all(self.board[row, :] == player):
                return True

        # Check columns
        for col in range(3):
            if np.all(self.board[:, col] == player):
                return True

        # Check diagonals
        if np.all(np.diag(self.board) == player) or np.all(
            np.diag(np.fliplr(self.board)) == player
        ):
            return True

        return False


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99  # Decay rate for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.1  # Learning rate
        self.discount_factor = 0.99  # Discount factor for future rewards
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(
            loss="mse", optimizer=keras.optimizers.Adam(lr=self.learning_rate)
        )
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        target = self.model.predict(state)[0]
        if done:
            target[action] = reward
        else:
            next_q_values = self.model.predict(next_state)[0]
            target[action] = reward + self.discount_factor * np.max(next_q_values)

        self.model.fit(state, np.array([target]), epochs=1, verbose=0)


if __name__ == "__main__":
    env = TicTacToeEnv()
    agent = QLearningAgent(
        env.observation_space.shape[0] * env.observation_space.shape[1],
        env.action_space.n,
    )

    episodes = 100

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(
            state, [1, env.observation_space.shape[0] * env.observation_space.shape[1]]
        )

        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(
                next_state,
                [1, env.observation_space.shape[0] * env.observation_space.shape[1]],
            )
            agent.train(state, action, reward, next_state, done)
            state = next_state

        if episode % 10 == 0:
            print("Episode:", episode)

    # Test the agent
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(
            np.reshape(
                state,
                [1, env.observation_space.shape[0] * env.observation_space.shape[1]],
            )
        )
        state, reward, done, _ = env.step(action)
        env.render()

    print("Game Over")
