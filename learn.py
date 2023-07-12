import gymnasium as gym
import pygame
import rl
import tensorflow as tf

from game import Game

env = Game()

observation = env.reset()
done = False

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

env.render()
