import gym

env = gym.make('CartPole-v1')  # make your environment!

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()  # render game screen
        print(observation)
        action = env.action_space.sample()  # this is random action. replace here to your algorithm!
        observation, reward, done, info = env.step(action)  # get reward and next scene
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        