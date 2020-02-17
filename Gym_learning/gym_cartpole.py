import gym
from DQN1 import Deep_Q_Network
import numpy as np

env = gym.make("CartPole-v0")
env = env.unwrapped

numbers_action = env.action_space.n
numbers_state = env.observation_space.shape[0]

# print(env.action_space.n)
# print(env.observation_space.shape[0])


dqn = Deep_Q_Network(numbers_action, numbers_state, )

total_steps = 0
ep_reward = []
l1 = []
ep_reward2=[]
for i in range(500):
    s = env.reset()

    ep_r = 0

    while True:
        # env.render()
        action = dqn.choose_action(s)
        s_n, reward, done, information = env.step(action)

        x, x_dot, theta, theta_dot = s_n
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward= r1 + r2
        # print("reward:",reward)
        dqn.store_data(s, action, reward, s_n)

        ep_r += reward

        if total_steps > 500:
            r, l = dqn.learn()
            l1.append(l)
            ep_reward2.append(r)
            # print("loss",l)
        if done:
            print('episode: ', i,
                  'ep_r: ', round(ep_r, 2))
            ep_reward.append(ep_r)
            break

        s = s_n

        total_steps += 1

np.savetxt('reward.txt', ep_reward, fmt='%.8f')
np.savetxt('reward2.txt', ep_reward2, fmt='%.8f')
np.savetxt('loss.txt', l1, fmt='%.8f')
