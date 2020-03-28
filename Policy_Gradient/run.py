import gym,cv2
import matplotlib.pyplot as plt



import gym
import numpy as np
from Policy_Gradient import PolicyGradient


DISPLAY_REWARD_THRESHOLD = 400 # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
# env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

pl=PolicyGradient(env.observation_space.shape[0],env.action_space.n)
total_reward=[]
total_loss=[]
for j in range(10000):
    state=env.reset()
    i=0
    while 1:
        if RENDER:
            env.render()
        action=pl.choose_action(state)
        next_state,reward,done,info=env.step(action)
        pl.store_data(state,action,reward)
        # print(done)
        i+=1
        if done:
        #     print(pl.ep_states,
        # pl.ep_actions,
        # pl.ep_rewards)
            ep_rs_sum = sum(pl.ep_rewards)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering

            print("episode:", j, "  reward:", int(running_reward))
            print(i)
            ep_total_reward=sum(pl.ep_rewards)
            r,l=pl.learn()
            total_reward.append(r)
            total_reward.append(l)
            print("done")
            break
        state=next_state

np.savetxt('reward.txt', total_reward, fmt='%0.8f')
np.savetxt('loss.txt', total_loss, fmt='%0.8f')

