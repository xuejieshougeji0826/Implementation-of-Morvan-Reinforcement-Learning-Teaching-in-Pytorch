from maze_env_new import Maze
from DQN import Deep_Q_Network
import numpy as np

if __name__ == "__main__":
    # maze game
    env = Maze()

    dqn = Deep_Q_Network(4, 2)
    print("storing the data...")
    for i in range(300):
        s = env.reset()
        # print(s)
        print("epsilon:", i)
        total_reward = []
        total_loss = []
        while 1:
            env.render()
            action = dqn.choose_action(s)
            s_n, reward, done = env.step(action)

            dqn.store_data(s, action, reward, s_n)
            if dqn.memory_counter > dqn.memory_size:
                reward_e, loss_e = dqn.learn()

                total_reward.append(reward_e)
                total_loss.append(loss_e)

                # print(total_reward)
            s = s_n
            if done:
                break
            dqn.counter += 1

    np.savetxt('reward.txt', total_reward, fmt='%0.8f')
    np.savetxt('loss.txt', total_loss, fmt='%0.8f')



