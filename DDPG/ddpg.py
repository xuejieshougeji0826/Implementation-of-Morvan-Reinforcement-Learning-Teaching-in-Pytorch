import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import gym
from itertools import count
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('Pendulum-v0')

# env.seed(1)
# torch.manual_seed(1)
# np.random.seed(1)


num_state = env.observation_space.shape[0]  # 3
num_action = env.action_space.shape[0]  # 1
action_max = env.action_space.high  # 2
action_min = env.action_space.low  # -2
# print(num_action)
# print(num_state)
# print(action_max, action_min)
replay_pool_size = 10000
train_iterations = 10
episodes = 100000
max_done_limit = 378

tau = 0.005
gamma = 0.95
noise = 0.05

render = False


class Actor(nn.Module):
    def __init__(self, num_state, num_action, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, num_action)

        self.max_action = torch.FloatTensor(max_action).to(device)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # print(F.tanh(x))
        x = self.max_action * torch.tanh(self.fc3(x))

        return x


class Critic(nn.Module):
    def __init__(self, num_state, num_action):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state + num_action, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, a, b):
        x = F.leaky_relu(self.fc1(torch.cat([a, b], 1)))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        return x


class Replay_buffer():
    # Expects tuples of (state, next_state, action, reward, done)

    def __init__(self, ):
        self.size = replay_pool_size
        self.pool = []
        self.ptr = 0

    def save_buffer(self, data):
        # print('1')
        if len(self.pool) == self.size:
            self.pool[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.size

        else:
            self.pool.append(data)
            
    def sample(self, batch_size):
        # print(self.size,batch_size)
        flags = np.random.randint(0, len(self.pool), size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in flags:
            a, b, c, d, e = self.pool[i]
            state.append(np.array(a, copy=False))
            next_state.append(np.array(b, copy=False))
            action.append(np.array(c, copy=False))
            reward.append(np.array(d, copy=False))
            done.append(np.array(e, copy=False))

        #   列向量
        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(
            done).reshape(-1, 1)


class DDPG(object):
    def __init__(self, num_state, num_action, max_action):
        self.actor = Actor(num_state, num_action, max_action).to(device)
        self.actor_target = Actor(num_state, num_action, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 0.001)

        self.critic = Critic(num_state, num_action)
        self.critic_target = Critic(num_state, num_action)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 0.001)

        self.replay_buffer = Replay_buffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_train = 0

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # print(state)
        # a = self.actor(state)
        # print(a)
        # print(a.cpu().data.numpy().flatten())

        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        for i in range(train_iterations):
            s, s_, a, r, d = self.replay_buffer.sample(512)
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(s_).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            #   compute target Q
            # print(next_state)
            # print(self.actor_target(next_state))
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()

            #   get current Q estimate
            current_Q = self.critic(state, action)

            #   compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            #   update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            #   compute target loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            #   update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # return actor_loss,critic_loss


def train():
    ddpg = DDPG(num_state, num_action, action_max)
    # ddpg.choose_action(np.ones((1, 3)))
    ep_reward = 0
    for i in range(episodes):
        state = env.reset()
        if render: env.render()
        for t in count():
            action = ddpg.choose_action(state)
            action = (action + np.random.normal(0, noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render: env.render()
            ddpg.replay_buffer.save_buffer((state, next_state, action, reward, np.float(done)))

            state = next_state
            # print("t",t)
            # print('done',done)
            if done or t >= max_done_limit:
                # print(done)
                # print(t)
                # if i % 5 == 0:
                # # print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_reward, j))
                # print("ep_i:",i,"ep_reward:",ep_reward,'j:',j)
                if i % 10 == 0:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_reward, t))
                ep_reward = 0
                break

        if len(ddpg.replay_buffer.pool) >= replay_pool_size - 1:
            # print(ddpg.replay_buffer.pool)
            ddpg.train()
            # print(a,c)


if __name__ == '__main__':
    train()
    # test()
