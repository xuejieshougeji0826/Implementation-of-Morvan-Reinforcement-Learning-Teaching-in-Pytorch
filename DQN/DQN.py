import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

np.random.seed(1)
torch.manual_seed(1)


class Net(nn.Module):

    def __init__(self,
                 numbers_actions,
                 numbers_features,
                 lr=0.01,
                 reward_decay=0.9,
                 epsilon_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,

                 ):
        super(Net, self).__init__()
        self.numbers_actions = numbers_actions
        self.numbers_features = numbers_features
        # print(type(self.numbers_features), type(self.numbers_actions))
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon_greedy = epsilon_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # learning step
        self.counter = 0

        # numbers means 1. state, 2. action 3. reward 4. state_next
        self.memory = np.zeros((self.memory_size, self.numbers_features * 2 + 2))

        # two net, target/evaluate net
        self.lay_dense = nn.Sequential((nn.Linear(self.numbers_features, 20, bias=True)), nn.ReLU(True))
        self.output = nn.Sequential((nn.Linear(20, numbers_actions, bias=True)))

    def forward(self, x):
        # print("s:",x)
        x = self.lay_dense(x)
        x = self.output(x)

        return x


class Deep_Q_Network(Net):
    def __init__(self, numbers_actions, numbers_features, ):

        super(Deep_Q_Network, self).__init__(numbers_actions, numbers_features)

        self.eval_net = Net(self.numbers_actions, self.numbers_features)
        self.target_net = self.eval_net

        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        random_number = np.random.uniform()
        state = torch.FloatTensor(state)
        if random_number < self.epsilon_greedy:
            actions_value = self.forward(state)
            action = torch.argmax(actions_value)
            # print(actions_value)
            # action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
            # print("1")
            # print(action.shape)
            # action=action.numpy()
            # print(action)
            # print("ashpe:",action.shape)
        else:
            action = np.random.randint(0, self.numbers_actions)
            # action = torch.tensor(action)
        # print(action)
        # action=torch.from_numpy(action)
        return action

    def store_data(self, state, action, reward, state_next):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state, action, reward, state_next))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # change parameters
        if self.counter % self.replace_target_iter == 0:
            self.target_net = self.eval_net
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :self.numbers_features])
        batch_action =torch.FloatTensor(batch_memory[:, self.numbers_features:self.numbers_features + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.numbers_features+1:self.numbers_features + 2])
        print(batch_reward)
        batch_state_next =torch.FloatTensor(batch_memory[:, -self.numbers_features:])

        q_evaluate = self.eval_net(batch_state).gather(1,batch_action.long())

        q_next=self.target_net(batch_state_next).detach()
        # print(q_next)
        # print(q_next.max(1))
        # print(q_next.max(1)[0])
        q_target=batch_reward+self.gamma*q_next.max(1)[0]
        loss=self.loss_func(q_evaluate,q_target)
        # print("loss:",loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("loss:",loss.data.numpy())
        print("sum,",batch_reward.sum().numpy())
        return batch_reward.sum().numpy(), loss.data.numpy()

if __name__ == '__main__':
    q_val = Deep_Q_Network(2, 4)
    a=np.zeros((1,4))
    q_val.choose_action(a)
