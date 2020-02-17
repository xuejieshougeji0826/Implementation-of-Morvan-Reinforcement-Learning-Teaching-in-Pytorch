import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

np.random.seed(1)
torch.manual_seed(1)


class Net(nn.Module):

    def __init__(self,
                 numbers_actions,
                 numbers_features,
                 lr=0.001,
                 reward_decay=0.9,
                 epsilon_greedy=0.9,
                 replace_target_iter=1000,
                 memory_size=5000,
                 batch_size=32,
                 e_greedy_increment=0.001,

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
        self.e_greedy_increment=e_greedy_increment
        # learning step
        self.counter = 0

        # numbers means 1. state, 2. action 3. reward 4. state_next
        self.memory = np.zeros((self.memory_size, self.numbers_features * 2 + 2))

        # two net, target/evaluate net
        self.lay_dense = nn.Linear(self.numbers_features, 10, bias=True)
        nn.init.normal_(self.lay_dense.weight,mean=0,std=0.3)
        nn.init.constant_(self.lay_dense.bias,val=0.1)
        self.output = nn.Linear(10, numbers_actions, bias=True)
        nn.init.normal_(self.output.weight,mean=0,std=0.3)
        nn.init.constant_(self.output.bias,val=0.1)
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

    def forward(self, x):
        # print("s:",x)
        x = F.leaky_relu(self.lay_dense(x))
        x = F.leaky_relu(self.output(x))

        return x


class Deep_Q_Network(Net):
    def __init__(self, numbers_actions, numbers_features, ):

        super(Deep_Q_Network, self).__init__(numbers_actions, numbers_features)

        self.eval_net = Net(self.numbers_actions, self.numbers_features)
        self.target_net = Net(self.numbers_actions, self.numbers_features)

        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        # self.loss_func = torch.mse_loss()

    def choose_action(self, state):
        random_number = np.random.uniform()
        state = torch.FloatTensor(state)
        if random_number < self.epsilon_greedy:
            actions_value = self.eval_net(state)
            action = torch.argmax(actions_value)
            # par = list(self.eval_net.named_parameters())
            # print("choose_action_para:",par[0],par[1])
            # print(actions_value)
            # action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
            # print("1")
            # print(action.shape)
            action=action.numpy()
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
        transition = np.hstack((state, [action, reward], state_next))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        # print(self.memory_counter)
        self.memory_counter += 1

    def learn(self):

        # change parameters
        if self.counter % self.replace_target_iter == 0:
            # self.target_net = self.eval_net.clone()
            # par2 = list(self.eval_net.named_parameters())
            # print("eval_before:", par2[0], par2[1])
            # par2 = list(self.target_net.named_parameters())
            # print("target_before:", par2[0], par2[1])
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
            # par2 = list(self.eval_net.named_parameters())
            # print("eval_replaced:", par2[0], par2[1])
            # par2 = list(self.target_net.named_parameters())
            # print("target_replaced:", par2[0], par2[1])

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :self.numbers_features])
        batch_action = torch.FloatTensor(batch_memory[:, self.numbers_features:self.numbers_features + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.numbers_features+1:self.numbers_features + 2].transpose()[0])
        # print(batch_reward)
        batch_state_next = torch.FloatTensor(batch_memory[:, -self.numbers_features:])


        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.numbers_features].astype(int)

        # q_evaluate = self.eval_net(batch_state).gather(1,batch_action.long())
        q_next= self.target_net(batch_state_next)

        q_eval = self.eval_net(batch_state)

        q_target = q_eval.clone().detach()
        # print(self.numbers_features)
        # print("batch_state,",batch_state)
        # print("q_target",q_target)
        # print("q_target.max(1)",q_next.max(1))
        # print("q_target.max(1)[0]",q_next.max(1)[0])
        # print("q_target[batch_index, eval_act_index]",q_next[batch_index, eval_act_index])
        # print("batch_reward",batch_reward )
        # print("self.gamma * q_target.max(1)[0]",self.gamma * q_next.max(1)[0])
        # print("batch_reward + self.gamma * q_target.max(1)[0])",batch_reward + self.gamma * q_next.max(1)[0])
        q_target[batch_index, eval_act_index] = batch_reward + (self.gamma * q_next.max(1)[0])
        # print("q_target", q_target)
        # print("q_eval", q_eval)
        # loss=q_eval-q_target
        # loss=loss.mean()
        # print(q_eval-q_target)
        loss1 = F.mse_loss(q_eval,q_target)
        loss2 = F.mse_loss(q_eval, q_target)
        # print("loss:",loss)
        # par=list(self.eval_net.named_parameters())
        # par2 = list(self.eval_net.named_parameters())
        # print("eval_before:", par2[0], par2[1])
        # par2 = list(self.target_net.named_parameters())
        # print("target_before:", par2[0], par2[1])


        # print("eval_before:",par[0],par[1])
        self.optimizer.zero_grad()
        loss2.backward()
        # nn.utils.clip_grad_norm(self.eval_net.parameters(), 10.0)
        self.optimizer.step()
        par1 = list(self.eval_net.named_parameters())
        # print("eval_after:",par1[0],par1[1])
        # print("loss:",loss.data.numpy())
        # print("sum,",batch_reward.sum().numpy())
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_greedy  else self.epsilon_greedy

        self.counter+=1
        # self.target_net.load_state_dict(self.eval_net.state_dict())

        # par2 = list(self.eval_net.named_parameters())
        # print("eval_replaced:", par2[0], par2[1])
        # par2 = list(self.target_net.named_parameters())
        # print("target_replaced:", par2[0], par2[1])
        # print("counter:",self.counter)
        return batch_reward.sum().numpy(), loss1.data.numpy()

if __name__ == '__main__':
    q_val = Deep_Q_Network(2, 4)
    a=np.zeros((1,4))
    q_val.choose_action(a)
